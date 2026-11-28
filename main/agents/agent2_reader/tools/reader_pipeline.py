"""
Shared Multi-PDF Reader Pipeline for all execution mode Readers.
Provides:
  1. Multi-PDF path collection and resilient per-file ingestion (partial-failure handling).
  2. Exact-text deduplication across documents while preserving multi-source provenance.
  3. Strict preservation of primary SourceUnit context_hash for CitationAudit & ComparisonPlan.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Iterable, Optional

from core.coordination import AgentDecision
from core.pdf_ingest import is_parser_error_text
from core.state import SourceUnit, compute_content_hash
from utils.logger import logger


TextCleaner = Callable[[str], str]
PdfChunkLoader = Callable[[str], list[dict[str, Any]]]


def collect_unique_pdf_paths(
    pdf_path: Optional[str],
    pdf_paths: Optional[Iterable[Any]] = None,
) -> list[str]:
    """Returns deduplicated PDF paths in deterministic order."""
    ordered: list[str] = []
    raw_items: list[Any] = []
    if pdf_path:
        raw_items.append(pdf_path)
    if pdf_paths:
        raw_items.extend(pdf_paths)
    for item in raw_items:
        if isinstance(item, str) and item.strip() and item not in ordered:
            ordered.append(item)
    return ordered


def merge_or_append_unit(
    unit: SourceUnit,
    units_by_content_hash: dict[str, SourceUnit],
    ordered_units: list[SourceUnit],
) -> bool:
    """
    Appends `unit` if its normalized content is new, or merges its provenance into the
    existing identical-text SourceUnit. Returns True when merged as a duplicate.
    """
    c_hash = compute_content_hash(unit.text)
    existing = units_by_content_hash.get(c_hash)
    if existing is not None:
        existing.merge_provenance(unit)
        return True
    units_by_content_hash[c_hash] = unit
    ordered_units.append(unit)
    return False


def build_source_units_from_docs(
    retrieved_docs: list[dict[str, Any]],
    *,
    limit: int = 16,
    text_cleaner: Optional[TextCleaner] = None,
    deduplicate: bool = True,
) -> tuple[list[SourceUnit], dict[str, SourceUnit], set[int], int, int]:
    """
    Converts ranked retrieved chunks into SourceUnits with optional exact-text deduplication
    and multi-source provenance preservation.
    Returns (ordered_units, units_by_content_hash, pages_covered, merged_duplicates, next_offset).
    """
    ordered: list[SourceUnit] = []
    by_hash: dict[str, SourceUnit] = {}
    pages_covered: set[int] = set()
    merged_count = 0
    curr_offset = 0

    for idx, doc in enumerate((retrieved_docs or [])[:limit]):
        if not isinstance(doc, dict):
            continue
        raw_text = str(doc.get("text") or doc.get("content") or "").strip()
        if is_parser_error_text(raw_text):
            continue
        clean_text = text_cleaner(raw_text) if text_cleaner else raw_text
        if not clean_text or is_parser_error_text(clean_text):
            continue

        doc_id = str(doc.get("doc_id") or doc.get("title") or f"doc_{idx + 1}")
        raw_page = doc.get("page_no") if doc.get("page_no") is not None else doc.get("page", 1)
        try:
            page_no = int(raw_page) if raw_page is not None and not isinstance(raw_page, bool) else 1
        except (TypeError, ValueError):
            page_no = 1
        pages_covered.add(page_no)

        orig_offset = doc.get("character_offset")
        if orig_offset is None:
            orig_offset = doc.get("start_offset")
        if orig_offset is None:
            orig_offset = doc.get("offset")
        unit_offset = int(orig_offset) if orig_offset is not None else curr_offset

        extra_meta = {
            k: doc[k]
            for k in ("latex", "equation", "equation_number", "variable_definitions", "domain_constraints", "equation_components", "table_metadata", "diagram_metadata", "figure_id", "ocr_engine", "ocr_confidence")
            if k in doc and doc[k] is not None
        }
        unit = SourceUnit(
            id=str(doc.get("id") or f"{doc_id}_p{page_no}_u{idx + 1}"),
            text=clean_text,
            doc_id=doc_id,
            page_no=page_no,
            bbox=doc.get("bbox"),
            character_offset=unit_offset,
            provenance=list(doc.get("provenance") or []),
            unit_type=str(doc.get("unit_type") or "text"),
            document_sha256=str(doc.get("document_sha256") or ""),
            page_rotation=int(doc.get("page_rotation") or 0),
            coordinate_space=str(doc.get("coordinate_space") or "pdf_points_top_left"),
            extractor_version=str(doc.get("extractor_version") or ""),
            source_path=doc.get("source_path"),
            metadata=extra_meta,
        )
        if deduplicate:
            if merge_or_append_unit(unit, by_hash, ordered):
                merged_count += 1
        else:
            ordered.append(unit)
        curr_offset = unit_offset + len(clean_text) + 1

    return ordered, by_hash, pages_covered, merged_count, curr_offset


def ingest_pdf_fallbacks(
    pdf_paths: list[str],
    *,
    ordered_units: list[SourceUnit],
    units_by_content_hash: dict[str, SourceUnit],
    pages_covered: set[int],
    curr_offset: int,
    chunk_loader: PdfChunkLoader,
    text_cleaner: Optional[TextCleaner] = None,
    max_units_per_pdf: int = 10,
    deduplicate: bool = True,
) -> tuple[list[dict[str, str]], int, int]:
    """
    Resiliently ingests multiple PDFs when retrieved_docs is empty (or needs direct fallback).
    A corrupted or unreadable PDF is recorded in `failed_pdfs` without discarding units
    extracted from remaining valid PDFs.
    Returns (failed_pdfs, merged_duplicates, next_offset).
    """
    failed_pdfs: list[dict[str, str]] = []
    merged_count = 0

    for path in pdf_paths:
        if not path or not os.path.exists(str(path)):
            failed_pdfs.append({"path": str(path), "error": "file_not_found"})
            continue
        try:
            chunks = chunk_loader(str(path))
            if not chunks:
                failed_pdfs.append({"path": str(path), "error": "no_extractable_text"})
                continue
        except Exception as exc:
            logger.warning("[ReaderPipeline] Failed to parse PDF '%s': %s", path, exc)
            failed_pdfs.append({"path": str(path), "error": str(exc)})
            continue

        fallback_doc_id = os.path.splitext(os.path.basename(str(path)))[0]
        for idx, ch in enumerate(chunks[:max_units_per_pdf]):
            if isinstance(ch, SourceUnit):
                raw_text = ch.text
                doc_id = ch.doc_id or fallback_doc_id
                page_no = ch.page_no
                uid = ch.id or f"u_{len(ordered_units) + 1}"
                bbox = ch.bbox
                orig_offset = ch.character_offset
            elif isinstance(ch, dict):
                raw_text = str(ch.get("text") or ch.get("content") or "").strip()
                doc_id = str(ch.get("doc_id") or fallback_doc_id)
                try:
                    page_no = int(ch.get("page_no") or ch.get("page") or 1)
                except (TypeError, ValueError):
                    page_no = 1
                uid = str(ch.get("id") or f"{doc_id}_p{page_no}_u{idx + 1}")
                bbox = ch.get("bbox")
                orig_offset = ch.get("character_offset") or ch.get("start_offset") or ch.get("offset")
            else:
                continue

            clean_text = text_cleaner(raw_text) if text_cleaner else raw_text.strip()
            if not clean_text or is_parser_error_text(clean_text):
                continue
            pages_covered.add(page_no)
            unit_offset = int(orig_offset) if orig_offset is not None else curr_offset
            fb_meta = (
                {
                    k: ch[k]
                    for k in ("latex", "equation", "equation_number", "variable_definitions", "domain_constraints", "equation_components", "table_metadata", "diagram_metadata", "figure_id", "ocr_engine", "ocr_confidence")
                    if k in ch and ch[k] is not None
                }
                if isinstance(ch, dict)
                else dict(getattr(ch, "metadata", None) or {})
            )
            unit = SourceUnit(
                id=uid,
                text=clean_text,
                doc_id=doc_id,
                page_no=page_no,
                bbox=bbox,
                character_offset=unit_offset,
                unit_type=str(ch.get("unit_type") if isinstance(ch, dict) else getattr(ch, "unit_type", "text")) or "text",
                document_sha256=str(ch.get("document_sha256") if isinstance(ch, dict) else getattr(ch, "document_sha256", "")) or "",
                page_rotation=int((ch.get("page_rotation") if isinstance(ch, dict) else getattr(ch, "page_rotation", 0)) or 0),
                source_path=str(path),
                metadata=fb_meta,
            )
            if deduplicate:
                if merge_or_append_unit(unit, units_by_content_hash, ordered_units):
                    merged_count += 1
            else:
                ordered_units.append(unit)
            curr_offset = unit_offset + len(clean_text) + 1

    return failed_pdfs, merged_count, curr_offset


_QUERY_TABLE_REF_RE = re.compile(
    r"\b(?:table|tab\.?|bảng|bang)\s*([A-Z]?\d+(?:[.-]\d+)*)\b",
    re.IGNORECASE,
)
_CROSS_REF_TABLE_RE = re.compile(
    r"(?:see|shown\s+in|reported\s+in|presented\s+in|listed\s+in|summarized\s+in|"
    r"detailed\s+in|refer\s+to|as\s+in|given\s+in|found\s+in|"
    r"xem(?:\s+chi\s+tiết)?(?:\s+(?:ở|trong|tại))?|"
    r"trình\s+bày\s+(?:ở|trong|tại)|nêu\s+(?:ở|trong|tại)|cho\s+(?:ở|trong|tại)|"
    r"liệt\s+kê\s+(?:ở|trong|tại)|tham\s+khảo)\s+"
    r"(?:table|tab\.?|bảng|bang)\s*([A-Z]?\d+(?:[.-]\d+)*)\b",
    re.IGNORECASE,
)
_TABLE_DECISION_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "that", "this", "what", "which", "where",
        "when", "how", "are", "was", "were", "does", "did", "can", "into", "about",
        "table", "tab", "bảng", "bang", "see", "xem", "trong", "của", "cho", "với",
        "các", "những", "một", "bao", "nhiêu", "giá", "trị", "value", "values",
    }
)


def _content_tokens(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-zA-ZÀ-ỹ0-9_-]{3,}", (text or "").lower())
        if tok not in _TABLE_DECISION_STOPWORDS
    }


_TABLE_SPECIFIC_COLUMNS = frozenset(
    {
        "p99", "p95", "p90", "p50", "wer", "cer", "bleu", "rouge", "f1",
        "ndcg", "mrr", "qps", "tps", "vram", "co2", "rmse", "mae", "mse", "fps",
    }
)
_TABLE_METRIC_ALIASES: dict[str, set[str]] = {
    "latency": {"latency", "độ trễ", "ms", "msec", "p99", "p95", "p50"},
    "độ trễ": {"latency", "độ trễ", "ms", "msec", "p99", "p95", "p50"},
    "accuracy": {"accuracy", "acc", "độ chính xác", "%"},
    "độ chính xác": {"accuracy", "acc", "độ chính xác", "%"},
    "precision": {"precision", "độ chính xác dương"},
    "recall": {"recall", "độ phủ"},
    "throughput": {"throughput", "qps", "tps", "tok/s", "tokens/s", "thông lượng"},
    "memory": {"memory", "vram", "ram", "gb", "mb", "bộ nhớ"},
    "vram": {"vram", "gpu memory", "memory", "gb", "mb"},
    "carbon": {"carbon", "co2", "kg"},
    "energy": {"energy", "power", "kwh", "kw", "watt", "điện năng"},
    "error": {"error", "wer", "cer", "loss", "mse", "rmse", "mae", "tỷ lệ lỗi"},
}
_TABLE_PHYSICAL_UNITS = frozenset({"ms", "gb", "mb", "kb", "kg", "kwh", "mg/l", "qps", "tps", "km/h", "%"})
_TABLE_ENTITY_STOPWORDS = frozenset(
    {
        "what", "which", "where", "when", "who", "how", "does", "did", "was", "were", "are", "is",
        "the", "for", "with", "from", "under", "over", "into", "about", "exact", "empirical",
        "table", "tab", "bảng", "bang", "page", "trang", "section", "figure", "fig", "hình",
        "production", "traffic", "load", "benchmark", "benchmarks", "evaluation", "experiment",
        "study", "paper", "report", "result", "results", "value", "values", "score", "scores",
        "rate", "ratio", "metric", "metrics", "measurement", "measurements", "system", "model",
        "method", "engine", "dataset", "split", "splits", "test", "train", "validation",
        "latency", "accuracy", "precision", "recall", "throughput", "memory", "vram", "carbon",
        "footprint", "energy", "error", "loss", "cost", "time", "speed", "flow", "drip",
        "irrigation", "cho", "biết", "giá", "trị", "trong", "của", "báo", "cáo", "tài", "liệu",
        "p99", "p95", "p90", "p50", "wer", "cer", "bleu", "rouge", "f1", "ndcg", "mrr", "qps",
        "tps", "co2", "rmse", "mae", "mse", "fps", "ms", "gb", "mb", "kg", "kwh",
    }
)


def _extract_table_query_requirements(query: str) -> dict[str, Any]:
    q = str(query or "").strip()
    if not q:
        return {"specific_columns": set(), "metric_groups": {}, "entities": set(), "units": set()}
    q_l = q.lower()
    words = re.findall(r"[a-zA-ZÀ-ỹ0-9_/-]+", q_l)
    word_set = set(words)

    specific_columns = {w for w in word_set if w in _TABLE_SPECIFIC_COLUMNS}
    metric_groups: dict[str, set[str]] = {}
    for key, aliases in _TABLE_METRIC_ALIASES.items():
        if re.search(rf"\b{re.escape(key)}\b", q_l):
            metric_groups[key] = aliases

    units: set[str] = set()
    if "%" in q:
        units.add("%")
    for u in _TABLE_PHYSICAL_UNITS:
        if u != "%" and re.search(rf"\b{re.escape(u)}\b", q_l):
            units.add(u)

    # Extract distinctive target row/entity identifiers (e.g. Hybrid-X, Vector-Q, Sparse-K, Model Orion)
    entities: set[str] = set()
    # 1. Hyphenated / versioned technical identifiers anywhere in query (e.g. Hybrid-X, GPT-3, Baseline-V1)
    for m in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*-[A-Za-z0-9]+\b", q):
        tok = m.group(0).lower()
        if tok not in _TABLE_ENTITY_STOPWORDS:
            entities.add(tok)
    # 2. Capitalized proper nouns not at position 0 (ignoring Table/Bảng/Section headers)
    for m in re.finditer(r"\b[A-ZĐ][A-Za-zÀ-ỹ0-9_-]{2,}\b", q):
        if m.start() == 0:
            continue
        tok = m.group(0).lower()
        if tok in _TABLE_ENTITY_STOPWORDS or tok in _TABLE_SPECIFIC_COLUMNS:
            continue
        # Skip single word right before table/section number (e.g. 'Table 4')
        entities.add(tok)

    return {
        "specific_columns": specific_columns,
        "metric_groups": metric_groups,
        "entities": entities,
        "units": units,
    }


def _table_region_matches_query(table_text: str, query: str) -> tuple[bool, list[str]]:
    reqs = _extract_table_query_requirements(query)
    if not any(reqs.values()):
        return True, []

    txt_l = (table_text or "").lower()
    missing: list[str] = []

    for col in sorted(reqs["specific_columns"]):
        if not re.search(rf"\b{re.escape(col)}\b", txt_l):
            missing.append(col)

    for m_name, aliases in sorted(reqs["metric_groups"].items()):
        if not any(
            (alias == "%" and "%" in txt_l)
            or (alias != "%" and re.search(rf"\b{re.escape(alias)}\b", txt_l))
            for alias in aliases
        ):
            missing.append(m_name)

    for ent in sorted(reqs["entities"]):
        if ent not in txt_l:
            missing.append(ent)

    for u in sorted(reqs["units"]):
        if u == "%":
            if "%" not in txt_l:
                missing.append("%")
        elif not re.search(rf"\b{re.escape(u)}\b", txt_l):
            missing.append(u)

    return len(missing) == 0, list(dict.fromkeys(missing))


def inspect_table_completeness(
    units: Iterable[dict[str, Any]],
    table_id: str,
    *,
    query: str = "",
) -> tuple[bool, list[str]]:
    """
    Checks whether `units` contains structured tabular data for `table_id` AND (when `query`
    specifies target rows/entities, columns/metrics, or units) whether that table actually
    covers the requested cells rather than being a same-numbered table with unrelated content
    or missing columns (e.g. `p50` present but `p99` absent).

    Returns `(is_complete, missing_cell_requirements)`.
    """
    tid = str(table_id or "").strip()
    if not tid:
        return False, []
    ref_pat = re.compile(rf"\b(?:table|tab\.?|bảng|bang)\s*{re.escape(tid)}\b", re.IGNORECASE)
    caption_pat = re.compile(rf"\b(?:table|tab\.?|bảng|bang)\s*{re.escape(tid)}\s*[:.\-–]", re.IGNORECASE)

    best_missing: list[str] | None = None

    for unit in units:
        if not isinstance(unit, dict):
            continue
        txt = str(unit.get("text") or unit.get("content") or "")
        if not txt.strip():
            continue
        meta = unit.get("table_metadata") if isinstance(unit.get("table_metadata"), dict) else {}
        caption = str(meta.get("caption") or "")
        u_type = str(unit.get("unit_type") or "text").lower()

        mentions_table = bool(ref_pat.search(caption) or ref_pat.search(txt))
        if not mentions_table:
            continue

        candidate_table_text: str | None = None
        pipe_lines = [ln for ln in txt.splitlines() if ln.count("|") >= 2]
        if u_type == "table" and (pipe_lines or re.search(r"\d+(?:[.,]\d+)?", txt)):
            candidate_table_text = f"{caption}\n{txt}".strip()
        elif len(pipe_lines) >= 2:
            candidate_table_text = f"{caption}\n{txt}".strip()
        else:
            cap_match = caption_pat.search(txt)
            if cap_match is not None:
                after_caption = txt[cap_match.end():]
                numeric_cells = re.findall(r"\b\d+(?:[.,]\d+)?%?\b", after_caption)
                non_caption_lines = [ln.strip() for ln in after_caption.splitlines() if ln.strip()]
                if len(numeric_cells) >= 2 or (
                    non_caption_lines and any(("\t" in ln or "  " in ln or "|" in ln) for ln in non_caption_lines) and numeric_cells
                ):
                    candidate_table_text = f"{caption}\n{txt[cap_match.start():]}".strip()

        if candidate_table_text is None:
            # Inline direct numeric measurement alongside Table N (NOT a 'see Table N' pointer!)
            for sent in re.split(r"(?<=[.!?])\s+|\n+", txt):
                if not ref_pat.search(sent) or _CROSS_REF_TABLE_RE.search(sent):
                    continue
                cleaned_no_tid = ref_pat.sub(" ", sent)
                if re.search(r"\b\d+(?:[.,]\d+)?\s*(?:%|ms|s|sec|mb|gb|kb|hz|mhz|ghz|bleu|f1|x)\b|\b\d+\.\d+\b", cleaned_no_tid, re.IGNORECASE):
                    candidate_table_text = sent
                    break

        if candidate_table_text is None:
            continue

        ok, missing_parts = _table_region_matches_query(candidate_table_text, query)
        if ok:
            return True, []
        if best_missing is None or len(missing_parts) < len(best_missing):
            best_missing = missing_parts

    return False, (best_missing or [])


def has_structured_table(units: Iterable[dict[str, Any]], table_id: str, *, query: str = "") -> bool:
    """
    Returns True if `units` already contains structured tabular content (or an inline numeric
    result directly answering) for `table_id` AND covers the row/column/metric/unit requested
    in `query` (when `query` is provided).
    """
    ok, _ = inspect_table_completeness(units, table_id, query=query)
    return ok


def extract_referenced_table_ids(query: str, units: Iterable[dict[str, Any]]) -> list[str]:
    """
    Extracts ordered table identifiers referenced either directly in `query`
    or via query-relevant cross-references inside `units` (e.g. 'see Table 4', 'xem Bảng 4').
    """
    ordered: list[str] = []
    q_text = str(query or "").strip()
    q_tokens = _content_tokens(q_text)

    for m in _QUERY_TABLE_REF_RE.finditer(q_text):
        tid = m.group(1).strip()
        if tid and tid.lower() not in {x.lower() for x in ordered}:
            ordered.append(tid)

    unit_list = [u for u in units if isinstance(u, dict)]
    for unit in unit_list:
        txt = str(unit.get("text") or unit.get("content") or "")
        if not txt.strip():
            continue
        for sent in re.split(r"(?<=[.!?])\s+|\n+", txt):
            matches = list(_CROSS_REF_TABLE_RE.finditer(sent))
            if not matches:
                continue
            sent_tokens = _content_tokens(sent)
            overlap = bool(q_tokens & sent_tokens)
            # Only trigger on cross-references relevant to the user's question (or when query is broad/short)
            if not overlap and len(q_tokens) >= 2 and len(unit_list) > 2:
                continue
            for m in matches:
                tid = m.group(1).strip()
                if tid and tid.lower() not in {x.lower() for x in ordered}:
                    ordered.append(tid)
    return ordered


def evaluate_reader_table_completeness(
    state: dict[str, Any],
    source_units: list[dict[str, Any]],
    *,
    allow_handoff: bool = True,
) -> tuple[AgentDecision | None, list[str]]:
    """
    Local deterministic check (zero extra LLM calls) for missing referenced tables in Reader output.
    Returns `(agent_decision_or_none, unresolved_table_labels)`:
      - When all referenced tables are already present -> `(None, [])` (zero unnecessary handoff).
      - When a referenced table is missing and retrieval may still fetch it -> `(AgentDecision, [])`.
      - When a referenced table is STILL missing after retrieval returned `no_new_evidence=True`
        -> `(None, ["Table <N>", ...])` so Summarizer/Fact-Checker can report unverified evidence cleanly.
    """
    if not allow_handoff or state.get("enable_reader_handoff") is False:
        return None, []

    query = str(state.get("resolved_query") or state.get("query") or "").strip()
    if not query:
        return None, []

    prior_units = [u for u in (state.get("source_units") or []) if isinstance(u, dict)]
    combined_units = [*(source_units or []), *prior_units]
    if not combined_units:
        return None, []

    referenced = extract_referenced_table_ids(query, combined_units)
    if not referenced:
        return None, []

    missing_entries: list[tuple[str, list[str]]] = []
    for tid in referenced:
        complete, missing_parts = inspect_table_completeness(combined_units, tid, query=query)
        if not complete:
            missing_entries.append((tid, missing_parts))
    if not missing_entries:
        return None, []

    unresolved_labels = [
        f"Table {tid} (missing: {', '.join(parts)})" if parts else f"Table {tid}"
        for tid, parts in missing_entries
    ]
    if state.get("no_new_evidence"):
        return None, unresolved_labels

    primary_tid, primary_missing = missing_entries[0]
    retry_count = int(state.get("retry_count") or 0)
    hint_terms = list(dict.fromkeys([*primary_missing, *sorted(_content_tokens(query))[:5]]))[:6]
    query_hint = " ".join(hint_terms).strip()
    focused_queries = [f"Table {primary_tid} Bảng {primary_tid}"]
    if query_hint:
        focused_queries.append(f"Table {primary_tid} {query_hint}")

    reason = (
        f"Referenced Table {primary_tid} lacks required cells ({', '.join(primary_missing)}) in currently read evidence"
        if primary_missing
        else f"Referenced Table {primary_tid} is not present in currently read evidence"
    )
    decision = AgentDecision(
        source_agent="reader",
        status="needs_followup",
        action="request_more_evidence",
        target_agent="retrieval",
        reason=reason,
        suggested_queries=focused_queries,
        focus=[f"Table {primary_tid}", *(focused_queries[1:2])],
        round_index=retry_count,
    )
    return decision, []
