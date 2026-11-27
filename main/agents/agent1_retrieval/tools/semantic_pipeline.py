"""
Shared Semantic Retrieval Pipeline for Hybrid & Gemini Modes.
Provides:
  1. Cross-lingual bilingual query bridging (Vietnamese <-> English) with zero-latency offline fallback.
  2. Page-diverse passage ordering to maximize multi-page evidence coverage.
  3. Table-aware priority boosting for numeric/table/comparison queries.
  4. Adaptive Dense Retrieval (Kaggle BGE-M3 primary, optional Gemini Embedding secondary)
     + Guarded MiniLM Cross-Encoder Reranking (preserving top-k Hybrid prefix).
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Iterable

import httpx

from agents.agent1_retrieval.tools.hybrid import reciprocal_rank_fusion
from agents.agent1_retrieval.tools.remote_dense import (
    DEFAULT_DENSE_RRF_K,
    DEFAULT_DENSE_WEIGHT,
    DenseRetrievalError,
    RemoteDenseRetriever,
)
from agents.agent1_retrieval.tools.remote_rerank import MAX_CANDIDATES, RerankError, rerank_passages
from core.config import get_run_settings
from core.pdf_figures import extract_pdf_figures
from core.token_usage import metered_post
from utils.bilingual import expand_bilingual_tokens
from utils.logger import logger


def extract_searchable_figure_chunks(pdf_paths: Iterable[str | None]) -> list[dict[str, Any]]:
    """
    Indexes PDF figures that carry meaningful captions or embedded axis/legend labels
    as searchable retrieval units (unit_type='figure') so data questions can locate
    relevant charts/diagrams even without explicitly naming 'Figure X'.
    """
    chunks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw_path in dict.fromkeys(p for p in pdf_paths if p):
        path = str(raw_path)
        if not path.lower().endswith(".pdf") or not os.path.isfile(path):
            continue
        try:
            figures = extract_pdf_figures(path)
        except Exception as exc:
            logger.debug("[SemanticPipeline] Skipping figure indexing for %s: %s", path, exc)
            continue
        for fig in figures:
            caption = (fig.get("caption") or "").strip()
            embedded = (fig.get("embedded_text") or "").strip()
            ocr_text = ""
            if fig.get("kind") == "scan" and not embedded:
                try:
                    import fitz
                    from core.pdf_ingest import compute_file_sha256, ocr_pdf_page_if_needed

                    sha = compute_file_sha256(path)
                    with fitz.open(path) as doc:
                        p_idx = int(fig["page_no"]) - 1
                        if 0 <= p_idx < len(doc):
                            ocr_res = ocr_pdf_page_if_needed(doc[p_idx], doc_sha256=sha, page_no=int(fig["page_no"]))
                            ocr_text = str(ocr_res.get("text") or "").strip()
                except Exception:
                    ocr_text = ""
            combined = " ".join(part for part in (caption, embedded, ocr_text) if part).strip()
            if len(combined) < 8:
                continue
            fid = fig["figure_id"]
            cid = f"{fid}:caption" if caption else (f"{fid}:ocr_scan" if ocr_text else f"{fid}:embedded_text")
            if cid in seen_ids:
                continue
            seen_ids.add(cid)
            chunks.append(
                {
                    "id": cid,
                    "chunk_id": cid,
                    "doc_id": fig["doc_id"],
                    "title": fig["doc_id"],
                    "page_no": fig["page_no"],
                    "page": fig["page_no"],
                    "text": combined[:1800],
                    "content": combined[:1800],
                    "sentences": [s for s in (caption, embedded, ocr_text) if s],
                    "source_path": fig["source_path"],
                    "is_selected_pdf": True,
                    "unit_type": "ocr_scan" if ocr_text and not embedded else "figure",
                    "figure_id": fid,
                    "bbox": list(fig.get("caption_bbox") or fig["bbox"]),
                }
            )
    return chunks


from utils.language import detect_corpus_language, is_vietnamese_text

_TABLE_QUERY_RE = re.compile(
    r"\b(?:table|tab\.?|bảng|bang|so sánh|so sanh|compare|comparison|metric|score|accuracy|f1|bleu|rouge|ndcg|mrr|recall|"
    r"kết quả|ket qua|results|số liệu|so lieu|bao nhiêu|bao nhieu|tỷ lệ|ty le|phần trăm|phan tram|"
    r"%|giá trị|gia tri|value|row|column|hàng|hang|cột|cot|validation|test\s+set|tập\s+test|tập\s+kiểm\s+tra)\b",
    re.IGNORECASE,
)
_EQUATION_QUERY_RE = re.compile(
    r"(?:\b(?:equation|eq\.?|formula|phương\s+trình|phuong\s+trinh|công\s+thức|cong\s+thuc|biểu\s+thức|bieu\s+thuc|"
    r"fraction|phân\s+số|phan\s+so|domain|miền\s+xác\s+định|điều\s+kiện\s+xác\s+định|derivative|đạo\s+hàm|stepsize)\b"
    r"|[=≈≤≥≠←→∑∏∫√αβγθλμπσΩΔ]|\\frac|\^[0-9a-zA-Z{])",
    re.IGNORECASE,
)
_EQUATION_NUM_REF_RE = re.compile(
    r"(?:(?:equation|eq\.?|phương\s+trình|phuong\s+trinh|công\s+thức|cong\s+thuc)\s*\(?(\d+(?:\.\d+)*[a-z]?)\)?|\((\d+(?:\.\d+)*[a-z]?)\))",
    re.IGNORECASE,
)
_TABLE_NUM_REF_RE = re.compile(r"\b(?:table|tab\.?|bảng|bang)\s*([A-Z]?\d+(?:[.-]\d+)*)\b", re.IGNORECASE)
_FIGURE_NUM_REF_RE = re.compile(r"\b(?:figure|fig\.?|hình|hinh|biểu\s+đồ|bieu\s+do)\s*([A-Z]?\d+(?:[.-]\d+)*)\b", re.IGNORECASE)
_MATH_VAR_TOKEN_RE = re.compile(r"(?:[A-Za-z][ππαβγθλσ]|[αβγθλμπσΩΔ]|[A-Za-z]_[A-Za-z0-9{]+|[A-Za-z]\^[A-Za-z0-9{]+)")


def extract_structural_search_subqueries(query: str) -> list[str]:
    """
    Produces targeted structural lookup queries for equation numbers (`Eq. (1)`),
    math variable symbols (`Qπ`, `α`, `d_k`), table IDs (`Table 2`), or figure IDs (`Figure 3`).
    """
    clean_q = (query or "").strip()
    if not clean_q:
        return []
    out: list[str] = []
    eq_nums = [m.group(1) or m.group(2) for m in _EQUATION_NUM_REF_RE.finditer(clean_q) if (m.group(1) or m.group(2))]
    math_vars = list(dict.fromkeys(_MATH_VAR_TOKEN_RE.findall(clean_q)))
    if eq_nums or math_vars:
        parts = [*(f"({n})" for n in eq_nums[:2]), *math_vars[:4]]
        if parts:
            out.append(" ".join(parts))
    tbl_nums = [m.group(1) for m in _TABLE_NUM_REF_RE.finditer(clean_q)]
    if tbl_nums:
        out.append(" ".join(f"Table {n} Bảng {n}" for n in tbl_nums[:2]))
    fig_nums = [m.group(1) for m in _FIGURE_NUM_REF_RE.finditer(clean_q)]
    if fig_nums:
        out.append(" ".join(f"Figure {n} Hình {n}" for n in fig_nums[:2]))
    return list(dict.fromkeys(q.strip() for q in out if q.strip()))[:2]


def is_cross_lingual(query: str, corpus_docs: Iterable[dict[str, Any]]) -> bool:
    """Returns True when query language (VI vs EN) differs from the sampled document pool."""
    q_vi = is_vietnamese_text(query)
    sample = list(corpus_docs)[:8]
    if not sample:
        return False
    doc_vi = detect_corpus_language(sample, default="en") == "vi"
    return q_vi != doc_vi


def build_cross_lingual_subqueries(
    query: str,
    corpus_docs: list[dict[str, Any]],
    *,
    llm_translator: Any | None = None,
) -> list[str]:
    """
    Produces 1-2 bridged cross-lingual sub-queries when query and document languages differ
    (covering all 4 VI/EN combinations plus unaccented Vietnamese) and structural equation/table hints.
    Always runs deterministic bilingual phrase mapping first (offline, <1ms); optionally adds
    one concise LLM translation when an online translator callable is supplied.
    """
    clean_q = (query or "").strip()
    if not clean_q:
        return []
    cross = is_cross_lingual(clean_q, corpus_docs)
    base_tokens = set(re.findall(r"[a-zA-ZÀ-ỹ0-9_.-]+", clean_q.lower()))
    expanded = expand_bilingual_tokens(clean_q, base_tokens)
    novel = [tok for tok in sorted(expanded - base_tokens) if len(tok) >= 2]

    extra: list[str] = []
    if novel and (cross or not corpus_docs):
        # Preserve proper nouns, math symbols, and numbers from original query alongside translated terms
        anchors = re.findall(r"\b(?:[A-Z][A-Za-z0-9_-]{1,}|\d+(?:[.,]\d+)?%?)\b", clean_q)
        math_vars = _MATH_VAR_TOKEN_RE.findall(clean_q)
        bridged = " ".join(dict.fromkeys([*anchors[:4], *math_vars[:3], *novel[:14]]))
        if bridged.strip() and bridged.strip().lower() != clean_q.lower():
            extra.append(bridged.strip())

    # Include direct structural sub-queries for equation numbers, math variables, or table/figure IDs
    for struct_q in extract_structural_search_subqueries(clean_q):
        if struct_q.lower() != clean_q.lower() and struct_q not in extra:
            extra.append(struct_q)

    if cross and llm_translator is not None:
        try:
            translated = str(llm_translator(clean_q) or "").strip()
            if translated and translated.lower() != clean_q.lower() and translated not in extra:
                extra.insert(0, translated[:300])
        except Exception as exc:
            logger.debug("[SemanticPipeline] Optional query translation skipped: %s", exc)

    return extra[:3]


def prefer_distinct_pages(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first hit per (doc_id, page_no) ahead of lower-ranked passages from the same page."""
    seen_pages: set[tuple[Any, Any]] = set()
    first_hits: list[dict[str, Any]] = []
    remaining: list[dict[str, Any]] = []
    for chunk in chunks:
        document = chunk.get("doc_id")
        page = chunk.get("page_no") or chunk.get("page")
        if document and page:
            key = (document, page)
            if key in seen_pages:
                remaining.append(chunk)
                continue
            seen_pages.add(key)
        first_hits.append(chunk)
    return first_hits + remaining


def boost_entities_tables_and_equations(
    query: str,
    candidates: list[dict[str, Any]],
    required_entities: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """
    Applies entity coverage, intact-table, equation-number/variable-symbol, and figure/OCR
    structural priority boosts while preserving RRF rank ordering.
    """
    if not candidates:
        return []
    clean_q = query or ""
    wants_table = bool(_TABLE_QUERY_RE.search(clean_q))
    wants_eq = bool(_EQUATION_QUERY_RE.search(clean_q))
    eq_nums = {
        (m.group(1) or m.group(2)).strip()
        for m in _EQUATION_NUM_REF_RE.finditer(clean_q)
        if (m.group(1) or m.group(2))
    }
    tbl_nums = {m.group(1).strip().lower() for m in _TABLE_NUM_REF_RE.finditer(clean_q)}
    fig_nums = {m.group(1).strip().lower() for m in _FIGURE_NUM_REF_RE.finditer(clean_q)}
    math_vars = [v.lower() for v in _MATH_VAR_TOKEN_RE.findall(clean_q)]

    if not required_entities and not wants_table and not wants_eq and not tbl_nums and not fig_nums:
        return candidates[:top_k]

    def _score(idx_doc: tuple[int, dict[str, Any]]) -> float:
        rank_idx, doc = idx_doc
        txt = str(doc.get("text") or doc.get("content") or "")
        txt_l = txt.lower()
        u_type = str(doc.get("unit_type") or "text").lower()
        ent_hits = sum(1 for e in required_entities if e and e.lower() in txt_l)
        bonus = 0.0

        if wants_table and u_type == "table":
            bonus += 0.25
        if tbl_nums and u_type == "table":
            caption_l = str((doc.get("table_metadata") or {}).get("caption") or txt[:120]).lower()
            if any(f"table {n}" in caption_l or f"bảng {n}" in caption_l or f"tab. {n}" in caption_l for n in tbl_nums):
                bonus += 0.55

        if wants_eq and u_type == "equation":
            bonus += 0.30
        if u_type == "equation" and (eq_nums or math_vars):
            eq_meta = doc.get("equation_metadata") or {}
            eq_num = str(doc.get("equation_number") or eq_meta.get("equation_number") or "").strip("() ")
            if eq_num and eq_num in eq_nums:
                bonus += 0.65
            var_hits = sum(1 for v in math_vars if v in txt_l)
            bonus += 0.20 * min(var_hits, 3)

        if fig_nums and u_type in {"figure", "ocr_scan"}:
            if any(f"figure {n}" in txt_l or f"fig. {n}" in txt_l or f"hình {n}" in txt_l for n in fig_nums):
                bonus += 0.50

        return (1.0 / (rank_idx + 1)) + 0.35 * ent_hits + bonus

    ranked = sorted(enumerate(candidates), key=_score, reverse=True)
    return [doc for _, doc in ranked[:top_k]]


def boost_entities_and_tables(
    query: str,
    candidates: list[dict[str, Any]],
    required_entities: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper delegating to `boost_entities_tables_and_equations`."""
    return boost_entities_tables_and_equations(query, candidates, required_entities, top_k)


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na > 0 and nb > 0 else 0.0


def _embed_gemini_batch(
    texts: list[str],
    *,
    api_key: str,
    model: str,
    client: httpx.Client,
    timeout: float = 20.0,
) -> list[list[float]]:
    """Embeds a small batch of texts via Gemini batchEmbedContents API."""
    clean_model = model if model.startswith("models/") else f"models/{model}"
    url = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:batchEmbedContents"
    requests_payload = [
        {"model": clean_model, "content": {"parts": [{"text": (t or " ")[:2000]}]}}
        for t in texts
    ]
    response = metered_post(
        client.post,
        url,
        provider="gemini",
        operation="embedding",
        protocol="gemini",
        headers={"x-goog-api-key": api_key},
        json={"requests": requests_payload},
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    embeddings = data.get("embeddings") or []
    vectors = [item.get("values") for item in embeddings if isinstance(item, dict) and isinstance(item.get("values"), list)]
    if len(vectors) != len(texts):
        raise DenseRetrievalError("gemini_embedding_count_mismatch")
    return vectors


def enhance_with_dense_and_rerank(
    query: str,
    sparse_docs: list[dict[str, Any]],
    active_pool: list[dict[str, Any]],
    *,
    top_k: int,
    http_client: httpx.Client,
    dense_retriever: RemoteDenseRetriever | None = None,
    enable_kaggle_dense: bool = False,
    enable_gemini_dense: bool = False,
    enable_rerank: bool = False,
    keep_prefix: int = 3,
    page_diverse: bool = True,
    dense_weight: float = DEFAULT_DENSE_WEIGHT,
    dense_rrf_k: int = DEFAULT_DENSE_RRF_K,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """
    Fuses sparse lexical candidates with dense multilingual retrieval (Kaggle BGE-M3 primary,
    optional Gemini embedding secondary), applies page diversity, and runs Guarded MiniLM
    reranking when available. Always fails open to sparse_docs on remote unavailability.
    """
    settings = get_run_settings()
    candidate_limit = 2 * top_k if page_diverse else top_k
    fused = list(sparse_docs)
    dense_trace: dict[str, Any] = {"status": "disabled"}

    # 1. Primary Dense: Kaggle Remote BGE-M3 Worker (/api/embed)
    if enable_kaggle_dense and active_pool:
        retriever = dense_retriever or RemoteDenseRetriever()
        try:
            if not settings.REMOTE_INFERENCE_URL:
                raise DenseRetrievalError("embedding_worker_not_configured")
            dense_docs, revision = retriever.rank(
                query,
                active_pool,
                top_k=candidate_limit,
                client=http_client,
                url=settings.REMOTE_INFERENCE_URL,
                model=settings.EMBEDDING_MODEL,
                timeout=float(settings.REMOTE_TIMEOUT_SECONDS),
            )
            fusion_limit = 2 * candidate_limit if page_diverse else top_k
            fused = reciprocal_rank_fusion(
                [fused, dense_docs] if fused else [dense_docs],
                k=dense_rrf_k,
                top_n=fusion_limit,
                weights=[1.0, dense_weight] if fused else [1.0],
            )
            dense_trace = {
                "status": "used",
                "backend": "kaggle_bge_m3",
                "model": settings.EMBEDDING_MODEL,
                "revision": revision,
            }
        except DenseRetrievalError as exc:
            dense_trace = {"status": "fallback", "backend": "kaggle_bge_m3", "reason": str(exc)}
            logger.warning("[SemanticPipeline] Kaggle dense fallback to sparse: %s", exc)

    # 2. Secondary Dense: Gemini Embedding API (when Kaggle dense wasn't used & Gemini dense is enabled)
    if dense_trace.get("status") != "used" and enable_gemini_dense and active_pool and settings.GEMINI_API_KEY:
        try:
            # Cap pool size per request to keep latency & API quota bounded
            bounded_pool = active_pool[:64]
            texts = [query] + [str(d.get("text") or d.get("content") or "") for d in bounded_pool]
            vectors = _embed_gemini_batch(
                texts,
                api_key=settings.GEMINI_API_KEY,
                model=settings.GEMINI_EMBEDDING_MODEL,
                client=http_client,
                timeout=min(float(settings.REMOTE_TIMEOUT_SECONDS), 25.0),
            )
            q_vec, doc_vecs = vectors[0], vectors[1:]
            scored = sorted(
                zip(bounded_pool, (_cosine(q_vec, v) for v in doc_vecs)),
                key=lambda item: item[1],
                reverse=True,
            )
            dense_docs = [doc for doc, _ in scored[:candidate_limit]]
            fusion_limit = 2 * candidate_limit if page_diverse else top_k
            fused = reciprocal_rank_fusion(
                [fused, dense_docs] if fused else [dense_docs],
                k=dense_rrf_k,
                top_n=fusion_limit,
                weights=[1.0, dense_weight] if fused else [1.0],
            )
            dense_trace = {
                "status": "used",
                "backend": "gemini_embedding",
                "model": settings.GEMINI_EMBEDDING_MODEL,
            }
        except Exception as exc:
            dense_trace = {"status": "fallback", "backend": "gemini_embedding", "reason": type(exc).__name__}
            logger.warning("[SemanticPipeline] Gemini dense fallback to sparse: %s", exc)

    if not fused and active_pool:
        fused = active_pool[:candidate_limit]

    if page_diverse and fused:
        fused = prefer_distinct_pages(fused)

    final_docs = fused[:top_k]

    # 3. Guarded Cross-Encoder Reranking (Kaggle /api/rerank, protecting top-k prefix)
    rerank_trace: dict[str, Any] = {"status": "disabled"}
    safe_prefix = max(0, min(int(keep_prefix), MAX_CANDIDATES - 1))
    if enable_rerank and settings.REMOTE_INFERENCE_URL:
        if dense_trace.get("status") == "fallback" and dense_trace.get("backend") == "kaggle_bge_m3":
            rerank_trace = {"status": "skipped", "reason": "dense_fallback"}
        elif min(len(final_docs), MAX_CANDIDATES) - safe_prefix < 2:
            rerank_trace = {"status": "skipped", "reason": "insufficient_candidates"}
        else:
            try:
                final_docs, revision, changed = rerank_passages(
                    query,
                    final_docs,
                    client=http_client,
                    url=settings.REMOTE_INFERENCE_URL,
                    model=settings.RERANKER_MODEL,
                    timeout=float(settings.REMOTE_TIMEOUT_SECONDS),
                    keep_prefix=safe_prefix,
                )
                rerank_trace = {
                    "status": "used",
                    "model": settings.RERANKER_MODEL,
                    "revision": revision,
                    "changed": changed,
                    "kept_prefix": safe_prefix,
                }
            except RerankError as exc:
                rerank_trace = {"status": "fallback", "reason": str(exc)}
                logger.warning("[SemanticPipeline] Reranker fallback to fused order: %s", exc)

    return final_docs, dense_trace, rerank_trace
