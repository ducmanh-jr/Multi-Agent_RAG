"""
Unified Agentic Tool Registry & Document-Scoped Tool Executor.
Provides 7 standardized tools for L3-L4 Autonomous RAG Agents across all 6 modes:
  1. search_selected_doc(query, top_k=5)
  2. search_hybrid_corpus(query, alpha=0.5, top_k=5)
  3. read_pdf_page(doc_id_or_path, page_number)
  4. extract_table_or_equations(doc_id_or_path, page_number)
  5. python_math_eval(expression)
  6. request_more_context(missing_points)
  7. finish_answer(answer, citations, confidence)

Also resolves the Critical Retrieval Scope Bug by ensuring user-selected PDFs
(state["pdf_path"]) are dynamically parsed, cached, and prioritized across EVERY mode.
"""

from __future__ import annotations

import ast
import math
import operator
import os
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Optional

from core.config import get_run_settings
from core.pdf_ingest import (
    EXTRACTOR_VERSION,
    clear_ingest_caches,
    compute_file_sha256,
    extract_page_diagrams_graph,
    extract_page_equations_2d,
    extract_page_tables_sharded,
    ocr_pdf_page_if_needed,
    record_ingest_manifest,
    stitch_cross_page_tables,
)
from utils.logger import logger


# In-memory LRU cache of parsed PDF chunks keyed by absolute file path + mtime
_PDF_CHUNK_CACHE: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
_PDF_CACHE_LOCK = threading.Lock()
_MAX_PDF_CACHE_ENTRIES = 16

def clear_pdf_cache() -> None:
    """Clears the in-memory parsed PDF chunks cache to free memory and force re-index."""
    with _PDF_CACHE_LOCK:
        _PDF_CHUNK_CACHE.clear()
    clear_ingest_caches()


_SECTION_HEADING_RE = re.compile(
    r"^(?:(?:\d+(?:\.\d+)*|[IVX]+)[.)]?\s+[A-ZÀ-Ỹ][^\n]{2,90}"
    r"|(?:Abstract|Introduction|Related Work|Background|Methodology|Method|Methods|Approach|"
    r"Experiments?|Experimental Results?|Evaluation|Results?|Discussion|Limitations?|Conclusion|Conclusions|References|"
    r"Tóm tắt|Giới thiệu|Tổng quan|Phương pháp|Thực nghiệm|Kết quả|Thảo luận|Hạn chế|Kết luận|Tài liệu tham khảo))\s*$",
    re.IGNORECASE,
)
_TABLE_CAPTION_RE = re.compile(
    r"^\s*(?:Table|Tab\.|Bảng)\s*[A-Z]?\d+(?:[.-]\d+)*\b",
    re.IGNORECASE,
)
_EQUATION_LINE_RE = re.compile(
    r"(?:"
    r"[A-Za-zΑ-Ωα-ω0-9_)\]}]\s*(?:=|≈|≤|≥|≠|∈|→|⇒)\s*[A-Za-zΑ-Ωα-ω0-9_(\[{−+\-√∑∏∫\\]"
    r"|[∑∏∫√∂∇]|\\(?:frac|sum|prod|sqrt|int|mathcal|mathbf|text|alpha|beta|gamma|theta|lambda|sigma)\b"
    r"|\b(?:softmax|argmax|argmin|LayerNorm|Attention|MultiHead|CrossEntropy|KL)\s*\("
    r"|\(\s*\d+(?:\.\d+)*\s*\)\s*$"
    r")",
    re.UNICODE,
)


def _is_math_equation_line(text: str) -> bool:
    """Detects whether a text line or sentence contains a mathematical formula or numbered equation."""
    s = (text or "").strip()
    if len(s) < 3:
        return False
    # Exclude plain URLs or key=value config lines without math structure
    if s.startswith(("http://", "https://")):
        return False
    if _EQUATION_LINE_RE.search(s):
        # Verify it has either a math relation/operator or function call, not just a trailing "(1)" list item
        if any(op in s for op in ("=", "≈", "≤", "≥", "≠", "∈", "∑", "∏", "∫", "√", "^", "²", "³", "₁", "₂", "\\frac", "\\sum", "softmax", "log(")):
            return True
    return False


_TRAILING_MATH_OP_RE = re.compile(r"(?:[=≈≤≥≠+−\-*/×⋅∑∏∫√^(\[{])$")
_LEADING_MATH_CONTINUATION_RE = re.compile(
    r"^(?:[)\]}]|[+−\-*/×⋅=≈≤≥≠]|\(\s*\d+(?:\.\d+)*\s*\)$)"
)


def _reconstruct_broken_pdf_equations(raw_text: str) -> str:
    """
    Reconstructs display equations that PyMuPDF split across consecutive short lines
    (e.g., unbalanced parentheses, trailing math operators '=', '+', '/', '√', or
    a standalone equation number '(1)' on the following line).
    """
    if not raw_text:
        return ""
    raw_lines = raw_text.splitlines()
    merged_lines: list[str] = []
    i = 0
    n = len(raw_lines)
    while i < n:
        curr = raw_lines[i].strip()
        if not curr:
            merged_lines.append("")
            i += 1
            continue

        # Check if `curr` starts or is part of a multi-line equation block
        open_parens = curr.count("(") - curr.count(")")
        looks_math = _is_math_equation_line(curr) or bool(_TRAILING_MATH_OP_RE.search(curr))
        steps = 0
        while (
            i + 1 < n
            and steps < 4
            and len(curr) < 180
        ):
            nxt = raw_lines[i + 1].strip()
            if not nxt:
                break
            should_join = False
            if looks_math and open_parens > 0 and len(nxt) < 100:
                should_join = True
            elif looks_math and _TRAILING_MATH_OP_RE.search(curr) and len(nxt) < 100:
                should_join = True
            elif (looks_math or "=" in curr) and _LEADING_MATH_CONTINUATION_RE.match(nxt) and len(nxt) < 80:
                should_join = True
            if not should_join:
                break
            curr = f"{curr} {nxt}"
            open_parens = curr.count("(") - curr.count(")")
            looks_math = True
            i += 1
            steps += 1

        merged_lines.append(curr)
        i += 1
    return "\n".join(merged_lines)


def _extract_page_equation_chunks(
    raw_text: str,
    *,
    doc_title: str,
    page_no: int,
    resolved: str,
    section_title: str,
) -> list[dict[str, Any]]:
    """
    Extracts mathematical equation blocks from page text together with their immediate
    preceding/following explanatory context (e.g. 'where Q, K, V are...').
    """
    if not raw_text or not raw_text.strip():
        return []
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in raw_text.splitlines() if ln.strip()]
    eq_chunks: list[dict[str, Any]] = []
    seen_bodies: set[str] = set()
    eq_idx = 0

    for i, line in enumerate(lines):
        if not _is_math_equation_line(line):
            continue
        start_i = max(0, i - 1)
        end_i = min(len(lines), i + 3)
        window_lines = lines[start_i:end_i]
        body = " ".join(window_lines).strip()
        norm_sig = re.sub(r"\s+", "", body.lower())
        if not body or norm_sig in seen_bodies:
            continue
        seen_bodies.add(norm_sig)
        eq_idx += 1
        cid = f"{doc_title}_p{page_no}_eq{eq_idx}"
        eq_chunks.append(
            {
                "id": cid,
                "chunk_id": cid,
                "doc_id": doc_title,
                "title": doc_title,
                "page_no": page_no,
                "page": page_no,
                "text": body,
                "content": body,
                "equation": line,
                "sentences": window_lines,
                "source_path": resolved,
                "is_selected_pdf": True,
                "unit_type": "equation",
                "section_title": section_title,
            }
        )
    return eq_chunks


def _extract_page_text_layout_aware(page: Any) -> str:
    """
    Extracts page text while preserving reading order on 2-column academic layouts.
    Falls back to standard page.get_text("text") on single-column pages so existing
    plain-text documents keep identical chunking behavior.
    """
    try:
        page_rect = page.rect
        page_w = float(page_rect.width) if page_rect else 0.0
        raw_blocks = page.get_text("blocks") or []
    except Exception:
        return page.get_text("text") or ""

    text_blocks = [
        (float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4]).strip())
        for b in raw_blocks
        if len(b) >= 7 and int(b[6]) == 0 and str(b[4]).strip()
    ]
    if page_w <= 0 or len(text_blocks) < 4:
        return page.get_text("text") or ""

    mid_x = page_w * 0.5
    left_col: list[tuple[float, float, float, float, str]] = []
    right_col: list[tuple[float, float, float, float, str]] = []
    spanning: list[tuple[float, float, float, float, str]] = []

    for blk in text_blocks:
        x0, y0, x1, y1, _ = blk
        w = x1 - x0
        if w > page_w * 0.58 or (x0 < mid_x * 0.85 and x1 > mid_x * 1.15):
            spanning.append(blk)
        elif x1 <= mid_x * 1.08:
            left_col.append(blk)
        elif x0 >= mid_x * 0.92:
            right_col.append(blk)
        else:
            spanning.append(blk)

    # Only switch to column-sorted reading order when a genuine 2-column layout is present
    if len(left_col) < 2 or len(right_col) < 2:
        return page.get_text("text") or ""

    left_col.sort(key=lambda b: (b[1], b[0]))
    right_col.sort(key=lambda b: (b[1], b[0]))
    spanning.sort(key=lambda b: (b[1], b[0]))

    col_top = min(left_col[0][1], right_col[0][1])
    col_bottom = max(left_col[-1][3], right_col[-1][3])
    header_blocks = [b for b in spanning if b[3] <= col_top + 12.0]
    footer_blocks = [b for b in spanning if b[1] >= col_bottom - 12.0]
    mid_spanning = [b for b in spanning if b not in header_blocks and b not in footer_blocks]

    ordered = header_blocks + left_col + right_col + mid_spanning + footer_blocks
    return "\n\n".join(b[4] for b in ordered if b[4])


def _extract_page_table_chunks(
    page: Any,
    *,
    doc_title: str,
    page_no: int,
    resolved: str,
    section_title: str,
) -> list[dict[str, Any]]:
    """Extracts intact Markdown table chunks from a PDF page using PyMuPDF find_tables()."""
    if not hasattr(page, "find_tables"):
        return []
    table_chunks: list[dict[str, Any]] = []
    try:
        from core.pdf_ingest import _extract_page_table_entries_robust, _format_markdown_rows

        table_entries = _extract_page_table_entries_robust(page)
    except Exception:
        return []
    if not table_entries:
        return []

    # Collect nearby caption candidates (Table N / Bảng N)
    caption_blocks: list[tuple[float, float, float, float, str]] = []
    try:
        for b in page.get_text("blocks") or []:
            if len(b) >= 7 and int(b[6]) == 0:
                txt = re.sub(r"[ \t]+", " ", str(b[4]).replace("\n", " ")).strip()
                if txt and _TABLE_CAPTION_RE.match(txt):
                    caption_blocks.append((float(b[0]), float(b[1]), float(b[2]), float(b[3]), txt))
    except Exception:
        caption_blocks = []

    for idx, (bbox, extracted_rows) in enumerate(table_entries, start=1):
        try:
            if len(extracted_rows) < 2:
                continue
            max_cols = max((len(r) for r in extracted_rows if isinstance(r, list)), default=0)
            if max_cols < 2:
                continue
            non_empty_cells = sum(
                1 for row in extracted_rows if isinstance(row, list)
                for cell in row if cell is not None and str(cell).strip()
            )
            if non_empty_cells < 3:
                continue
            md = _format_markdown_rows(extracted_rows[0], extracted_rows[1:]).strip()
            if not md:
                continue
        except Exception:
            continue
        # Match closest Table caption within 65pt above or below the table
        matched_caption = ""
        best_dist = 65.0
        for cx0, cy0, cx1, cy1, ctxt in caption_blocks:
            dist_above = bbox[1] - cy1
            dist_below = cy0 - bbox[3]
            if -6.0 <= dist_above <= best_dist:
                best_dist = max(0.0, dist_above)
                matched_caption = ctxt
            elif -6.0 <= dist_below <= best_dist:
                best_dist = max(0.0, dist_below)
                matched_caption = ctxt

        table_body = f"{matched_caption}\n{md}" if matched_caption else md
        cid = f"{doc_title}_p{page_no}_t{idx}"
        table_chunks.append(
            {
                "id": cid,
                "chunk_id": cid,
                "doc_id": doc_title,
                "title": doc_title,
                "page_no": page_no,
                "page": page_no,
                "text": table_body,
                "content": table_body,
                "sentences": [line.strip() for line in table_body.splitlines() if line.strip()],
                "source_path": resolved,
                "is_selected_pdf": True,
                "unit_type": "table",
                "bbox": [round(float(v), 2) for v in bbox[:4]],
                "section_title": section_title,
            }
        )
    return table_chunks


def _detect_section_heading(text: str, current_section: str) -> str:
    """Scans leading lines of a page/chunk for an academic section heading."""
    for line in (text or "").splitlines():
        clean = re.sub(r"[ \t]+", " ", line).strip()
        if 3 <= len(clean) <= 100 and _SECTION_HEADING_RE.match(clean):
            return clean
    return current_section


def load_or_parse_selected_pdf(
    pdf_path: Optional[str],
    chunk_size_words: Any = 240,
    overlap_words: int = 50,
    root_dir: Optional[str] = None,
) -> list[dict[str, Any]]:
    """
    Dynamically parses and caches the user-selected PDF document with:
      - 2-column academic reading-order preservation,
      - intact Markdown table chunk extraction (unit_type='table'),
      - section heading tracking, and sentence-window preservation.
    Guarantees that when a user selects a document in the UI, ALL modes retrieve directly from it.
    """
    if not pdf_path:
        return []
    # Support legacy/convenience call load_or_parse_selected_pdf(selected_doc, root_dir_str)
    if isinstance(chunk_size_words, (str, Path)):
        root_dir = str(chunk_size_words)
        chunk_size_words = 240
    else:
        try:
            chunk_size_words = int(chunk_size_words)
        except Exception:
            chunk_size_words = 240

    resolved = str(pdf_path).strip()
    if not os.path.exists(resolved):
        candidate = os.path.join(get_run_settings().UPLOAD_DIR, os.path.basename(resolved))
        if os.path.exists(candidate):
            resolved = candidate
        elif root_dir and os.path.exists(os.path.join(root_dir, "uploads", os.path.basename(resolved))):
            resolved = os.path.join(root_dir, "uploads", os.path.basename(resolved))
        else:
            return []

    try:
        mtime = os.path.getmtime(resolved)
    except Exception:
        mtime = 0.0
    cache_key = f"{os.path.abspath(resolved)}::{mtime}"
    with _PDF_CACHE_LOCK:
        if cache_key in _PDF_CHUNK_CACHE:
            _PDF_CHUNK_CACHE.move_to_end(cache_key)
            return _PDF_CHUNK_CACHE[cache_key]
        parse_locks = getattr(load_or_parse_selected_pdf, "_parse_locks", None)
        if parse_locks is None:
            parse_locks = {}
            setattr(load_or_parse_selected_pdf, "_parse_locks", parse_locks)
        file_lock = parse_locks.get(cache_key)
        if file_lock is None:
            file_lock = threading.Lock()
            parse_locks[cache_key] = file_lock
            if len(parse_locks) > 64:
                oldest_k = next(iter(parse_locks))
                if oldest_k != cache_key:
                    parse_locks.pop(oldest_k, None)

    # Single-flight per-file lock prevents 4 parallel DAG threads from parsing a 70-page PDF 4x concurrently
    with file_lock:
        with _PDF_CACHE_LOCK:
            if cache_key in _PDF_CHUNK_CACHE:
                _PDF_CHUNK_CACHE.move_to_end(cache_key)
                return _PDF_CHUNK_CACHE[cache_key]
        return _parse_and_cache_selected_pdf_locked(resolved, cache_key, chunk_size_words)


def _parse_and_cache_selected_pdf_locked(
    resolved: str,
    cache_key: str,
    chunk_size_words: int,
) -> list[dict[str, Any]]:

    doc_title = Path(resolved).stem
    doc_title = re.sub(r"^[0-9a-f]{8}_", "", doc_title)
    chunks: list[dict[str, Any]] = []
    pages_status: list[dict[str, Any]] = []
    doc_sha256 = ""
    try:
        doc_sha256 = compute_file_sha256(resolved)
    except Exception:
        doc_sha256 = ""

    # Support plain text / markdown files selected via local path
    if Path(resolved).suffix.lower() in {".txt", ".md"}:
        try:
            txt_content = Path(resolved).read_text(encoding="utf-8", errors="replace").strip()
            if txt_content:
                cid = f"{doc_title}_p1_c1"
                chunks.append({
                    "id": cid,
                    "chunk_id": cid,
                    "doc_id": doc_title,
                    "document_sha256": doc_sha256,
                    "title": doc_title,
                    "page_no": 1,
                    "page": 1,
                    "page_rotation": 0,
                    "coordinate_space": "text_offset",
                    "extractor_version": EXTRACTOR_VERSION,
                    "text": txt_content,
                    "raw_text": txt_content,
                    "normalized_text": txt_content,
                    "content": txt_content,
                    "sentences": [s.strip() for s in txt_content.splitlines() if s.strip()],
                    "source_path": resolved,
                    "is_selected_pdf": True,
                    "unit_type": "text",
                    "section_title": "",
                })
                record_ingest_manifest(resolved, {
                    "sha256": doc_sha256,
                    "extractor_version": EXTRACTOR_VERSION,
                    "page_count": 1,
                    "status": "completed",
                    "pages": [{"page_no": 1, "status": "text_ready", "tier": "tier1_native", "chunks_count": 1}],
                })
        except Exception as exc:
            logger.warning(f"[ToolRegistry] Text file read failed on '{resolved}': {exc}")
        return chunks

    try:
        import fitz  # PyMuPDF

        pdf_doc = fitz.open(resolved)
        current_section = ""
        total_pages = len(pdf_doc)
        page_heights_map: dict[int, float] = {}
        try:
            for page_idx in range(total_pages):
                page_no = page_idx + 1
                try:
                    page = pdf_doc[page_idx]
                    rotation = int(page.rotation or 0)
                    derot = page.derotation_matrix
                    unrot_bounds = page.rect * derot
                    page_heights_map[page_no] = float(unrot_bounds.height)
                    default_page_bbox = [round(float(v), 2) for v in unrot_bounds]

                    raw_text = _reconstruct_broken_pdf_equations(_extract_page_text_layout_aware(page))
                    current_section = _detect_section_heading(raw_text, current_section)
                    clean_text = re.sub(r"[ \t]+", " ", raw_text).strip()

                    page_tables = extract_page_tables_sharded(
                        page,
                        doc_title=doc_title,
                        doc_sha256=doc_sha256,
                        page_no=page_no,
                        resolved=resolved,
                        section_title=current_section,
                    )
                    table_bboxes_on_page = [
                        list((tc.get("table_metadata") or {}).get("table_bbox_on_page") or tc.get("bbox") or [])
                        for tc in page_tables
                        if tc.get("bbox")
                    ]
                    page_diagrams = extract_page_diagrams_graph(
                        page,
                        doc_title=doc_title,
                        doc_sha256=doc_sha256,
                        page_no=page_no,
                        resolved=resolved,
                        section_title=current_section,
                        excluded_table_bboxes=table_bboxes_on_page,
                    )
                    page_equations = extract_page_equations_2d(
                        page,
                        doc_title=doc_title,
                        doc_sha256=doc_sha256,
                        page_no=page_no,
                        resolved=resolved,
                        section_title=current_section,
                    )
                    if not page_equations and clean_text:
                        page_equations = _extract_page_equation_chunks(
                            raw_text,
                            doc_title=doc_title,
                            page_no=page_no,
                            resolved=resolved,
                            section_title=current_section,
                        )
                        for eq_c in page_equations:
                            eq_c.setdefault("bbox", default_page_bbox)
                            eq_c.setdefault("page_rotation", rotation)
                            eq_c.setdefault("document_sha256", doc_sha256)
                            eq_c.setdefault("extractor_version", EXTRACTOR_VERSION)

                    page_chunk_start = len(chunks)
                    tier_used = "tier1_native"

                    # Collect block-level bboxes on this page to localize text chunks accurately
                    page_text_blocks: list[tuple[list[float], str]] = []
                    try:
                        for b in page.get_text("blocks") or []:
                            if len(b) >= 7 and int(b[6]) == 0 and str(b[4]).strip():
                                rb = fitz.Rect(b[:4]) * derot
                                page_text_blocks.append(([round(float(v), 2) for v in rb], " ".join(str(b[4]).split())))
                    except Exception:
                        page_text_blocks = []

                    def _infer_chunk_bbox(sents_group: list[str]) -> list[float]:
                        matched_boxes = []
                        for sb, btxt in page_text_blocks:
                            b_low = btxt.lower()
                            if any(st[:32].lower() in b_low or b_low[:32] in st.lower() for st in sents_group if st):
                                matched_boxes.append(sb)
                        if not matched_boxes:
                            return default_page_bbox
                        return [
                            round(min(b[0] for b in matched_boxes), 2),
                            round(min(b[1] for b in matched_boxes), 2),
                            round(max(b[2] for b in matched_boxes), 2),
                            round(max(b[3] for b in matched_boxes), 2),
                        ]

                    # Tier-1 Scan OCR: when native text is absent/minimal and raster images or drawings exist
                    ocr_info: dict[str, Any] | None = None
                    if len(clean_text) < 15:
                        has_raster_or_drawings = bool(page.get_image_info()) or bool(page.get_drawings())
                        if has_raster_or_drawings:
                            ocr_info = ocr_pdf_page_if_needed(page, doc_sha256=doc_sha256 or doc_title, page_no=page_no)
                            ocr_txt = (ocr_info.get("text") or "").strip()
                            if ocr_txt:
                                tier_used = "tier1_ocr"
                                cid = f"{doc_title}_p{page_no}_ocr1"
                                chunks.append({
                                    "id": cid,
                                    "chunk_id": cid,
                                    "doc_id": doc_title,
                                    "document_sha256": doc_sha256,
                                    "title": doc_title,
                                    "page_no": page_no,
                                    "page": page_no,
                                    "page_rotation": rotation,
                                    "coordinate_space": "pdf_points_top_left",
                                    "extractor_version": EXTRACTOR_VERSION,
                                    "bbox": ocr_info.get("bbox") or default_page_bbox,
                                    "text": ocr_txt,
                                    "raw_text": ocr_txt,
                                    "normalized_text": ocr_txt,
                                    "content": ocr_txt,
                                    "sentences": [l["text"] for l in (ocr_info.get("lines") or []) if l.get("text")],
                                    "source_path": resolved,
                                    "is_selected_pdf": True,
                                    "unit_type": "ocr_scan",
                                    "ocr_engine": ocr_info.get("engine", ""),
                                    "ocr_confidence": ocr_info.get("confidence", 0.9),
                                    "section_title": current_section,
                                })

                    if clean_text:
                        # Split into sentences first so chunks never cut mid-sentence; keep short math equations intact
                        sentences = [
                            s.strip()
                            for s in re.split(r"(?<=[.!?।])\s+|\n{2,}", clean_text)
                            if len(s.strip()) > 12 or _is_math_equation_line(s)
                        ]
                        if not sentences:
                            sentences = [clean_text]

                        current_sents: list[str] = []
                        current_words = 0
                        sub_idx = 1

                        for sent in sentences:
                            current_section = _detect_section_heading(sent, current_section)
                            w_count = len(sent.split())
                            if current_words + w_count > chunk_size_words and current_sents:
                                chunk_body = " ".join(current_sents).strip()
                                cid = f"{doc_title}_p{page_no}_c{sub_idx}"
                                chunks.append(
                                    {
                                        "id": cid,
                                        "chunk_id": cid,
                                        "doc_id": doc_title,
                                        "document_sha256": doc_sha256,
                                        "title": doc_title,
                                        "page_no": page_no,
                                        "page": page_no,
                                        "page_rotation": rotation,
                                        "coordinate_space": "pdf_points_top_left",
                                        "extractor_version": EXTRACTOR_VERSION,
                                        "bbox": _infer_chunk_bbox(current_sents),
                                        "text": chunk_body,
                                        "raw_text": chunk_body,
                                        "normalized_text": chunk_body,
                                        "content": chunk_body,
                                        "sentences": list(current_sents),
                                        "source_path": resolved,
                                        "is_selected_pdf": True,
                                        "unit_type": "text",
                                        "section_title": current_section,
                                    }
                                )
                                sub_idx += 1
                                current_sents = current_sents[-2:] if len(current_sents) >= 2 else current_sents[-1:]
                                current_words = sum(len(x.split()) for x in current_sents)

                            current_sents.append(sent)
                            current_words += w_count

                        if current_sents:
                            chunk_body = " ".join(current_sents).strip()
                            cid = f"{doc_title}_p{page_no}_c{sub_idx}"
                            chunks.append(
                                {
                                    "id": cid,
                                    "chunk_id": cid,
                                    "doc_id": doc_title,
                                    "document_sha256": doc_sha256,
                                    "title": doc_title,
                                    "page_no": page_no,
                                    "page": page_no,
                                    "page_rotation": rotation,
                                    "coordinate_space": "pdf_points_top_left",
                                    "extractor_version": EXTRACTOR_VERSION,
                                    "bbox": _infer_chunk_bbox(current_sents),
                                    "text": chunk_body,
                                    "raw_text": chunk_body,
                                    "normalized_text": chunk_body,
                                    "content": chunk_body,
                                    "sentences": list(current_sents),
                                    "source_path": resolved,
                                    "is_selected_pdf": True,
                                    "unit_type": "text",
                                    "section_title": current_section,
                                }
                            )

                    if page_tables:
                        chunks.extend(page_tables)
                    if page_diagrams:
                        chunks.extend(page_diagrams)
                    if page_equations:
                        chunks.extend(page_equations)

                    added_on_page = len(chunks) - page_chunk_start
                    p_status = "ocr_completed" if tier_used == "tier1_ocr" else ("text_ready" if added_on_page > 0 else "empty_or_pending_ocr")
                    pages_status.append({
                        "page_no": page_no,
                        "status": p_status,
                        "tier": tier_used,
                        "page_rotation": rotation,
                        "chunks_count": added_on_page,
                        "tables_count": len(page_tables),
                        "diagrams_count": len(page_diagrams),
                        "equations_count": len(page_equations),
                    })
                except Exception as page_exc:
                    # Per-page fault isolation: one damaged page NEVER aborts remaining valid pages!
                    logger.warning(f"[ToolRegistry] Page {page_no} parse error on '{resolved}': {page_exc}")
                    pages_status.append({
                        "page_no": page_no,
                        "status": "failed",
                        "tier": "failed",
                        "chunks_count": 0,
                        "error": str(page_exc),
                    })

            # M2b: Stitch multi-page tables across consecutive pages while preserving per-page (page_no, bbox)
            stitched_tables_info: list[dict[str, Any]] = []
            if chunks:
                chunks, stitched_tables_info = stitch_cross_page_tables(
                    chunks,
                    doc_title=doc_title,
                    page_heights=page_heights_map,
                )

            failed_pages = [p["page_no"] for p in pages_status if p["status"] == "failed"]
            overall = "completed" if not failed_pages else ("partial_completed" if chunks else "failed")
            record_ingest_manifest(resolved, {
                "sha256": doc_sha256,
                "extractor_version": EXTRACTOR_VERSION,
                "page_count": total_pages,
                "chunks_count": len(chunks),
                "stitched_logical_tables": stitched_tables_info,
                "diagrams_count": sum(1 for c in chunks if c.get("unit_type") == "diagram"),
                "failed_pages": failed_pages,
                "status": overall,
                "pages": pages_status,
            })
        finally:
            pdf_doc.close()
    except Exception as exc:
        logger.warning(f"[ToolRegistry] PyMuPDF parse failed on '{resolved}': {exc}")
        record_ingest_manifest(resolved, {
            "sha256": doc_sha256,
            "extractor_version": EXTRACTOR_VERSION,
            "page_count": 0,
            "chunks_count": 0,
            "failed_pages": [],
            "status": "failed",
            "error": str(exc),
            "pages": [],
        })

    if chunks:
        with _PDF_CACHE_LOCK:
            if cache_key in _PDF_CHUNK_CACHE:
                _PDF_CHUNK_CACHE.move_to_end(cache_key)
            else:
                if len(_PDF_CHUNK_CACHE) >= _MAX_PDF_CACHE_ENTRIES:
                    _PDF_CHUNK_CACHE.popitem(last=False)
            _PDF_CHUNK_CACHE[cache_key] = chunks
    return chunks


def read_pdf_page_tool(pdf_path: Optional[str], page_number: int = 1) -> str:
    """Reads the full text of a specific page (1-indexed) from the selected PDF."""
    chunks = load_or_parse_selected_pdf(pdf_path)
    if not chunks:
        return f"Không tìm thấy tài liệu PDF tại đường dẫn: {pdf_path}"
    page_chunks = [c["text"] for c in chunks if int(c.get("page_no", 1)) == int(page_number)]
    if not page_chunks:
        max_page = max((int(c.get("page_no", 1)) for c in chunks), default=1)
        return f"Trang {page_number} không có nội dung (Tài liệu có tối đa {max_page} trang)."
    return f"[Trang {page_number}]:\n" + "\n\n".join(page_chunks)


def extract_table_or_equations_tool(pdf_path: Optional[str], page_number: int = 1) -> str:
    """Extracts structured Markdown tables and mathematical equations from a specific PDF page."""
    if not pdf_path or not os.path.exists(str(pdf_path)):
        return "Không có đường dẫn PDF hợp lệ để trích xuất bảng hoặc công thức."
    try:
        import fitz

        doc = fitz.open(str(pdf_path))
        try:
            if page_number < 1 or page_number > len(doc):
                return f"Số trang {page_number} nằm ngoài phạm vi tài liệu (1..{len(doc)})."
            page = doc[page_number - 1]
            sections: list[str] = []
            doc_stem = Path(str(pdf_path)).stem
            sharded_tabs = extract_page_tables_sharded(
                page,
                doc_title=doc_stem,
                doc_sha256="",
                page_no=page_number,
                resolved=str(pdf_path),
                section_title="",
            )
            for idx, tchunk in enumerate(sharded_tabs, start=1):
                sections.append(f"### Bảng {idx} (Trang {page_number})\n{tchunk['text']}")

            eq_chunks = extract_page_equations_2d(
                page,
                doc_title=doc_stem,
                doc_sha256="",
                page_no=page_number,
                resolved=str(pdf_path),
                section_title="",
            )
            if not eq_chunks:
                raw_text = _reconstruct_broken_pdf_equations(_extract_page_text_layout_aware(page))
                eq_chunks = _extract_page_equation_chunks(
                    raw_text,
                    doc_title=doc_stem,
                    page_no=page_number,
                    resolved=str(pdf_path),
                    section_title="",
                )
            if eq_chunks:
                eq_lines = []
                for idx, c in enumerate(eq_chunks, start=1):
                    latex_part = f" | LaTeX: `${c['latex']}$`" if c.get("latex") and c.get("latex") != c.get("equation") else ""
                    bbox_part = f" | BBox: `{c['bbox']}`" if c.get("bbox") else ""
                    eq_lines.append(
                        f"- **Công thức {idx}:** `{c['equation']}`{latex_part}{bbox_part}\n  *Ngữ cảnh:* {c['text']}"
                    )
                sections.append(f"### Công thức / Biểu thức Toán học (Trang {page_number})\n" + "\n".join(eq_lines))

            if not sections:
                return f"Không phát hiện bảng biểu hoặc công thức toán học trên trang {page_number}."
            return "\n\n".join(sections)
        finally:
            doc.close()
    except Exception as exc:
        return f"Lỗi trích xuất bảng biểu/công thức: {exc}"


def _safe_pow(base: float, exp: float) -> float:
    """Safe power calculation strictly bounded to avoid CPU/RAM exhaustion DoS."""
    if abs(exp) > 100:
        raise ValueError(f"Số mũ {exp} vượt quá giới hạn an toàn (tối đa 100).")
    if abs(base) > 1e6 and exp > 10:
        raise ValueError("Cơ số và số mũ quá lớn, có nguy cơ gây tràn tài nguyên.")
    try:
        res = operator.pow(base, exp)
        if math.isinf(res) or math.isnan(res):
            raise OverflowError("Kết quả lũy thừa vượt quá giá trị số thực.")
        return float(res)
    except OverflowError:
        raise OverflowError("Phép tính lũy thừa gây tràn số học.")


_SAFE_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: _safe_pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def safe_math_eval(expression: str) -> str:
    """
    Safely evaluates mathematical & scientific expressions (sum, diff, percentage, ratio,
    exponents, log/ln/exp/sqrt/trig, constants pi/e) without eval().
    """
    safe_constants = {
        "pi": math.pi,
        "e": math.e,
    }
    safe_funcs: dict[str, Callable[..., Any]] = {
        "round": lambda val, ndigits=0: round(float(val), int(ndigits)),
        "abs": abs,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "log": math.log,
        "ln": math.log,
        "log10": math.log10,
        "log2": math.log2,
        "exp": math.exp,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "pow": _safe_pow,
    }

    def _eval_node(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Num):  # Python 3.7 compat
            return float(node.n)  # type: ignore[arg-type]
        if isinstance(node, ast.Name) and node.id in safe_constants:
            return float(safe_constants[node.id])
        if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right)))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
            return float(_SAFE_OPS[type(node.op)](_eval_node(node.operand)))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fn_name = node.func.id
            args = [_eval_node(a) for a in node.args]
            if fn_name in safe_funcs:
                return float(safe_funcs[fn_name](*args))
        raise ValueError(f"Biểu thức không được phép: {ast.dump(node)}")

    try:
        clean_expr = (
            expression.strip()
            .replace("−", "-")
            .replace("×", "*")
            .replace("⋅", "*")
            .replace("^", "**")
            .replace("%", "/100.0")
        )
        tree = ast.parse(clean_expr, mode="eval")
        result = _eval_node(tree)
        return f"{result:.6g}"
    except Exception as exc:
        return f"MathError: {exc}"


def get_gemini_tool_declarations() -> list[dict[str, Any]]:
    """Returns OpenAI / Gemini compatible tool schemas for L3-L4 ReAct agents."""
    return [
        {
            "name": "search_selected_doc",
            "description": "Tìm kiếm đoạn văn bản trong tài liệu PDF người dùng đang chọn theo câu truy vấn.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "query": {"type": "STRING", "description": "Câu truy vấn tìm kiếm cụ thể."},
                    "top_k": {"type": "INTEGER", "description": "Số lượng đoạn văn cần lấy (mặc định 5)."},
                },
                "required": ["query"],
            },
        },
        {
            "name": "read_pdf_page",
            "description": "Đọc toàn văn nội dung của một trang cụ thể trong tài liệu PDF đang chọn.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "page_number": {"type": "INTEGER", "description": "Số trang PDF cần đọc (bắt đầu từ 1)."},
                },
                "required": ["page_number"],
            },
        },
        {
            "name": "extract_table_or_equations",
            "description": "Trích xuất bảng số liệu hoặc công thức từ một trang PDF cụ thể dưới dạng Markdown.",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "page_number": {"type": "INTEGER", "description": "Số trang PDF chứa bảng biểu."},
                },
                "required": ["page_number"],
            },
        },
        {
            "name": "python_math_eval",
            "description": "Tính toán biểu thức toán học chính xác (tỷ lệ %, tổng, hiệu, trung bình).",
            "parameters": {
                "type": "OBJECT",
                "properties": {
                    "expression": {"type": "STRING", "description": "Biểu thức số học hợp lệ, ví dụ '(125 - 100) / 100 * 100'."},
                },
                "required": ["expression"],
            },
        },
    ]
