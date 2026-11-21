"""
Shared Multi-PDF Reader Pipeline for all execution mode Readers.
Provides:
  1. Multi-PDF path collection and resilient per-file ingestion (partial-failure handling).
  2. Exact-text deduplication across documents while preserving multi-source provenance.
  3. Strict preservation of primary SourceUnit context_hash for CitationAudit & ComparisonPlan.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Optional

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
        clean_text = text_cleaner(raw_text) if text_cleaner else raw_text
        if not clean_text:
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

        unit = SourceUnit(
            id=str(doc.get("id") or f"{doc_id}_p{page_no}_u{idx + 1}"),
            text=clean_text,
            doc_id=doc_id,
            page_no=page_no,
            bbox=doc.get("bbox"),
            character_offset=unit_offset,
            provenance=list(doc.get("provenance") or []),
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
            if not clean_text:
                continue
            pages_covered.add(page_no)
            unit_offset = int(orig_offset) if orig_offset is not None else curr_offset
            unit = SourceUnit(
                id=uid,
                text=clean_text,
                doc_id=doc_id,
                page_no=page_no,
                bbox=bbox,
                character_offset=unit_offset,
            )
            if deduplicate:
                if merge_or_append_unit(unit, units_by_content_hash, ordered_units):
                    merged_count += 1
            else:
                ordered_units.append(unit)
            curr_offset = unit_offset + len(clean_text) + 1

    return failed_pdfs, merged_count, curr_offset
