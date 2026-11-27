"""
CPU Min Mode - Agentic Reader Agent (L3-Lite).
Features:
  - Multi-document conversion into SourceUnits with exact page numbers and character offsets.
  - Deep text sanitization: eliminates <br>, HTML tags, and table serialization noise.
  - Preserves cross-page document structure for comprehensive summary generation.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable

from agents.agent2_reader.tools.reader_pipeline import (
    build_source_units_from_docs,
    collect_unique_pdf_paths,
    ingest_pdf_fallbacks,
)
from core.base_agent import BaseAgent
from core.citation_audit import citation_location, evidence_quote
from core.claim_semantics import strip_html_tags_preserve_math
from core.state import AgentState, SourceUnit, compute_context_hash
from core.tools import extract_table_or_equations_tool, load_or_parse_selected_pdf


def sanitize_unit_text(text: str) -> str:
    """Strip raw HTML tags, line breaks, and table markdown artifacts from source unit text."""
    if not text:
        return ""
    s = strip_html_tags_preserve_math(text)
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


def read_comparison_document(
    reader: Callable[..., dict[str, Any]], query: str, path: str, ranking: list[dict[str, Any]],
) -> dict[str, Any]:
    ranked_text = {
        (chunk["id"], citation_location(chunk)): sanitize_unit_text(evidence_quote(chunk)) for chunk in ranking
    }
    result = reader(
        {"query": query, "pdf_path": path, "pdf_paths": [path], "retrieved_docs": ranking},
        allow_pdf_fallback=False,
    )
    units = []
    for unit in result.get("source_units") or []:
        location = citation_location(unit)
        text = evidence_quote(unit)
        if (
            not location or not text.strip() or text != ranked_text.get((unit.get("id"), location))
            or unit.get("context_hash") != compute_context_hash(text, *location)
        ):
            raise ValueError("Reader returned evidence that cannot be traced to the ranked source.")
        units.append({**unit, "source_path": path})
    if not units:
        raise ValueError("Reader returned no traceable evidence from the selected document.")
    return {**result, "source_units": units}


class CPUMinReaderAgent(BaseAgent):
    """Reader Agent optimized for CPU Min: Multi-document SourceUnit builder + clean text pipeline."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState, *, allow_pdf_fallback: bool = True) -> dict[str, Any]:
        pdf_paths = collect_unique_pdf_paths(state.get("pdf_path"), state.get("pdf_paths"))
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[CPU Min Reader] Building SourceUnits from {len(retrieved_docs)} ranked chunks "
            f"(docs_count={len(pdf_paths)})"
        )

        # Keep 1:1 chunk traceability for single-doc comparison reader; dedup in unified multi-PDF mode
        dedup = bool(allow_pdf_fallback)
        ordered_units, by_hash, pages_covered, merged_dups, curr_offset = build_source_units_from_docs(
            retrieved_docs,
            limit=16,
            text_cleaner=sanitize_unit_text,
            deduplicate=dedup,
        )

        failed_pdfs: list[dict[str, str]] = []
        if allow_pdf_fallback and not ordered_units and pdf_paths:
            failed_pdfs, extra_merged, curr_offset = ingest_pdf_fallbacks(
                pdf_paths,
                ordered_units=ordered_units,
                units_by_content_hash=by_hash,
                pages_covered=pages_covered,
                curr_offset=curr_offset,
                chunk_loader=load_or_parse_selected_pdf,
                text_cleaner=sanitize_unit_text,
                max_units_per_pdf=10,
                deduplicate=True,
            )
            merged_dups += extra_merged

        source_units = [u.to_dict() for u in ordered_units]

        document_schema = {
            "doc_id": source_units[0]["doc_id"] if source_units else "unknown",
            "pages_covered": sorted(list(pages_covered)),
            "engine": "Sanitized_SourceUnit_Pipeline",
            "total_units": len(source_units),
            "failed_pdfs": failed_pdfs,
            "dedup_merged_count": merged_dups,
        }

        trace = self.log_action(
            "cpu_min_reading_completed",
            {
                "engine": "Sanitized_SourceUnit_Pipeline",
                "total_source_units": len(source_units),
                "pages_covered": sorted(list(pages_covered)),
                "dedup_merged_count": merged_dups,
                "failed_pdf_count": len(failed_pdfs),
                "failed_pdfs": failed_pdfs,
            },
        )

        return {
            "document_schema": document_schema,
            "source_units": source_units,
            "reasoning_trace": [trace],
        }
