"""
CPU Max Mode - Agentic Reader Agent (L3).
High-precision Structural Parsing & Deep Text Sanitization:
  - Multi-document scoped SourceUnit construction across all state["pdf_paths"].
  - Cleans PDF artifacts, HTML <br> tags, and stuck words (BáoThủ -> Báo Thủ, trênđường).
  - Produces uniform SourceUnits with SHA-256 context hashes and page numbers.
"""

from __future__ import annotations

import os
import re
from typing import Any

from agents.agent2_reader.tools.pdf_parser import parse_pdf_document
from agents.agent2_reader.tools.reader_pipeline import (
    build_source_units_from_docs,
    collect_unique_pdf_paths,
    evaluate_reader_table_completeness,
    ingest_pdf_fallbacks,
)
from core.base_agent import BaseAgent
from core.claim_semantics import strip_html_tags_preserve_math
from core.state import AgentState
from core.tools import load_or_parse_selected_pdf


def clean_source_text(text: str) -> str:
    """Deeply cleans raw PDF/OCR text, removes HTML line breaks, and separates glued words safely."""
    if not text:
        return ""
    # 1. Replace HTML breaks/tags safely without stripping mathematical inequalities like 'x < 3 and y > 2'
    s = strip_html_tags_preserve_math(text)

    # 2. Fix markdown underscores in words (_Báo Thủ đô_ -> Báo Thủ đô), preserving code snake_case and LaTeX subscripts like z_{1}, L_{total}
    s = re.sub(r"(?<![A-Za-z0-9\\}])_+(?![{A-Za-z0-9])", "", s)

    # 3. Fix glued lowercase-uppercase words (BáoThủ -> Báo Thủ), preserving camelCase code identifiers like generateHosts
    s = re.sub(r"([à-ỹ])([A-ZĐ])", r"\1 \2", s)
    s = re.sub(r"([a-z])([Đ])", r"\1 \2", s)

    # 4. Fix glued words around punctuation (preserve file extensions like .conf, domains like .org, URLs, ports, and floats)
    s = re.sub(r"([A-Za-zÀ-ỹ0-9])\.([A-ZĐ])", r"\1. \2", s)
    s = re.sub(r"([A-Za-zÀ-ỹ0-9]),([A-Za-zÀ-ỹ])", r"\1, \2", s)
    s = re.sub(r"([A-Za-zÀ-ỹ0-9]);([A-Za-zÀ-ỹ])", r"\1; \2", s)
    s = re.sub(r"(?<!http)(?<!https)([A-Za-zÀ-ỹ]):([A-Za-zÀ-ỹ])", r"\1: \2", s, flags=re.I)

    # 5. Clean whitespace
    s = re.sub(r"[ \t]+", " ", s)
    return s.strip()


class CPUMaxReaderAgent(BaseAgent):
    """Reader Agent for CPU Max: Multi-doc SourceUnit builder + clean text formatting."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="CPUMaxReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_paths = collect_unique_pdf_paths(state.get("pdf_path"), state.get("pdf_paths"))
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[CPU Max Reader] Building SourceUnits from {len(retrieved_docs)} ranked chunks "
            f"(docs_count={len(pdf_paths)})"
        )

        ordered_units, by_hash, pages_covered, merged_dups, curr_offset = build_source_units_from_docs(
            retrieved_docs,
            limit=max(16, len(retrieved_docs)),
            text_cleaner=clean_source_text,
            deduplicate=True,
        )

        failed_pdfs: list[dict[str, str]] = []
        covered_stems = {
            os.path.splitext(os.path.basename(str(u.doc_id or "")))[0].lower()
            for u in ordered_units
            if u.doc_id
        }
        uncovered_pdfs = [
            p for p in pdf_paths
            if os.path.splitext(os.path.basename(str(p)))[0].lower() not in covered_stems
        ]
        if uncovered_pdfs:
            failed_pdfs, extra_merged, curr_offset = ingest_pdf_fallbacks(
                uncovered_pdfs,
                ordered_units=ordered_units,
                units_by_content_hash=by_hash,
                pages_covered=pages_covered,
                curr_offset=curr_offset,
                chunk_loader=load_or_parse_selected_pdf,
                text_cleaner=clean_source_text,
                max_units_per_pdf=12,
                deduplicate=True,
            )
            merged_dups += extra_merged

        source_units = [u.to_dict() for u in ordered_units]
        reader_decision, unresolved_tables = evaluate_reader_table_completeness(state, source_units)

        trace = self.log_action(
            "cpu_max_reading_completed",
            {
                "units_count": len(source_units),
                "docs_count": len(pdf_paths),
                "dedup_merged_count": merged_dups,
                "failed_pdf_count": len(failed_pdfs),
                "failed_pdfs": failed_pdfs,
                "proposed_handoff": reader_decision.action if reader_decision else None,
                "unresolved_table_references": unresolved_tables,
            },
        )

        result: dict[str, Any] = {
            "source_units": source_units,
            "document_schema": {
                "pages_covered": sorted(list(pages_covered)),
                "total_units": len(source_units),
                "failed_pdfs": failed_pdfs,
                "dedup_merged_count": merged_dups,
                "unresolved_table_references": unresolved_tables,
            },
            "reasoning_trace": [trace],
        }
        if reader_decision is not None:
            result["agent_decision"] = reader_decision
        return result
