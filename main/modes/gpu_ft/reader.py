"""
GPU FT Mode - Agentic Reader Agent (L3 Multi-LoRA Specialist).
Converts ranked retrieved chunks from the selected PDF(s) into deduplicated SourceUnits
with multi-source provenance and extracts structured Markdown tables on top hit pages.
"""

from __future__ import annotations

import os
from typing import Any

from agents.agent2_reader.tools.reader_pipeline import (
    build_source_units_from_docs,
    collect_unique_pdf_paths,
    ingest_pdf_fallbacks,
    merge_or_append_unit,
)
from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit
from core.tools import extract_table_or_equations_tool, load_or_parse_selected_pdf


class GPUFTReaderAgent(BaseAgent):
    """Reader Agent for GPU FT Mode: Ranked SourceUnit builder + Multi-PDF Dedup & Table Extractor."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFTReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        pdf_paths = collect_unique_pdf_paths(pdf_path, state.get("pdf_paths"))
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[GPU FT Reader] Building SourceUnits from {len(retrieved_docs)} chunks "
            f"(pdf_count={len(pdf_paths)})"
        )

        ordered_units, by_hash, pages_covered, merged_dups, curr_offset = build_source_units_from_docs(
            retrieved_docs,
            limit=12,
            deduplicate=True,
        )
        document_schema: dict[str, Any] = {}
        failed_pdfs: list[dict[str, str]] = []

        for p_path in pdf_paths:
            if p_path and os.path.exists(str(p_path)):
                doc_id = os.path.splitext(os.path.basename(str(p_path)))[0]
                for p_no in sorted(list(pages_covered))[:3]:
                    try:
                        table_md = extract_table_or_equations_tool(p_path, p_no)
                    except Exception:
                        table_md = ""
                    if table_md and ("Bảng" in table_md and "|" in table_md):
                        unit = SourceUnit(
                            id=f"{doc_id}_p{p_no}_table",
                            text=table_md,
                            doc_id=doc_id,
                            page_no=p_no,
                            character_offset=curr_offset,
                        )
                        if merge_or_append_unit(unit, by_hash, ordered_units):
                            merged_dups += 1
                        curr_offset += len(table_md) + 1

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
                max_units_per_pdf=8,
                deduplicate=True,
            )
            merged_dups += extra_merged

        source_units = [u.to_dict() for u in ordered_units]

        if not document_schema:
            document_schema = {
                "doc_id": source_units[0]["doc_id"] if source_units else "unknown",
                "pages_covered": sorted(list(pages_covered)),
                "total_units": len(source_units),
                "failed_pdfs": failed_pdfs,
                "dedup_merged_count": merged_dups,
            }

        trace = self.log_action(
            "gpu_ft_reading_completed",
            {
                "has_pdf": bool(pdf_paths),
                "pdf_count": len(pdf_paths),
                "total_units": len(source_units),
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
