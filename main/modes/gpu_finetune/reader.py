"""
GPU Finetune Mode - Agentic Reader Agent (7-Model SOTA Pipeline).
Leverages Remote GPU Worker powered by:
  - Fine-Tuned LayoutLMv3 Base (02_agent2_reader):
    Predicts 11 2D Layout Classes (Title, Text, Table, Figure, Equation, Header, Footer, Caption, List, Metadata, Footnote)
    to perform high-fidelity structural parsing of scientific papers.
  - Generates verified SourceUnits with exact page_no, offsets, and extracted Markdown tables.
"""

from __future__ import annotations

import os
from typing import Any

from agents.agent2_reader.tools.pdf_parser import parse_pdf_document
from agents.agent2_reader.tools.reader_pipeline import (
    build_source_units_from_docs,
    collect_unique_pdf_paths,
    ingest_pdf_fallbacks,
    merge_or_append_unit,
)
from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit
from core.tools import extract_table_or_equations_tool, load_or_parse_selected_pdf
from utils.llm_client import llm_client


class GPUFinetuneReaderAgent(BaseAgent):
    """Reader Agent using Fine-Tuned LayoutLMv3 2D Layout Parser."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GPUFinetuneReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        pdf_paths = collect_unique_pdf_paths(pdf_path, state.get("pdf_paths"))
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[GPU Finetune Reader] Building SourceUnits from {len(retrieved_docs)} chunks "
            f"(pdf_count={len(pdf_paths)})"
        )

        ordered_units, by_hash, pages_covered, merged_dups, curr_offset = build_source_units_from_docs(
            retrieved_docs,
            limit=12,
            deduplicate=True,
        )
        document_schema: dict[str, Any] = {}
        failed_pdfs: list[dict[str, str]] = []

        if not ordered_units and pdf_paths:
            failed_pdfs, extra_merged, curr_offset = ingest_pdf_fallbacks(
                pdf_paths,
                ordered_units=ordered_units,
                units_by_content_hash=by_hash,
                pages_covered=pages_covered,
                curr_offset=curr_offset,
                chunk_loader=load_or_parse_selected_pdf,
                max_units_per_pdf=10,
                deduplicate=True,
            )
            merged_dups += extra_merged

        # 2. LayoutLMv3 Document Layout Analysis
        extracted_tables: list[tuple[int, str]] = []
        primary_pdf = pdf_path or (pdf_paths[0] if pdf_paths else None)
        if primary_pdf and os.path.exists(primary_pdf):
            if self.settings.REMOTE_INFERENCE_URL:
                try:
                    req_kwargs: dict[str, Any] = {
                        "json": {"pdf_path": os.path.basename(primary_pdf), "pages": list(pages_covered)[:3]},
                        "timeout": float(self.settings.REMOTE_TIMEOUT_SECONDS) * 2,
                    }
                    worker_headers = llm_client._worker_headers()
                    if worker_headers:
                        req_kwargs["headers"] = worker_headers
                    layout_resp = llm_client.http_client.post(
                        f"{self.settings.REMOTE_INFERENCE_URL.rstrip('/')}/api/reader/layout",
                        **req_kwargs,
                    )
                    if layout_resp.status_code == 200:
                        layout_data = layout_resp.json()
                        for entry in layout_data.get("extracted_tables", []):
                            if not isinstance(entry, dict):
                                continue  # Legacy string-only detections have no verifiable page.
                            page = entry.get("page_no")
                            table = entry.get("text")
                            if type(page) is int and page in pages_covered and isinstance(table, str) and table.strip():
                                extracted_tables.append((page, table.strip()))
                        document_schema = layout_data.get("document_schema", {})
                        self.logger.info(f"[GPU Finetune Reader] LayoutLMv3 parsed {len(extracted_tables)} tables/structures.")
                except Exception as e:
                    self.logger.debug(f"[GPU Finetune Reader] Remote layout worker skipped ({e}), falling back to local extractor.")

            if not extracted_tables:
                for p in sorted(pages_covered)[:2]:
                    tbl = extract_table_or_equations_tool(primary_pdf, p)
                    if tbl and tbl.strip():
                        extracted_tables.append((p, tbl.strip()))

        # 3. Augment source units with extracted tables if any
        for t_idx, (page, tbl_str) in enumerate(extracted_tables[:2]):
            t_unit = SourceUnit(
                id=f"table_layoutlmv3_{t_idx + 1}",
                text=tbl_str,
                doc_id=str(os.path.basename(primary_pdf) if primary_pdf else "extracted_tables"),
                page_no=page,
                character_offset=curr_offset,
            )
            curr_offset += len(t_unit.text) + 1
            if merge_or_append_unit(t_unit, by_hash, ordered_units):
                merged_dups += 1

        source_units = [u.to_dict() for u in ordered_units]

        trace = self.log_action(
            "gpu_finetune_reader_completed",
            {
                "units_count": len(source_units),
                "pages_covered": list(pages_covered),
                "tables_count": len(extracted_tables),
                "dedup_merged_count": merged_dups,
                "failed_pdf_count": len(failed_pdfs),
                "failed_pdfs": failed_pdfs,
            },
        )

        return {
            "source_units": source_units,
            "document_schema": document_schema,
            "extracted_tables": extracted_tables,
            "reasoning_trace": [trace],
        }
