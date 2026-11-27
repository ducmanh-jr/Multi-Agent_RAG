"""
GPU Finetune Mode - Agentic Reader Agent (7-Model SOTA Pipeline).
Leverages Remote GPU Worker powered by:
  - Fine-Tuned LayoutLMv3 Base (02_agent2_reader):
    Predicts 11 2D Layout Classes (Title, Text, Table, Figure, Equation, Header, Footer, Caption, List, Metadata, Footnote)
    to perform high-fidelity structural parsing of scientific papers.
  - Generates verified SourceUnits with exact page_no, offsets, and extracted Markdown tables.
"""

from __future__ import annotations

import base64
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
from core.pdf_ingest import is_parser_error_text
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
                max_units_per_pdf=10,
                deduplicate=True,
            )
            merged_dups += extra_merged

        # 2. LayoutLMv3 / 2D Document Layout Analysis (sends real page/crop PNGs and never injects parser errors!)
        extracted_tables: list[tuple[int, str, list[float] | None]] = []
        primary_pdf = pdf_path or (pdf_paths[0] if pdf_paths else None)
        if primary_pdf and os.path.exists(primary_pdf):
            if self.settings.REMOTE_INFERENCE_URL:
                try:
                    import fitz

                    images_payload: list[dict[str, Any]] = []
                    try:
                        with fitz.open(primary_pdf) as doc:
                            for p_no in sorted(pages_covered)[:3]:
                                if 1 <= p_no <= len(doc):
                                    pg = doc[p_no - 1]
                                    scale = min(1.5, 1200.0 / max(pg.rect.width, pg.rect.height, 1.0))
                                    png_b = pg.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False).tobytes("png")
                                    images_payload.append({
                                        "page_no": p_no,
                                        "kind": "full_page",
                                        "image_b64": base64.b64encode(png_b).decode("ascii"),
                                    })
                    except Exception:
                        images_payload = []

                    req_kwargs: dict[str, Any] = {
                        "json": {
                            "pdf_path": os.path.basename(primary_pdf),
                            "pages": list(pages_covered)[:3],
                            "images": images_payload,
                            "include_full_page_context": True,
                        },
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
                                continue
                            page = entry.get("page_no")
                            table = entry.get("text")
                            tbbox = entry.get("bbox") if isinstance(entry.get("bbox"), list) else None
                            if (
                                type(page) is int
                                and page in pages_covered
                                and isinstance(table, str)
                                and table.strip()
                                and not is_parser_error_text(table)
                            ):
                                extracted_tables.append((page, table.strip(), tbbox))
                        document_schema = layout_data.get("document_schema", {})
                        if isinstance(layout_data.get("regions"), list):
                            document_schema["layout_regions"] = layout_data["regions"]
                        self.logger.info(f"[GPU Finetune Reader] Layout worker parsed {len(extracted_tables)} tables/structures.")
                except Exception as e:
                    self.logger.debug(f"[GPU Finetune Reader] Remote layout worker skipped ({e}), falling back to local extractor.")

            if not extracted_tables:
                for p in sorted(pages_covered)[:2]:
                    tbl = extract_table_or_equations_tool(primary_pdf, p)
                    if tbl and not is_parser_error_text(tbl):
                        extracted_tables.append((p, tbl.strip(), None))

        # 3. Augment source units with extracted tables (strictly excluding parser failure messages)
        for t_idx, (page, tbl_str, tbbox) in enumerate(extracted_tables[:2]):
            if is_parser_error_text(tbl_str):
                continue
            t_unit = SourceUnit(
                id=f"table_layoutlmv3_{t_idx + 1}",
                text=tbl_str,
                doc_id=str(os.path.basename(primary_pdf) if primary_pdf else "extracted_tables"),
                page_no=page,
                bbox=tbbox,
                character_offset=curr_offset,
                unit_type="table",
                source_path=primary_pdf,
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
