"""
CPU Min Mode - Agentic Reader Agent (L3-Lite).
Converts ranked retrieved chunks from the selected PDF (or corpus) into SourceUnits
with exact page numbers, context hashes, and table extraction on numeric/comparative queries.
"""

from __future__ import annotations

import os
from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit
from core.tools import extract_table_or_equations_tool, load_or_parse_selected_pdf


class CPUMinReaderAgent(BaseAgent):
    """Reader Agent optimized for CPU Min: Ranked chunk-to-SourceUnit builder + fast table extraction."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        retrieved_docs = state.get("retrieved_docs", []) or []
        is_numeric = bool(state.get("is_numeric_query", False))
        is_comparative = bool(state.get("is_comparative_query", False))

        self.logger.info(
            f"[CPU Min Reader] Building SourceUnits from {len(retrieved_docs)} ranked chunks "
            f"(pdf_path={os.path.basename(pdf_path) if pdf_path else 'None'})"
        )

        source_units: list[dict[str, Any]] = []
        curr_offset = 0
        pages_covered: set[int] = set()

        # 1. Convert top retrieved chunks into SourceUnits first (preserves Retrieval ranking!)
        for idx, doc in enumerate(retrieved_docs[:10]):
            text = str(doc.get("text") or doc.get("content") or "").strip()
            if not text:
                continue
            doc_id = str(doc.get("doc_id") or doc.get("title") or f"doc_{idx + 1}")
            page_no = int(doc.get("page_no") or doc.get("page") or 1)
            pages_covered.add(page_no)
            unit = SourceUnit(
                id=str(doc.get("id") or f"{doc_id}_p{page_no}_u{idx + 1}"),
                text=text,
                doc_id=doc_id,
                page_no=page_no,
                character_offset=curr_offset,
            )
            source_units.append(unit.to_dict())
            curr_offset += len(text) + 1

        # 2. If query is numeric/comparative and a PDF is selected, extract tables from the top hit pages
        if pdf_path and os.path.exists(str(pdf_path)) and (is_numeric or is_comparative) and pages_covered:
            doc_id = os.path.splitext(os.path.basename(str(pdf_path)))[0]
            for p_no in sorted(list(pages_covered))[:2]:
                table_md = extract_table_or_equations_tool(pdf_path, p_no)
                if table_md and "Bảng" in table_md:
                    unit = SourceUnit(
                        id=f"{doc_id}_p{p_no}_table",
                        text=table_md,
                        doc_id=doc_id,
                        page_no=p_no,
                        character_offset=curr_offset,
                    )
                    source_units.append(unit.to_dict())
                    curr_offset += len(table_md) + 1

        # 3. Fallback if retrieved_docs was somehow empty but pdf_path exists
        if not source_units and pdf_path:
            pdf_chunks = load_or_parse_selected_pdf(pdf_path)
            for idx, ch in enumerate(pdf_chunks[:8]):
                text = str(ch.get("text", "")).strip()
                if not text:
                    continue
                unit = SourceUnit(
                    id=str(ch.get("id", f"u_{idx + 1}")),
                    text=text,
                    doc_id=str(ch.get("doc_id", "selected_pdf")),
                    page_no=int(ch.get("page_no", 1)),
                    character_offset=curr_offset,
                )
                source_units.append(unit.to_dict())
                curr_offset += len(text) + 1

        document_schema = {
            "doc_id": source_units[0]["doc_id"] if source_units else "unknown",
            "pages_covered": sorted(list(pages_covered)),
            "engine": "PyMuPDF_Ranked_Window",
            "total_units": len(source_units),
        }

        trace = self.log_action(
            "cpu_min_reading_completed",
            {
                "engine": "PyMuPDF_Ranked_Window",
                "total_source_units": len(source_units),
                "pages_covered": sorted(list(pages_covered)),
            },
        )

        return {
            "document_schema": document_schema,
            "source_units": source_units,
            "reasoning_trace": [trace],
        }
