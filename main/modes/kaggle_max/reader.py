"""
Kaggle Max Mode - Agentic Reader Agent (L3-L4 32B Reasoning on Dual T4).
Converts ranked retrieved chunks from the selected PDF into SourceUnits and extracts
structured Markdown tables on top hit pages.
"""

from __future__ import annotations

import os
from typing import Any

from agents.agent2_reader.tools.pdf_parser import parse_pdf_document
from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit
from core.tools import extract_table_or_equations_tool


class KaggleMaxReaderAgent(BaseAgent):
    """Reader Agent for Kaggle Max Mode: Ranked SourceUnit builder + Table Extractor."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleMaxReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[Kaggle Max Reader] Building SourceUnits from {len(retrieved_docs)} chunks "
            f"(pdf_path={os.path.basename(pdf_path) if pdf_path else 'None'})"
        )

        source_units: list[dict[str, Any]] = []
        document_schema: dict[str, Any] = {}
        curr_offset = 0
        pages_covered: set[int] = set()

        for idx, d in enumerate(retrieved_docs[:10]):
            t = str(d.get("text") or d.get("content") or "").strip()
            if not t:
                continue
            d_id = str(d.get("doc_id") or d.get("title") or f"doc_{idx + 1}")
            p_no = int(d.get("page_no") or d.get("page") or 1)
            pages_covered.add(p_no)
            unit = SourceUnit(
                id=str(d.get("id") or f"{d_id}_p{p_no}_u{idx + 1}"),
                text=t,
                doc_id=d_id,
                page_no=p_no,
                character_offset=curr_offset,
            )
            source_units.append(unit.to_dict())
            curr_offset += len(t) + 1

        if pdf_path and os.path.exists(str(pdf_path)):
            doc_id = os.path.splitext(os.path.basename(str(pdf_path)))[0]
            for p_no in sorted(list(pages_covered))[:3]:
                table_md = extract_table_or_equations_tool(pdf_path, p_no)
                if table_md and ("Bảng" in table_md and "|" in table_md):
                    unit = SourceUnit(
                        id=f"{doc_id}_p{p_no}_table",
                        text=table_md,
                        doc_id=doc_id,
                        page_no=p_no,
                        character_offset=curr_offset,
                    )
                    source_units.append(unit.to_dict())
                    curr_offset += len(table_md) + 1

            if not source_units:
                schema, units = parse_pdf_document(pdf_path)
                document_schema = schema
                source_units.extend([u.to_dict() for u in units[:12]])

        if not document_schema:
            document_schema = {
                "doc_id": source_units[0]["doc_id"] if source_units else "unknown",
                "pages_covered": sorted(list(pages_covered)),
                "total_units": len(source_units),
            }

        trace = self.log_action(
            "kaggle_max_reading_completed",
            {
                "has_pdf": bool(pdf_path),
                "total_units": len(source_units),
                "pages_covered": sorted(list(pages_covered)),
            },
        )

        return {
            "document_schema": document_schema,
            "source_units": source_units,
            "reasoning_trace": [trace],
        }
