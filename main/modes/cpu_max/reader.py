"""
CPU Max Mode - Agentic Reader Agent (L3).
High-precision Structural & Table Parsing:
  - Prioritizes ranked retrieved chunks from the selected PDF / corpus.
  - Automatically runs PyMuPDF structural table & equation extraction on top relevant pages.
  - Produces uniform SourceUnits with SHA-256 context hashes and page numbers.
"""

from __future__ import annotations

import os
from typing import Any

from agents.agent2_reader.tools.pdf_parser import parse_pdf_document
from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit
from core.tools import extract_table_or_equations_tool


class CPUMaxReaderAgent(BaseAgent):
    """Reader Agent for CPU Max: Ranked SourceUnit builder + structural table parser."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="CPUMaxReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[CPU Max Reader] Parsing: pdf_path={os.path.basename(pdf_path) if pdf_path else 'None'}, "
            f"retrieved_docs={len(retrieved_docs)}"
        )

        source_units: list[dict[str, Any]] = []
        document_schema: dict[str, Any] = {}
        curr_offset = 0
        pages_covered: set[int] = set()

        # 1. Convert top ranked retrieved_docs into SourceUnits first (keeps top-k relevance order!)
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

        # 2. Extract Markdown tables from the top hit pages of the selected PDF
        if pdf_path and os.path.exists(str(pdf_path)):
            doc_id = os.path.splitext(os.path.basename(str(pdf_path)))[0]
            for p_no in sorted(list(pages_covered))[:3]:
                table_md = extract_table_or_equations_tool(pdf_path, p_no)
                if table_md and ("Bảng" in table_md or "Dữ liệu số học" in table_md):
                    unit = SourceUnit(
                        id=f"{doc_id}_p{p_no}_table",
                        text=table_md,
                        doc_id=doc_id,
                        page_no=p_no,
                        character_offset=curr_offset,
                    )
                    source_units.append(unit.to_dict())
                    curr_offset += len(table_md) + 1

            # Fallback if retrieved_docs was empty
            if not source_units:
                schema, units = parse_pdf_document(pdf_path)
                document_schema = schema
                source_units.extend([u.to_dict() for u in units[:12]])

        if not document_schema:
            document_schema = {
                "doc_id": source_units[0]["doc_id"] if source_units else "unknown",
                "pages_covered": sorted(list(pages_covered)),
                "engine": "CPUMax_Ranked_Structural_Reader",
                "total_units": len(source_units),
            }

        trace = self.log_action(
            "cpu_max_reading_completed",
            {
                "has_pdf": bool(pdf_path),
                "total_source_units": len(source_units),
                "pages_covered": sorted(list(pages_covered)),
            },
        )

        return {
            "document_schema": document_schema,
            "source_units": source_units,
            "reasoning_trace": [trace],
        }
