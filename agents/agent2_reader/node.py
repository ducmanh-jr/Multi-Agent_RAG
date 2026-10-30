"""
Agent 2: Reader Agent (Document Structure & Evidence Unit Parser).
Combines:
  1. DocLayNet-style structural parsing (Sections, Headers, Tables).
  2. Exact character offset tracking and PaperQA2 context hashing.
  3. Consistent SourceUnit schema creation for both PDF uploads and retrieved chunks.
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit
from agents.agent2_reader.tools.pdf_parser import parse_pdf_document


class ReaderAgent(BaseAgent):
    """Reader Agent responsible for structural layout parsing and source unit creation."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="ReaderAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        pdf_path = state.get("pdf_path")
        retrieved_docs = state.get("retrieved_docs", [])

        self.logger.info(f"ReaderAgent processing: pdf_path={pdf_path}, retrieved_docs={len(retrieved_docs)}")

        source_units: list[dict[str, Any]] = []
        document_schema: dict[str, Any] = {}

        # 1. If PDF is supplied, parse layout and verbatim units
        if pdf_path:
            schema, units = parse_pdf_document(pdf_path)
            document_schema = schema
            source_units.extend([u.to_dict() for u in units])

        # 2. Also map retrieved chunks into uniform SourceUnits
        curr_offset = sum(len(u.get("text", "")) for u in source_units)
        for i, doc in enumerate(retrieved_docs):
            text = doc.get("text", "")
            if not text:
                continue
            doc_id = doc.get("doc_id") or doc.get("id") or f"retrieved_{i}"
            page_no = doc.get("page_no", 1)

            unit = SourceUnit(
                id=f"{doc_id}_unit_{i}",
                text=text,
                doc_id=doc_id,
                page_no=page_no,
                character_offset=curr_offset,
            )
            source_units.append(unit.to_dict())
            curr_offset += len(text) + 1

        if not document_schema:
            document_schema = {
                "source": "retrieved_corpus",
                "total_units": len(source_units),
                "sections": [{"unit_id": u["id"], "category": "chunk"} for u in source_units],
            }

        trace = self.log_action(
            "document_reading_completed",
            {
                "has_pdf": bool(pdf_path),
                "total_source_units": len(source_units),
            },
        )

        return {
            "document_schema": document_schema,
            "source_units": source_units,
            "reasoning_trace": [trace],
        }
