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
from core.base_agent import BaseAgent
from core.state import AgentState, SourceUnit


def clean_source_text(text: str) -> str:
    """Deeply cleans raw PDF/OCR text, removes HTML line breaks, and separates glued words safely."""
    if not text:
        return ""
    # 1. Replace HTML breaks with space
    s = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)

    # 2. Fix markdown underscores in words (_Báo Thủ đô_ -> Báo Thủ đô), preserving code snake_case like proxy_pass
    s = re.sub(r"(?<![A-Za-z0-9])_+|_+(?![A-Za-z0-9])", "", s)

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
        pdf_path = state.get("pdf_path")
        pdf_paths = state.get("pdf_paths") or ([pdf_path] if pdf_path else [])
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[CPU Max Reader] Building SourceUnits from {len(retrieved_docs)} ranked chunks "
            f"(docs_count={len(pdf_paths)})"
        )

        source_units: list[dict[str, Any]] = []
        curr_offset = 0

        # 1. Convert top ranked retrieved_docs into clean SourceUnits
        for idx, doc in enumerate(retrieved_docs):
            raw_text = str(doc.get("text") or doc.get("content") or "").strip()
            if not raw_text:
                continue
            text = clean_source_text(raw_text)
            if not text:
                continue

            doc_id = str(doc.get("doc_id") or doc.get("title") or f"doc_{idx + 1}")
            page_no = int(doc.get("page_no") or doc.get("page") or 1)
            unit = SourceUnit(
                id=str(doc.get("id") or f"{doc_id}_p{page_no}_u{idx + 1}"),
                text=text,
                doc_id=doc_id,
                page_no=page_no,
                character_offset=curr_offset,
            )
            source_units.append(unit.to_dict())
            curr_offset += len(text) + 1

        # 2. Fallback to direct PDF parsing if no chunks retrieved
        if not source_units and pdf_paths:
            for p in pdf_paths:
                if p and os.path.exists(str(p)):
                    d_id = os.path.splitext(os.path.basename(str(p)))[0]
                    p_units, _ = parse_pdf_document(str(p))
                    for u in p_units:
                        u_clean_text = clean_source_text(str(u.get("text", "")))
                        unit = SourceUnit(
                            id=str(u.get("id") or f"{d_id}_p{u.get('page_no', 1)}"),
                            text=u_clean_text,
                            doc_id=d_id,
                            page_no=int(u.get("page_no", 1)),
                            character_offset=curr_offset,
                        )
                        source_units.append(unit.to_dict())
                        curr_offset += len(u_clean_text) + 1

        trace = self.log_action(
            "cpu_max_reading_completed",
            {
                "units_count": len(source_units),
                "docs_count": len(pdf_paths),
            },
        )

        return {
            "source_units": source_units,
            "document_schema": {},
            "reasoning_trace": [trace],
        }
