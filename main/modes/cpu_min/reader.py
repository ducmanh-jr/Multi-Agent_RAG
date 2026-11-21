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

from core.base_agent import BaseAgent
from core.citation_audit import citation_location, evidence_quote
from core.state import AgentState, SourceUnit, compute_context_hash
from core.tools import extract_table_or_equations_tool, load_or_parse_selected_pdf


def sanitize_unit_text(text: str) -> str:
    """Strip raw HTML tags, line breaks, and table markdown artifacts from source unit text."""
    if not text:
        return ""
    s = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    s = re.sub(r"<[^>]+>", " ", s)
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
        pdf_path = state.get("pdf_path")
        pdf_paths = state.get("pdf_paths") or ([pdf_path] if pdf_path else [])
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[CPU Min Reader] Building SourceUnits from {len(retrieved_docs)} ranked chunks "
            f"(docs_count={len(pdf_paths)})"
        )

        source_units: list[dict[str, Any]] = []
        curr_offset = 0
        pages_covered: set[int] = set()

        # 1. Convert top retrieved chunks into SourceUnits
        for idx, doc in enumerate(retrieved_docs[:16]):
            raw_text = str(doc.get("text") or doc.get("content") or "").strip()
            clean_text = sanitize_unit_text(raw_text)
            if not clean_text:
                continue
            doc_id = str(doc.get("doc_id") or doc.get("title") or f"doc_{idx + 1}")
            page_no = int(doc.get("page_no") or doc.get("page") or 1)
            pages_covered.add(page_no)
            orig_offset = doc.get("character_offset") or doc.get("start_offset") or doc.get("offset")
            unit_offset = int(orig_offset) if orig_offset is not None else curr_offset
            unit = SourceUnit(
                id=str(doc.get("id") or f"{doc_id}_p{page_no}_u{idx + 1}"),
                text=clean_text,
                doc_id=doc_id,
                page_no=page_no,
                character_offset=unit_offset,
            )
            source_units.append(unit.to_dict())
            curr_offset = unit_offset + len(clean_text) + 1

        # 2. Fallback if retrieved_docs was empty but pdf_paths exist
        if allow_pdf_fallback and not source_units and pdf_paths:
            for p in pdf_paths:
                if not p or not os.path.exists(p):
                    continue
                pdf_chunks = load_or_parse_selected_pdf(p)
                for idx, ch in enumerate(pdf_chunks[:10]):
                    raw_text = str(ch.get("text", "")).strip()
                    clean_text = sanitize_unit_text(raw_text)
                    if not clean_text:
                        continue
                    orig_offset = ch.get("character_offset") or ch.get("start_offset") or ch.get("offset")
                    unit_offset = int(orig_offset) if orig_offset is not None else curr_offset
                    unit = SourceUnit(
                        id=str(ch.get("id", f"u_{len(source_units) + 1}")),
                        text=clean_text,
                        doc_id=str(ch.get("doc_id", os.path.basename(p))),
                        page_no=int(ch.get("page_no", 1)),
                        character_offset=unit_offset,
                    )
                    source_units.append(unit.to_dict())
                    curr_offset = unit_offset + len(clean_text) + 1

        document_schema = {
            "doc_id": source_units[0]["doc_id"] if source_units else "unknown",
            "pages_covered": sorted(list(pages_covered)),
            "engine": "Sanitized_SourceUnit_Pipeline",
            "total_units": len(source_units),
        }

        trace = self.log_action(
            "cpu_min_reading_completed",
            {
                "engine": "Sanitized_SourceUnit_Pipeline",
                "total_source_units": len(source_units),
                "pages_covered": sorted(list(pages_covered)),
            },
        )

        return {
            "document_schema": document_schema,
            "source_units": source_units,
            "reasoning_trace": [trace],
        }
