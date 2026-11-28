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

from agents.agent2_reader.tools.reader_pipeline import (
    build_source_units_from_docs,
    collect_unique_pdf_paths,
    evaluate_reader_table_completeness,
    ingest_pdf_fallbacks,
)
from core.base_agent import BaseAgent
from core.citation_audit import citation_location, evidence_quote
from core.claim_semantics import strip_html_tags_preserve_math
from core.state import AgentState, SourceUnit, compute_context_hash
from core.tools import extract_table_or_equations_tool, load_or_parse_selected_pdf


def sanitize_unit_text(text: str) -> str:
    """Strip raw HTML tags, line breaks, and table markdown artifacts from source unit text."""
    if not text:
        return ""
    s = strip_html_tags_preserve_math(text)
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
        pdf_paths = collect_unique_pdf_paths(state.get("pdf_path"), state.get("pdf_paths"))
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(
            f"[CPU Min Reader] Building SourceUnits from {len(retrieved_docs)} ranked chunks "
            f"(docs_count={len(pdf_paths)})"
        )

        # Keep 1:1 chunk traceability for single-doc comparison reader; dedup in unified multi-PDF mode
        dedup = bool(allow_pdf_fallback)
        read_ctx: dict[str, Any] = {
            "ordered_units": [],
            "by_hash": {},
            "pages_covered": set(),
            "merged_dups": 0,
            "curr_offset": 0,
            "failed_pdfs": [],
            "source_units": [],
            "reader_decision": None,
            "unresolved_tables": [],
        }

        def _tool_sanitize_source_units() -> int:
            ordered_u, by_h, pages_c, merged_d, offset = build_source_units_from_docs(
                retrieved_docs,
                limit=16,
                text_cleaner=sanitize_unit_text,
                deduplicate=dedup,
            )
            read_ctx.update(
                ordered_units=ordered_u,
                by_hash=by_h,
                pages_covered=pages_c,
                merged_dups=merged_d,
                curr_offset=offset,
            )
            return len(ordered_u)

        def _tool_pdf_fallback_ingest() -> int:
            failed, extra_merged, offset = ingest_pdf_fallbacks(
                pdf_paths,
                ordered_units=read_ctx["ordered_units"],
                units_by_content_hash=read_ctx["by_hash"],
                pages_covered=read_ctx["pages_covered"],
                curr_offset=read_ctx["curr_offset"],
                chunk_loader=load_or_parse_selected_pdf,
                text_cleaner=sanitize_unit_text,
                max_units_per_pdf=10,
                deduplicate=True,
            )
            read_ctx["failed_pdfs"] = failed
            read_ctx["merged_dups"] += extra_merged
            read_ctx["curr_offset"] = offset
            return len(read_ctx["ordered_units"])

        def _tool_table_completeness_audit() -> int:
            read_ctx["source_units"] = [u.to_dict() for u in read_ctx["ordered_units"]]
            r_dec, unresolved = evaluate_reader_table_completeness(
                state, read_ctx["source_units"], allow_handoff=allow_pdf_fallback
            )
            read_ctx["reader_decision"] = r_dec
            read_ctx["unresolved_tables"] = unresolved
            return len(unresolved)

        def _reader_tool_planner(step_idx: int, history: list[dict[str, Any]]):
            if step_idx == 0:
                return ("sanitize_source_units", {})
            last_tool = history[-1]["tool"] if history else ""
            if last_tool == "sanitize_source_units" and allow_pdf_fallback and not read_ctx["ordered_units"] and pdf_paths:
                return ("pdf_fallback_ingest", {})
            if last_tool in {"sanitize_source_units", "pdf_fallback_ingest"}:
                return ("table_completeness_audit", {})
            return None

        tool_loop = self.run_bounded_tool_loop(
            {
                "sanitize_source_units": _tool_sanitize_source_units,
                "pdf_fallback_ingest": _tool_pdf_fallback_ingest,
                "table_completeness_audit": _tool_table_completeness_audit,
            },
            _reader_tool_planner,
            max_steps=3,
        )

        pages_covered = read_ctx["pages_covered"]
        merged_dups = read_ctx["merged_dups"]
        failed_pdfs = read_ctx["failed_pdfs"]
        source_units = read_ctx["source_units"]
        reader_decision = read_ctx["reader_decision"]
        unresolved_tables = read_ctx["unresolved_tables"]

        document_schema = {
            "doc_id": source_units[0]["doc_id"] if source_units else "unknown",
            "pages_covered": sorted(list(pages_covered)),
            "engine": "Sanitized_SourceUnit_Pipeline",
            "total_units": len(source_units),
            "failed_pdfs": failed_pdfs,
            "dedup_merged_count": merged_dups,
            "unresolved_table_references": unresolved_tables,
        }

        trace = self.log_action(
            "cpu_min_reading_completed",
            {
                "engine": "Sanitized_SourceUnit_Pipeline",
                "tools_used": tool_loop["tools_used"],
                "total_source_units": len(source_units),
                "pages_covered": sorted(list(pages_covered)),
                "dedup_merged_count": merged_dups,
                "failed_pdf_count": len(failed_pdfs),
                "failed_pdfs": failed_pdfs,
                "proposed_handoff": reader_decision.action if reader_decision else None,
                "unresolved_table_references": unresolved_tables,
            },
        )

        result: dict[str, Any] = {
            "document_schema": document_schema,
            "source_units": source_units,
            "tools_used": tool_loop["tools_used"],
            "reasoning_trace": [trace],
        }
        if reader_decision is not None:
            result["agent_decision"] = reader_decision
        return result
