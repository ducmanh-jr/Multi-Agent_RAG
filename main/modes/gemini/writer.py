"""
Gemini Mode - Citation-Grounded Writer Agent (L3-L4, Quota-Optimized).
Compiles verified findings, verbatim evidence quotes, and numbered page citations [1], [2].
"""

from __future__ import annotations

from typing import Any

from agents.agent5_writer.components.report_builder import build_academic_report
from core.base_agent import BaseAgent
from core.state import AgentState


class GeminiWriterAgent(BaseAgent):
    """Writer Agent for Gemini Mode: High-precision academic report & citation compiler."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GeminiWriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        evidence_spans = state.get("evidence_spans", []) or []

        self.logger.info(
            f"[Gemini Writer] Compiling verified academic report for: '{query[:60]}' "
            f"({len(verified_claims)} verified claims)"
        )

        final_report, report_data = build_academic_report(
            query=query,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs,
            answer=answer,
            evidence_spans=evidence_spans,
        )
        report_data["engine"] = "Gemini_Cloud_Agentic"

        trace = self.log_action(
            "gemini_report_written",
            {
                "length": len(final_report),
                "verified_claims_count": len(verified_claims),
            },
        )

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
