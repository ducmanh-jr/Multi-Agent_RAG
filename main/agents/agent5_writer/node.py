"""
Agent 5: Writer Agent (Academic Report Synthesizer).
Enforces:
  1. Deterministic Claim Pool Assembly (0% Hallucination by Design).
  2. Academic structured layout (Executive Summary, Verified Findings, Disclaimers, References).
  3. Interactive In-Text Citation Anchors.
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState
from agents.agent5_writer.components.report_builder import build_academic_report


class WriterAgent(BaseAgent):
    """Writer Agent responsible for final report generation and citation formatting."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="WriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        answer = state.get("answer", "")

        self.logger.info(
            f"WriterAgent compiling final report: {len(verified_claims)} verified claims, {len(rejected_claims)} rejected."
        )

        answer_status = state.get("answer_status")
        short_answer = state.get("short_answer")
        evidence_spans = state.get("evidence_spans", []) or []

        statistical_audit = state.get("statistical_audit")
        code_verification = state.get("code_verification")

        final_report, report_data = build_academic_report(
            query=query,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs,
            answer=answer,
            answer_status=answer_status,
            short_answer=short_answer,
            evidence_spans=evidence_spans,
            statistical_audit=statistical_audit,
            code_verification=code_verification,
        )

        trace = self.log_action(
            "report_generation_completed",
            {
                "report_length": len(final_report),
                "verified_claims_count": len(verified_claims),
                "rejected_claims_count": len(rejected_claims),
            },
        )

        confidence = "high" if verified_claims else ("medium" if not rejected_claims else "low")
        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "confidence": confidence,
            "reasoning_trace": [trace],
        }
