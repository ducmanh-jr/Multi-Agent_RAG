"""
Kaggle Max Mode - Writer Agent.
Comprehensive Scientific Paper and Report Generator on Dual T4 GPUs.
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState
from agents.agent5_writer.components.report_builder import build_academic_report


class KaggleMaxWriterAgent(BaseAgent):
    """Writer Agent for Kaggle Max: Comprehensive scholarly paper builder."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleMaxWriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        answer = state.get("answer", "")
        evidence_spans = state.get("evidence_spans", []) or []

        self.logger.info(f"[Kaggle Max Writer] Synthesizing full paper for: '{query}'")

        final_report, report_data = build_academic_report(
            query=query,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs,
            answer=answer,
            evidence_spans=evidence_spans,
        )

        trace = self.log_action("kaggle_max_report_written", {"length": len(final_report)})

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
