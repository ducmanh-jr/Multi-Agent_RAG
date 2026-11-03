"""
CPU Min Mode - Writer Agent.
Fast academic report generation:
  - Formats verified claims with markdown citations.
  - Builds full academic report and bibliography aligned with system report standards.
"""

from __future__ import annotations

from typing import Any

from agents.agent5_writer.components.report_builder import build_academic_report
from core.base_agent import BaseAgent
from core.state import AgentState


class CPUMinWriterAgent(BaseAgent):
    """Writer Agent for CPU Min: Fast academic report & citation compiler."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinWriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        evidence_spans = state.get("evidence_spans", []) or []

        self.logger.info(f"[CPU Min Writer] Compiling report for query: '{query}'")

        final_report, report_data = build_academic_report(
            query=query,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs,
            answer=answer,
            evidence_spans=evidence_spans,
        )
        report_data["engine"] = "CPU_Min_Extractive"

        trace = self.log_action("cpu_min_report_written", {"length": len(final_report)})

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
