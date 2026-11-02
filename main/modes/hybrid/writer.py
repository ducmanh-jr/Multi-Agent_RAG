"""
Hybrid Mode - Citation-Grounded Writer Agent (L4).
Compiles the final academic report with inline citations [1], [2], verified claims,
and Cross-Model Verification provenance metadata.
"""

from __future__ import annotations

from typing import Any

from agents.agent5_writer.components.report_builder import build_academic_report
from core.base_agent import BaseAgent
from core.state import AgentState


class HybridWriterAgent(BaseAgent):
    """Flagship Hybrid Writer Agent: Compiles verified findings + Cross-Model Audit Badge."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="HybridWriterAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        verified_claims = state.get("verified_claims", []) or []
        rejected_claims = state.get("rejected_claims", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []
        answer = state.get("answer", "")
        evidence_spans = state.get("evidence_spans", []) or []
        gen_family = state.get("generator_model_family", "auto")
        ver_family = state.get("verifier_model_family", "cross_model_guard")

        self.logger.info(
            f"[Hybrid Writer] Compiling report: Generator={gen_family} | Verifier={ver_family} | "
            f"Verified={len(verified_claims)}"
        )

        final_report, report_data = build_academic_report(
            query=query,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            retrieved_docs=retrieved_docs,
            answer=answer,
            evidence_spans=evidence_spans,
        )
        report_data["engine"] = "Hybrid_CrossModel_L4"
        report_data["generator_model_family"] = gen_family
        report_data["verifier_model_family"] = ver_family

        trace = self.log_action(
            "hybrid_report_written",
            {
                "report_length": len(final_report),
                "generator_model_family": gen_family,
                "verifier_model_family": ver_family,
                "verified_claims_count": len(verified_claims),
            },
        )

        return {
            "final_report": final_report,
            "written_report_data": report_data,
            "reasoning_trace": [trace],
        }
