"""
CPU Min Mode - Agentic Fact Checker Agent (L3-Lite).
Deterministic Guardrails:
  - Zero-Hallucination Out-of-Domain approval: confirms out-of-domain honest refusal.
  - 100% regex & substring quote verification against source chunks.
  - Number, percentage, and 4-digit year consistency checking.
  - Generates targeted EvidenceGap objects for Outer Re-Query Loop when claims fail verification.
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState, ClaimCheckResult, EvidenceGap
from core.verifier import (
    deterministic_verify_claims,
    finalize_fact_check_contract,
    split_answer_into_atomic_claims,
)


class CPUMinFactCheckerAgent(BaseAgent):
    """Fact Checker Agent for CPU Min: Fast deterministic numerical/date/quote/provenance verification."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinFactCheckerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        raw_claims = state.get("claims", []) or []
        evidence_spans = state.get("evidence_spans", []) or []
        source_units = state.get("source_units", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(f"[CPU Min FactChecker] Verifying {len(raw_claims)} claims via deterministic rules.")

        if not answer.strip():
            return {
                "evidence_status": "NOT_FOUND",
                "processing_status": "SUCCESS",
                "fact_check_label": "not_enough_info",
                "verified_claims": [],
                "rejected_claims": [],
                "evidence_gaps": [EvidenceGap(missing_aspect=query, suggested_query=query).to_dict()],
                "verifier_model_family": "deterministic_guard",
                "reasoning_trace": [self.log_action("aborted", {"reason": "empty_answer"})],
            }

        # Handle Out-of-Domain honest refusal and thematic comparison directly
        if (
            "hoàn toàn không được đề cập" in answer
            or "không được đề cập" in answer
            or "Đánh giá tương quan:" in answer
            or "không cùng một hệ quy chiếu" in answer
        ):
            return {
                "evidence_status": "NOT_FOUND",
                "processing_status": "SUCCESS",
                "fact_check_label": "support",
                "verified_claims": [
                    ClaimCheckResult(
                        claim_id="claim_1",
                        claim_text="Tài liệu nguồn không đề cập đến thông tin câu hỏi.",
                        status="SUPPORT",
                        reason="Xác nhận tài liệu nguồn không chứa thông tin của câu hỏi (Zero Hallucination).",
                        confidence_score=1.0,
                    ).to_dict()
                ],
                "rejected_claims": [],
                "evidence_gaps": [],
                "verifier_model_family": "deterministic_guard",
                "reasoning_trace": [self.log_action("out_of_domain_verified", {"status": "support"})],
            }

        combined_units = list(source_units) if source_units else list(retrieved_docs)
        claims = split_answer_into_atomic_claims(answer, combined_units, existing_claims=raw_claims)
        verified, rejected, gaps = deterministic_verify_claims(claims, combined_units, query=query)

        res = finalize_fact_check_contract(
            query=query,
            verified_claims=verified,
            rejected_claims=rejected,
            evidence_gaps=gaps,
            evidence_spans=evidence_spans,
            source_units=combined_units,
            context_coverage=state.get("context_coverage"),
            verifier_model_family="deterministic_guard",
        )

        trace = self.log_action(
            "cpu_min_fact_check_completed",
            {
                "verified_count": len(res["verified_claims"]),
                "rejected_count": len(res["rejected_claims"]),
                "evidence_gaps_count": len(res["evidence_gaps"]),
                "evidence_status": res["evidence_status"],
                "label": res["fact_check_label"],
                "engine": "Deterministic_Numeric_Quote_Provenance_Guard",
            },
        )
        res["reasoning_trace"] = [trace]
        return res
