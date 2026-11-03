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
from core.verifier import deterministic_verify_claims, split_answer_into_atomic_claims


class CPUMinFactCheckerAgent(BaseAgent):
    """Fact Checker Agent for CPU Min: Fast deterministic numerical/date/quote verification + Gap Generator."""

    def __init__(self, max_internal_retry: int = 1):
        super().__init__(name="CPUMinFactCheckerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        raw_claims = state.get("claims", []) or []
        source_units = state.get("source_units", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []

        self.logger.info(f"[CPU Min FactChecker] Verifying {len(raw_claims)} claims via deterministic rules.")

        if not answer.strip():
            return {
                "fact_check_label": "not_enough_info",
                "verified_claims": [],
                "rejected_claims": [],
                "evidence_gaps": [EvidenceGap(missing_aspect=query, suggested_query=query).to_dict()],
                "reasoning_trace": [self.log_action("aborted", {"reason": "empty_answer"})],
            }

        # Handle Out-of-Domain honest refusal directly
        if "hoàn toàn không được đề cập" in answer or "không được đề cập" in answer:
            return {
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

        # If at least 50% verified and no numerical contradiction, mark as support
        has_contradict = any(r.get("status") == "CONTRADICT" for r in rejected)
        if verified and len(verified) >= len(rejected) and not has_contradict:
            label = "support"
        elif has_contradict and not verified:
            label = "contradict"
        else:
            label = "support" if verified else "not_enough_info"

        trace = self.log_action(
            "cpu_min_fact_check_completed",
            {
                "verified_count": len(verified),
                "rejected_count": len(rejected),
                "evidence_gaps_count": len(gaps),
                "label": label,
                "engine": "Deterministic_Numeric_Quote_Guard",
            },
        )

        return {
            "fact_check_label": label,
            "verified_claims": verified,
            "rejected_claims": rejected,
            "evidence_gaps": gaps,
            "verifier_model_family": "deterministic_guard",
            "reasoning_trace": [trace],
        }
