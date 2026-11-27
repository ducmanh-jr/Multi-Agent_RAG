"""
CPU Max Mode - Fact Checker Agent.
Full Scientific NLI Verification:
  - MiniCheck-style NLI Judge.
  - Deterministic Numeric and Date Verifiers.
  - Statistical Methodology Auditor & Python Sandbox.
"""

from __future__ import annotations

from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState, EvidenceGap
from core.verifier import review_abstention, review_claims
from agents.agent4_fact_checker.components.claim_extractor import ClaimExtractor
from agents.agent4_fact_checker.components.judge import FactCheckJudge
from agents.agent4_fact_checker.components.numeric_verifier import verify_numbers_in_claim
from agents.agent4_fact_checker.components.dates import verify_dates_in_claim


class CPUMaxFactCheckerAgent(BaseAgent):
    """Fact Checker Agent for CPU Max: Full scientific NLI and validation suite."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="CPUMaxFactCheckerAgent", max_internal_retry=max_internal_retry)
        self.extractor = ClaimExtractor()
        self.judge = FactCheckJudge()

    def run(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        raw_claims = state.get("claims", [])
        evidence_spans = state.get("evidence_spans", [])
        retrieved_docs = state.get("retrieved_docs", [])
        seen_chunks = state.get("seen_chunks", []) or []
        retry_count = state.get("retry_count", 0)
        query = state.get("query", "")
        source_units = state.get("source_units", []) or []

        self.logger.info(f"[CPU Max FactChecker] Verifying claims (Retry: {retry_count})")

        if not answer or not answer.strip():
            trace = self.log_action("fact_check_aborted", {"reason": "Empty answer"})
            return {
                "claims": [],
                "verified_claims": [],
                "rejected_claims": [],
                "evidence_gaps": [EvidenceGap(missing_aspect="Nội dung trống", suggested_query=query).to_dict()],
                "fact_check_label": "not_enough_info",
                "reasoning_trace": [trace],
            }

        if self.judge.classifier is None:
            result = review_claims(state)
            result["verifier_model_family"] = "cpu_max_deterministic_nli"
            result.setdefault("reasoning_trace", []).append(
                self.log_action("cpu_max_fact_check_completed", {"label": result["fact_check_label"]}))
            return result
        refusal = review_abstention(state)
        if refusal is not None:
            return refusal

        # Even when a local HuggingFace NLI classifier is loaded, run deterministic semantic/math/derived guards
        det_review = review_claims(state)
        det_by_text = {
            (c.get("claim_text") or c.get("text") or ""): c
            for c in (det_review.get("verified_claims") or []) + (det_review.get("rejected_claims") or [])
            if isinstance(c, dict)
        }

        original_text = [str(u.get("text") or "") for u in (source_units or retrieved_docs)]
        ev_texts = [e["quote"] for e in evidence_spans if isinstance(e, dict) and e.get("quote")
                    and any(e["quote"] in text for text in original_text)]
        all_sources = list(retrieved_docs) + [c for c in seen_chunks if c not in retrieved_docs]
        unit_texts = [u.get("text", "") for u in source_units if isinstance(u, dict)]
        context_corpus = " ".join([d.get("text", "") for d in all_sources] + ev_texts + unit_texts)

        claims_to_check = []
        if raw_claims:
            for c in raw_claims:
                text = c.get("text") if isinstance(c, dict) else str(c)
                if text:
                    claims_to_check.append(text)
        if not claims_to_check:
            claims_to_check = self.extractor.extract_claims(answer)
        if not claims_to_check:
            claims_to_check = [answer.strip()]

        verified_claims = []
        rejected_claims = []
        evidence_gaps = []
        statuses = []

        for c_text in claims_to_check:
            det_hit = det_by_text.get(c_text)
            if det_hit and det_hit.get("claim_type") == "derived_claim" and str(det_hit.get("status")).upper() == "SUPPORT":
                verified_claims.append(det_hit)
                statuses.append("support")
                continue
            if det_hit and str(det_hit.get("status")).upper() == "CONTRADICT":
                rejected_claims.append({**det_hit, "status": "contradict"})
                evidence_gaps.append(EvidenceGap(missing_aspect=c_text[:120], suggested_query=c_text).to_dict())
                statuses.append("contradict")
                continue

            nums_ok, num_reason = verify_numbers_in_claim(c_text, context_corpus)
            dates_ok, date_reason = verify_dates_in_claim(c_text, context_corpus)

            eval_res = self.judge.judge_claim(c_text, ev_texts, context_corpus)
            eval_res["claim"] = c_text
            eval_res["text"] = c_text
            status = eval_res["status"]

            if not nums_ok:
                status = "not_enough_info"
                eval_res["rationale"] = f"Cảnh báo số liệu: {num_reason}"
            elif not dates_ok:
                status = "not_enough_info"
                eval_res["rationale"] = f"Cảnh báo mốc thời gian: {date_reason}"

            eval_res["status"] = status
            statuses.append(status)
            if status == "support":
                verified_claims.append(eval_res)
            else:
                rejected_claims.append(eval_res)
                evidence_gaps.append(EvidenceGap(missing_aspect=c_text[:120], suggested_query=c_text).to_dict())

        consensus_label = "support" if verified_claims and not rejected_claims else "not_enough_info"
        if "contradict" in statuses:
            consensus_label = "contradict"

        trace = self.log_action(
            "cpu_max_fact_check_completed",
            {
                "verified_count": len(verified_claims),
                "rejected_count": len(rejected_claims),
                "consensus_label": consensus_label,
            },
        )

        return {
            "verified_claims": verified_claims,
            "rejected_claims": rejected_claims,
            "evidence_gaps": evidence_gaps,
            "fact_check_label": consensus_label,
            "reasoning_trace": [trace],
        }
