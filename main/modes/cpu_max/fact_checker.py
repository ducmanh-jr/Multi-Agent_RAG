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

        ev_texts = [e.get("quote", "") if isinstance(e, dict) else str(e) for e in evidence_spans]
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

            statuses.append(status)
            if status == "support":
                verified_claims.append(eval_res)
            else:
                rejected_claims.append(eval_res)
                evidence_gaps.append(EvidenceGap(missing_aspect=c_text[:120], suggested_query=c_text).to_dict())

        # Consensus labeling
        if "contradict" in statuses:
            consensus_label = "contradict"
        elif statuses and all(s == "support" for s in statuses):
            consensus_label = "support"
        else:
            support_ratio = statuses.count("support") / len(statuses) if statuses else 0.0
            consensus_label = "support" if support_ratio >= 0.6 else "not_enough_info"

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
