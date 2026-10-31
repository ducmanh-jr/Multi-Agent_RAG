"""
Agent 4: Fact-Checker Agent (Scientific NLI Judge & Deterministic Guardrails).
Combines:
  1. SciFact-style NLI Verification (SUPPORT / CONTRADICT / NOT_ENOUGH_INFO).
  2. Deterministic Numerical Verification (prevents metric hallucinations).
  3. Gap Detection (identifies specific missing evidence for iterative re-retrieval).
"""

from __future__ import annotations

from typing import Any, Optional
from core.base_agent import BaseAgent
from core.state import AgentState, ClaimCheckResult, EvidenceGap
from agents.agent4_fact_checker.components.claim_extractor import ClaimExtractor
from agents.agent4_fact_checker.components.judge import FactCheckJudge
from agents.agent4_fact_checker.components.numeric_verifier import verify_numbers_in_claim
from agents.agent4_fact_checker.components.dates import verify_dates_in_claim


class FactCheckerAgent(BaseAgent):
    """Fact-Checker Agent responsible for scientific verification and consensus labeling."""

    def __init__(self, model_path: Optional[str] = None, max_internal_retry: int = 2):
        super().__init__(name="FactCheckerAgent", max_internal_retry=max_internal_retry)
        self.extractor = ClaimExtractor()
        self.judge = FactCheckJudge(model_path=model_path)
        self.logger.info(f"[{self.name}] Khởi tạo FactCheckerAgent hoàn tất.")

    def run(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        raw_claims = state.get("claims", [])
        evidence_spans = state.get("evidence_spans", [])
        retrieved_docs = state.get("retrieved_docs", [])
        seen_chunks = state.get("seen_chunks", []) or []
        retry_count = state.get("retry_count", 0)
        query = state.get("query", "")

        self.logger.info(f"[{self.name}] Kiểm chứng factual grounding (Retry: {retry_count}).")

        # Extract textual evidence strings
        ev_texts = [e.get("quote", "") if isinstance(e, dict) else str(e) for e in evidence_spans]

        # Combine text from retrieved_docs, seen_chunks, and evidence_spans for thorough evidence checking
        all_sources = list(retrieved_docs) + [c for c in seen_chunks if c not in retrieved_docs]
        context_corpus = " ".join([d.get("text", "") for d in all_sources] + ev_texts)

        # 1. Edge Case: Empty Answer
        if not answer or not answer.strip():
            self.logger.warning(f"[{self.name}] Câu trả lời rỗng! Gán nhãn 'not_enough_info'.")
            trace = self.log_action("fact_check_aborted", {"reason": "Empty answer"})
            return {
                "claims": [],
                "verified_claims": [],
                "rejected_claims": [],
                "evidence_gaps": [EvidenceGap(missing_aspect="Nội dung trống", suggested_query=query).to_dict()],
                "fact_check_label": "not_enough_info",
                "reasoning_trace": [trace],
            }

        # 2. Extract or Normalize Claims
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

        # 3. Check Each Claim with NLI Judge & Numeric Verifier
        verified_claims = []
        rejected_claims = []
        evidence_gaps = []
        statuses = []

        for c_text in claims_to_check:
            # Numeric verification
            nums_ok, num_reason = verify_numbers_in_claim(c_text, context_corpus)
            # Date & temporal verification
            dates_ok, date_reason = verify_dates_in_claim(c_text, context_corpus)

            # NLI evaluation
            eval_res = self.judge.judge_claim(c_text, ev_texts, context_corpus)
            status = eval_res["status"]

            # If numbers or dates are ungrounded in context, downgrade support to not_enough_info
            if not nums_ok:
                if status == "support":
                    status = "not_enough_info"
                eval_res["reasoning"] = f"{eval_res.get('reasoning', '')} | {num_reason}"

            if not dates_ok:
                if status == "support":
                    status = "not_enough_info"
                eval_res["reasoning"] = f"{eval_res.get('reasoning', '')} | {date_reason}"

            statuses.append(status)
            claim_result = {
                "claim": c_text,
                "status": status,
                "confidence": eval_res.get("confidence", 0.9),
                "evidence": eval_res.get("evidence", ""),
                "reasoning": eval_res.get("reasoning", ""),
                "numeric_valid": nums_ok,
            }

            if status == "support":
                verified_claims.append(claim_result)
            else:
                rejected_claims.append(claim_result)
                evidence_gaps.append(
                    EvidenceGap(
                        missing_aspect=f"Xác minh nhận định: {c_text[:80]}",
                        suggested_query=f"{c_text[:60]}",
                    ).to_dict()
                )

        # 4. Consensus Decision Logic
        total_claims = len(statuses)
        support_cnt = statuses.count("support")
        contradict_cnt = statuses.count("contradict")
        nei_cnt = statuses.count("not_enough_info")

        if contradict_cnt > 0:
            overall_label = "contradict"
            summary_reason = f"Phát hiện {contradict_cnt}/{total_claims} nhận định mâu thuẫn hoặc số liệu không khớp nguồn."
        elif nei_cnt >= (total_claims / 2):
            overall_label = "not_enough_info"
            summary_reason = f"Có {nei_cnt}/{total_claims} nhận định thiếu bằng chứng xác thực."
        elif support_cnt > (total_claims / 2):
            overall_label = "support"
            summary_reason = f"Đa số {support_cnt}/{total_claims} nhận định đã được kiểm chứng chuẩn xác."
        else:
            overall_label = "not_enough_info"
            summary_reason = "Chưa đạt đa số nhận định có chứng cứ rõ ràng."

        self.logger.info(f"[{self.name}] Phán quyết: {overall_label.upper()} ({summary_reason})")

        trace = self.log_action(
            "fact_checking_verdict",
            {
                "fact_check_label": overall_label,
                "total_claims": total_claims,
                "supported": support_cnt,
                "contradicted": contradict_cnt,
                "not_enough_info": nei_cnt,
                "summary": summary_reason,
            },
        )
        trace["fact_check_label"] = overall_label

        return {
            "claims": [{"id": f"claim_{i+1}", "text": c} for i, c in enumerate(claims_to_check)],
            "verified_claims": verified_claims,
            "rejected_claims": rejected_claims,
            "evidence_gaps": evidence_gaps,
            "fact_check_label": overall_label,
            "reasoning_trace": [trace],
        }
