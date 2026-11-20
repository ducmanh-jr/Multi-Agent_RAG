"""
Agent 4: Fact-Checker Agent (Scientific NLI Judge & Deterministic Guardrails).
Combines:
  1. SciFact-style NLI Verification (SUPPORT / CONTRADICT / NOT_ENOUGH_INFO).
  2. Deterministic Numerical Verification (prevents metric hallucinations).
  3. Gap Detection (identifies specific missing evidence for iterative re-retrieval).
"""

from __future__ import annotations

import re


from typing import Any, Optional
from core.base_agent import BaseAgent
from core.schemas import map_evidence_to_legacy_fact_check_label
from core.state import AgentState, ClaimCheckResult, EvidenceGap
from core.verifier import (
    check_numerical_and_date_consistency,
    verify_evidence_span_provenance,
)
from agents.agent4_fact_checker.components.claim_extractor import ClaimExtractor
from agents.agent4_fact_checker.components.judge import FactCheckJudge
from agents.agent4_fact_checker.components.dates import verify_dates_in_claim
from agents.agent4_fact_checker.components.statistical_auditor import audit_statistical_methodology
from tools.python_sandbox import (
    execute_python_code,
    verify_attention_matrix_shapes,
    verify_lora_decomposition,
)
from agents.agent4_fact_checker.components.quote_verifier import (
    verify_quote_in_source_units,
    classify_answer_and_status,
)


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
        source_units = state.get("source_units", []) or []

        self.logger.info(f"[{self.name}] Kiểm chứng factual grounding (Retry: {retry_count}).")

        # Extract textual evidence strings
        ev_texts = [e.get("quote", "") if isinstance(e, dict) else str(e) for e in evidence_spans]

        # Combine text from retrieved_docs, seen_chunks, and evidence_spans for thorough evidence checking
        all_sources = list(retrieved_docs) + [c for c in seen_chunks if c not in retrieved_docs]
        unit_texts = [u.get("text", "") for u in source_units if isinstance(u, dict)]
        context_corpus = " ".join([d.get("text", "") for d in all_sources] + ev_texts + unit_texts)

        # 1. Edge Case: Empty Answer
        if not answer or not answer.strip():
            self.logger.warning(f"[{self.name}] Câu trả lời rỗng! Gán nhãn 'NOT_FOUND'.")
            trace = self.log_action("fact_check_aborted", {"reason": "Empty answer"})
            return {
                "claims": [],
                "verified_claims": [],
                "rejected_claims": [],
                "evidence_gaps": [EvidenceGap(missing_aspect="Nội dung trống", suggested_query=query).to_dict()],
                "evidence_status": "NOT_FOUND",
                "processing_status": "SUCCESS",
                "fact_check_label": "not_enough_info",
                "reasoning_trace": [trace],
            }

        # 2. Extract or Normalize Claims (preserving linked evidence_ids if present)
        normalized_claim_items: list[dict[str, Any]] = []
        if raw_claims:
            for c in raw_claims:
                if isinstance(c, dict):
                    text = str(c.get("text") or c.get("claim") or "").strip()
                    ev_ids = list(c.get("evidence_ids") or [])
                    if c.get("unit_id") and c.get("unit_id") not in ev_ids:
                        ev_ids.append(c.get("unit_id"))
                    if text:
                        normalized_claim_items.append({"text": text, "evidence_ids": ev_ids})
                else:
                    text = str(c).strip()
                    if text:
                        normalized_claim_items.append({"text": text, "evidence_ids": []})
        if not normalized_claim_items:
            extracted = self.extractor.extract_claims(answer)
            normalized_claim_items = [{"text": t, "evidence_ids": []} for t in extracted if t]
        if not normalized_claim_items:
            normalized_claim_items = [{"text": answer.strip(), "evidence_ids": []}]

        claims_to_check = [item["text"] for item in normalized_claim_items]

        # 2b. Independent Citation Provenance Verification on all evidence_spans
        span_provenance_results: list[tuple[dict[str, Any], bool, str]] = []
        for sp in evidence_spans:
            if isinstance(sp, dict):
                ok_prov, prov_msg = verify_evidence_span_provenance(sp, source_units)
                span_provenance_results.append((sp, ok_prov, prov_msg))

        # 3. Evaluate Each Claim Independently (NLI + Unit/Context Numerical + Dates + Provenance)
        verified_claims = []
        rejected_claims = []
        evidence_gaps = []
        statuses = []
        nli_contradictions = 0

        for item in normalized_claim_items:
            c_text = item["text"]
            linked_ev_ids = set(str(x) for x in item.get("evidence_ids") or [])

            # Unit-aware & context-aware numerical + year check
            nums_ok, num_dates_ok, missing_nums = check_numerical_and_date_consistency(c_text, context_corpus)
            num_reason = (
                "Numerical check passed."
                if nums_ok
                else f"Số liệu/đơn vị không khớp nguồn: {', '.join(missing_nums)}"
            )

            # Date & temporal verification
            dates_ok_legacy, date_reason = verify_dates_in_claim(c_text, context_corpus)
            dates_ok = bool(num_dates_ok and dates_ok_legacy)
            if not num_dates_ok and not date_reason:
                date_reason = f"Năm không khớp nguồn: {', '.join(missing_nums)}"

            # Provenance check for spans linked to this claim (or overall spans when unlinked)
            if span_provenance_results:
                relevant_spans = [
                    r for r in span_provenance_results
                    if not linked_ev_ids
                    or str(r[0].get("id") or "") in linked_ev_ids
                    or str(r[0].get("source_unit_id") or r[0].get("unit_id") or "") in linked_ev_ids
                ]
                if not relevant_spans:
                    relevant_spans = span_provenance_results
                provenance_ok = any(r[1] for r in relevant_spans)
                prov_reason = (
                    "Citation provenance verified."
                    if provenance_ok
                    else "; ".join(r[2] for r in relevant_spans[:2])
                )
            else:
                # No explicit evidence spans supplied -> cannot claim citation provenance is verified
                provenance_ok = bool(source_units) and not bool(evidence_spans)
                prov_reason = "No evidence_spans provided; checked against source_units."

            # NLI evaluation
            eval_res = self.judge.judge_claim(c_text, ev_texts, context_corpus)
            raw_status = str(eval_res.get("status") or "not_enough_info").lower()
            status = raw_status if raw_status in ("support", "contradict", "not_enough_info") else "not_enough_info"

            if status == "contradict":
                nli_contradictions += 1

            # Missing numerical/date/provenance grounding downgrades 'support' to 'not_enough_info'
            # (never mislabels missing evidence as 'contradict'!)
            if not nums_ok:
                if status == "support":
                    status = "not_enough_info"
                eval_res["reasoning"] = f"{eval_res.get('reasoning', '')} | {num_reason}"

            if not dates_ok:
                if status == "support":
                    status = "not_enough_info"
                eval_res["reasoning"] = f"{eval_res.get('reasoning', '')} | {date_reason}"

            if span_provenance_results and not provenance_ok:
                if status == "support":
                    status = "not_enough_info"
                eval_res["reasoning"] = f"{eval_res.get('reasoning', '')} | {prov_reason}"

            statuses.append(status)
            claim_result = {
                "claim": c_text,
                "status": status,
                "confidence": eval_res.get("confidence", 0.9),
                "evidence": eval_res.get("evidence", ""),
                "reasoning": eval_res.get("reasoning", ""),
                "numerical_passed": nums_ok,
                "dates_passed": dates_ok,
                "provenance_ok": provenance_ok,
                "numeric_valid": nums_ok,  # legacy alias
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

        # 4. Per-Claim Evidence Aggregation (FOUND / PARTIAL / NOT_FOUND / CONFLICTING_EVIDENCE)
        total_claims = len(statuses)
        support_cnt = statuses.count("support")
        contradict_cnt = statuses.count("contradict")
        nei_cnt = statuses.count("not_enough_info")

        if total_claims == 0:
            evidence_status = "NOT_FOUND"
            summary_reason = "Không có nhận định nào để kiểm chứng."
        elif contradict_cnt > 0 and (support_cnt == 0 or contradict_cnt >= support_cnt):
            evidence_status = "CONFLICTING_EVIDENCE"
            summary_reason = f"Phát hiện {contradict_cnt}/{total_claims} nhận định mâu thuẫn trực tiếp với bằng chứng nguồn."
        elif support_cnt == total_claims:
            # All claims fully supported -> FOUND (never downgraded solely because context was truncated!)
            evidence_status = "FOUND"
            summary_reason = f"Toàn bộ {support_cnt}/{total_claims} nhận định đã được kiểm chứng đầy đủ."
        elif support_cnt > 0:
            evidence_status = "PARTIAL"
            summary_reason = (
                f"Hỗ trợ một phần: {support_cnt}/{total_claims} nhận định được xác minh, "
                f"{nei_cnt} thiếu bằng chứng, {contradict_cnt} mâu thuẫn."
            )
        else:
            evidence_status = "NOT_FOUND"
            summary_reason = f"Không tìm thấy bằng chứng xác thực cho {nei_cnt}/{total_claims} nhận định."

        overall_label = map_evidence_to_legacy_fact_check_label(evidence_status)

        context_coverage = state.get("context_coverage") or {}
        self.logger.info(
            f"[{self.name}] Phán quyết: Evidence={evidence_status} (legacy={overall_label.upper()}) — {summary_reason}"
        )

        trace = self.log_action(
            "fact_checking_verdict",
            {
                "evidence_status": evidence_status,
                "fact_check_label": overall_label,
                "total_claims": total_claims,
                "supported": support_cnt,
                "contradicted": contradict_cnt,
                "not_enough_info": nei_cnt,
                "provenance_spans_checked": len(span_provenance_results),
                "provenance_spans_valid": sum(1 for r in span_provenance_results if r[1]),
                "context_truncated": bool(context_coverage.get("truncated", False)),
                "summary": summary_reason,
            },
        )
        trace["fact_check_label"] = overall_label
        trace["evidence_status"] = evidence_status

        # 5. Senior Peer-Reviewer Statistical Audit
        stat_audit = audit_statistical_methodology(claims_to_check, context_corpus)

        # 6. Formal Mathematical / Tensor Sandbox Verification (Grounded in query/claim numbers)
        code_verif = None
        full_text_to_audit = f"{query} {' '.join(claims_to_check)}"
        
        has_lora_math = any(kw in full_text_to_audit.lower() for kw in ["lora", "low-rank", "phân rã ma trận"]) and bool(re.search(r"\b(?:rank|r\s*=|\d+\s*(?:x|×)\s*\d+)\b", full_text_to_audit, re.IGNORECASE))
        has_attn_math = any(kw in full_text_to_audit.lower() for kw in ["attention", "transformer", "flashattention"]) and bool(re.search(r"\b(?:heads?|qk\^t|chiều ma trận|num_heads)\b", full_text_to_audit, re.IGNORECASE))

        if has_lora_math:
            r_match = re.search(r"\b(?:rank|r)\s*[=:]?\s*(\d+)\b", full_text_to_audit, re.IGNORECASE)
            dim_match = re.search(r"\b(?:dim|dimension|d)\s*[=:]?\s*(\d+)\b", full_text_to_audit, re.IGNORECASE)
            if r_match:
                rank_val = int(r_match.group(1))
                dim_val = int(dim_match.group(1)) if dim_match else 4096
                self.logger.info(f"[{self.name}] Kích hoạt Python Sandbox kiểm chứng LoRA rank={rank_val}, dim={dim_val}...")
                code_verif = verify_lora_decomposition(d_in=dim_val, d_out=dim_val, rank=rank_val)
        elif has_attn_math:
            heads_match = re.search(r"\b(?:heads?|h)\s*[=:]?\s*(\d+)\b", full_text_to_audit, re.IGNORECASE)
            dim_match = re.search(r"\b(?:dim|d_model|d)\s*[=:]?\s*(\d+)\b", full_text_to_audit, re.IGNORECASE)
            if heads_match:
                heads_val = int(heads_match.group(1))
                dim_val = int(dim_match.group(1)) if dim_match else 64
                self.logger.info(f"[{self.name}] Kích hoạt Python Sandbox kiểm chứng Attention heads={heads_val}, dim={dim_val}...")
                code_verif = verify_attention_matrix_shapes(d_model=dim_val, num_heads=heads_val)

        # 7. Classify Answer Status and Generate Short Answer
        status_category, short_ans = classify_answer_and_status(
            query=query,
            verified_claims=verified_claims,
            rejected_claims=rejected_claims,
            source_units=source_units,
        )

        return {
            "answer": answer,
            "claims": [{"id": f"claim_{i+1}", "text": c} for i, c in enumerate(claims_to_check)],
            "verified_claims": verified_claims,
            "rejected_claims": rejected_claims,
            "evidence_gaps": evidence_gaps,
            "evidence_status": evidence_status,
            "processing_status": "SUCCESS",
            "fact_check_label": overall_label,
            "answer_status": status_category,
            "short_answer": short_ans,
            "statistical_audit": stat_audit,
            "code_verification": code_verif,
            "reasoning_trace": [trace],
        }
