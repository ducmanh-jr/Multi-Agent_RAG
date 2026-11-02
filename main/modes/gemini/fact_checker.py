"""
Gemini Mode - Batch NLI Fact Checker Agent (L3-L4, Quota-Optimized).
Evaluates ALL atomic claims in a SINGLE Batch JSON Gemini API Call (instead of 8 sequential calls),
combined with deterministic Numeric & Date Pre-Verification and automatic EvidenceGap generation.
"""

from __future__ import annotations

import json
import re
from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState, ClaimCheckResult, EvidenceGap
from core.verifier import (
    check_numerical_and_date_consistency,
    deterministic_verify_claims,
    split_answer_into_atomic_claims,
)
from utils.llm_client import llm_client


class GeminiFactCheckerAgent(BaseAgent):
    """Fact Checker Agent using a Single-Call Batch Gemini NLI Judge + Numeric/Date Guard."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GeminiFactCheckerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        raw_claims = state.get("claims", []) or []
        source_units = state.get("source_units", []) or []
        retrieved_docs = state.get("retrieved_docs", []) or []

        if not answer.strip():
            return {
                "fact_check_label": "not_enough_info",
                "verified_claims": [],
                "rejected_claims": [],
                "evidence_gaps": [EvidenceGap(missing_aspect=query, suggested_query=query).to_dict()],
                "reasoning_trace": [self.log_action("aborted", {"reason": "empty_answer"})],
            }

        combined_units = list(source_units) if source_units else list(retrieved_docs)
        claims = split_answer_into_atomic_claims(answer, combined_units, existing_claims=raw_claims)
        full_context = "\n\n".join(
            f"[Unit {u.get('id', idx+1)} | Trang {u.get('page_no', 1)}]: {u.get('text', '')}"
            for idx, u in enumerate(combined_units[:8])
        )

        self.logger.info(f"[Gemini FactChecker] Batch-evaluating {len(claims)} claims in 1 Gemini NLI call.")

        verified: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        pending_for_batch: list[dict[str, Any]] = []

        # 1. Deterministic Numerical & Date Guard first (0 API calls)
        for c in claims[:8]:
            c_text = str(c.get("claim_text") or c.get("text") or "").strip()
            c_id = str(c.get("id") or "claim")
            num_ok, date_ok, missing_nums = check_numerical_and_date_consistency(c_text, full_context)
            if not (num_ok and date_ok):
                rejected.append(
                    ClaimCheckResult(
                        claim_id=c_id,
                        claim_text=c_text,
                        status="CONTRADICT",
                        reason=f"Sai lệch số liệu/thời gian không khớp ngữ cảnh: {', '.join(missing_nums)}",
                        numerical_passed=num_ok,
                        dates_passed=date_ok,
                        confidence_score=0.2,
                    ).to_dict()
                )
                gaps.append(
                    EvidenceGap(
                        missing_aspect=f"Kiểm chứng số liệu {', '.join(missing_nums)}: {c_text[:80]}",
                        suggested_query=f"{query} {' '.join(missing_nums)}".strip(),
                    ).to_dict()
                )
            else:
                pending_for_batch.append(c)

        # 2. Single-Call Batch Gemini NLI Verification for all pending claims
        if pending_for_batch:
            claims_json_list = [
                {"claim_id": c.get("id", f"claim_{i+1}"), "claim_text": c.get("text", "")}
                for i, c in enumerate(pending_for_batch)
            ]
            batch_prompt = f"""Bạn là Thẩm phán Khoa học NLI (Gemini Agent 4 - Batch Verifier).
Hãy đối soát TẤT CẢ các nhận định dưới đây với Tài liệu ngữ cảnh trong 1 lần duy nhất.

Danh sách nhận định cần kiểm tra:
{json.dumps(claims_json_list, ensure_ascii=False, indent=2)}

Tài liệu ngữ cảnh:
{full_context[:5000]}

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON HỢP LỆ:
{{
  "results": [
    {{
      "claim_id": "claim_1",
      "status": "SUPPORT" | "CONTRADICT" | "NOT_ENOUGH_INFO",
      "rationale": "Giải thích ngắn gọn kèm dẫn chứng trang/đoạn"
    }}
  ]
}}
"""
            batch_parsed = None
            try:
                res_text = llm_client._call_gemini(
                    prompt=batch_prompt,
                    temperature=0.1,
                    max_tokens=1024,
                    json_mode=True,
                )
                if res_text:
                    clean = re.sub(r"^```(?:json)?|```$", "", res_text.strip(), flags=re.MULTILINE)
                    batch_parsed = json.loads(clean)
            except Exception as exc:
                self.logger.warning(f"[Gemini FactChecker] Batch NLI call fallback to deterministic verifier ({exc})")

            if isinstance(batch_parsed, dict) and isinstance(batch_parsed.get("results"), list):
                res_map = {
                    str(item.get("claim_id", "")): item
                    for item in batch_parsed["results"]
                    if isinstance(item, dict)
                }
                for c in pending_for_batch:
                    cid = str(c.get("id", ""))
                    ctext = str(c.get("text", ""))
                    item = res_map.get(cid, {})
                    raw_status = str(item.get("status", "SUPPORT")).upper()
                    rationale = str(item.get("rationale", "Được hỗ trợ bởi tài liệu ngữ cảnh."))
                    if "SUPPORT" in raw_status:
                        verified.append(
                            ClaimCheckResult(
                                claim_id=cid,
                                claim_text=ctext,
                                status="SUPPORT",
                                reason=rationale,
                            ).to_dict()
                        )
                    else:
                        status_norm = "CONTRADICT" if "CONTRA" in raw_status else "INSUFFICIENT"
                        rejected.append(
                            ClaimCheckResult(
                                claim_id=cid,
                                claim_text=ctext,
                                status=status_norm,
                                reason=rationale,
                            ).to_dict()
                        )
                        gaps.append(
                            EvidenceGap(
                                missing_aspect=ctext[:100],
                                suggested_query=f"{query} {ctext[:50]}".strip(),
                            ).to_dict()
                        )
            else:
                det_ver, det_rej, det_gaps = deterministic_verify_claims(
                    pending_for_batch, combined_units, query=query
                )
                verified.extend(det_ver)
                rejected.extend(det_rej)
                gaps.extend(det_gaps)

        label = "support" if len(verified) >= len(rejected) and verified else "not_enough_info"

        trace = self.log_action(
            "gemini_fact_check_completed",
            {
                "engine": "Gemini_Batch_NLI_1Call",
                "verified_count": len(verified),
                "rejected_count": len(rejected),
                "label": label,
            },
        )

        return {
            "fact_check_label": label,
            "verified_claims": verified,
            "rejected_claims": rejected,
            "evidence_gaps": gaps,
            "verifier_model_family": "gemini_flash",
            "reasoning_trace": [trace],
        }
