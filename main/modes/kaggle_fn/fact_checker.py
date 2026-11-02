"""
Kaggle FN Mode - Agentic Fact Checker Agent (L3 Multi-LoRA + Fine-Tuned NLI Specialist).
Combines:
  1. Deterministic Numerical & Date Guard.
  2. Remote Kaggle /api/factcheck endpoint running fine-tuned mDeBERTa-v3-base-xnli (agent4_nli)
     and 'critic' QLoRA adapter in batch mode.
"""

from __future__ import annotations

from typing import Any

from core.base_agent import BaseAgent
from core.state import AgentState, ClaimCheckResult, EvidenceGap
from core.verifier import (
    TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
    check_numerical_and_date_consistency,
    deterministic_verify_claims,
    split_answer_into_atomic_claims,
)
from utils.llm_client import llm_client


class KaggleFNFactCheckerAgent(BaseAgent):
    """Fact Checker Agent using Kaggle fine-tuned NLI ('agent4_nli') + 'critic' QLoRA."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="KaggleFNFactCheckerAgent", max_internal_retry=max_internal_retry)

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
        full_context = "\n\n".join(str(u.get("text", "")) for u in combined_units[:8])

        self.logger.info(f"[Kaggle FN FactChecker] Auditing {len(claims)} claims via fine-tuned NLI + 'critic' LoRA.")

        verified: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        gaps: list[dict[str, Any]] = []
        pending_claims: list[dict[str, Any]] = []

        # 1. Deterministic Numerical & Date Pre-Check
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
                        reason=f"Sai lệch số liệu/thời gian không có trong văn bản gốc: {', '.join(missing_nums)}",
                        numerical_passed=num_ok,
                        dates_passed=date_ok,
                        confidence_score=0.2,
                    ).to_dict()
                )
                gaps.append(
                    EvidenceGap(
                        missing_aspect=f"Kiểm chứng số liệu {', '.join(missing_nums)}",
                        suggested_query=f"{query} {' '.join(missing_nums)}".strip(),
                    ).to_dict()
                )
            else:
                pending_claims.append(c)

        # 2. Remote Batch NLI / Critic verification on Kaggle GPU Worker
        if pending_claims and self.settings.REMOTE_INFERENCE_URL:
            try:
                fc_res = llm_client.call_kaggle_specialist(
                    endpoint="api/factcheck",
                    payload={
                        "query": query,
                        "answer": answer,
                        "claims": [c.get("text", "") for c in pending_claims],
                        "evidence": [str(u.get("text", "")) for u in combined_units[:6]],
                        "adapter": "critic",
                    },
                    fallback_prompt=(
                        f'Context:\n"""\n{full_context[:3500]}\n"""\n\n'
                        f"Question: {query}\n\nGenerated Summary:\n{answer}"
                    ),
                    fallback_system=TRAINING_LAB_CRITIC_SYSTEM_PROMPT,
                    adapter="critic",
                )
                if isinstance(fc_res, dict) and "verified_claims" in fc_res:
                    for vc in fc_res.get("verified_claims", []):
                        txt = vc.get("claim") or vc.get("claim_text") or str(vc)
                        verified.append(ClaimCheckResult(claim_text=txt, status="SUPPORT", reason="Validated by Kaggle NLI + Critic LoRA").to_dict())
                    for rc in fc_res.get("rejected_claims", []):
                        txt = rc.get("claim") or rc.get("claim_text") or str(rc)
                        rejected.append(ClaimCheckResult(claim_text=txt, status="CONTRADICT", reason="Rejected by Kaggle NLI + Critic LoRA").to_dict())
                        gaps.append(EvidenceGap(missing_aspect=txt[:90], suggested_query=f"{query} {txt[:45]}").to_dict())
                elif isinstance(fc_res, dict) and "is_supported" in fc_res:
                    unsup_list = fc_res.get("Unsupported_claims") or fc_res.get("unsupported_claims") or []
                    for c in pending_claims:
                        ctext = str(c.get("text", ""))
                        if any(u.lower()[:30] in ctext.lower() for u in unsup_list if isinstance(u, str) and len(u) > 10):
                            rejected.append(ClaimCheckResult(claim_text=ctext, status="CONTRADICT", reason="Flagged by Critic LoRA").to_dict())
                            gaps.append(EvidenceGap(missing_aspect=ctext[:90], suggested_query=f"{query} {ctext[:45]}").to_dict())
                        else:
                            verified.append(ClaimCheckResult(claim_text=ctext, status="SUPPORT", reason="Approved by Critic LoRA").to_dict())
                else:
                    det_v, det_r, det_g = deterministic_verify_claims(pending_claims, combined_units, query=query)
                    verified.extend(det_v)
                    rejected.extend(det_r)
                    gaps.extend(det_g)
            except Exception as exc:
                self.logger.warning(f"[Kaggle FN FactChecker] Fallback to deterministic verifier: {exc}")
                det_v, det_r, det_g = deterministic_verify_claims(pending_claims, combined_units, query=query)
                verified.extend(det_v)
                rejected.extend(det_r)
                gaps.extend(det_g)
        elif pending_claims:
            det_v, det_r, det_g = deterministic_verify_claims(pending_claims, combined_units, query=query)
            verified.extend(det_v)
            rejected.extend(det_r)
            gaps.extend(det_g)

        label = "support" if len(verified) >= len(rejected) and verified else "not_enough_info"

        trace = self.log_action(
            "kaggle_fn_fact_check_completed",
            {
                "verified_count": len(verified),
                "rejected_count": len(rejected),
                "adapter": "critic + agent4_nli",
                "label": label,
            },
        )

        return {
            "fact_check_label": label,
            "verified_claims": verified,
            "rejected_claims": rejected,
            "evidence_gaps": gaps,
            "verifier_model_family": "mdeberta_nli_and_critic_qlora",
            "reasoning_trace": [trace],
        }
