"""
Summarizer Critic Component (Fail-Closed).
Audits the draft summary and claims against evidence to eliminate hallucinations before passing to Fact-Checker.
On exception or invalid schema, fails CLOSED (`VERIFICATION_FAILED`, `critic_action='verification_failed'`,
`confidence='low'`, empty approved claims) — never auto-approves unverified claims.
"""

from __future__ import annotations

from typing import Any
from utils.llm_client import llm_client
from utils.logger import logger


def build_verification_failed_state(
    answer: str,
    reason: str,
    context_coverage: dict[str, Any] | None = None,
    model_family: str = "unknown",
    trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds a full AgentState update dict for a fail-closed Critic verification failure across all modes."""
    base = _build_verification_failed_result(answer, reason)
    base["evidence_spans"] = []
    base["generator_model_family"] = model_family
    if context_coverage is not None:
        base["context_coverage"] = context_coverage
    if trace is not None:
        base["reasoning_trace"] = [trace]
    return base


def _build_verification_failed_result(answer: str, reason: str) -> dict[str, Any]:
    """Fail-closed contract payload when Critic raises or returns an invalid schema."""
    return {
        "answer": answer,
        "claims": [],
        "critic_action": "verification_failed",
        "critic_feedback": {
            "is_supported": False,
            "action": "verification_failed",
            "confidence": 0.0,
            "unsupported_claims": [],
            "missing_key_points": [],
            "critique_rationale": reason,
        },
        "processing_status": "VERIFICATION_FAILED",
        "error_stage": "summarizer_critic",
        "error_code": "CRITIC_VERIFICATION_FAILED",
        "error_message": reason,
        "confidence": "low",
        "warnings": [f"VERIFICATION_FAILED: {reason}"],
    }


def evaluate_critic_contract(
    answer: str,
    claims: list[dict[str, Any]],
    evidence_spans: list[dict[str, Any]],
    *,
    precomputed_critic_data: dict[str, Any] | None = None,
    critic_exception: Exception | None = None,
    use_fast_critic: bool = False,
    source_units: list[dict[str, Any]] | None = None,
    query: str = "",
    required_entities: list[str] | None = None,
) -> dict[str, Any]:
    """
    Unified Fail-Closed Critic Contract Evaluator shared across ALL 6 execution modes:
      - Mode 1 (Default LLM Critic): Calls `llm_client.generate_json` and validates `approved_claim_ids` / `critic_warnings`.
      - Mode 2 (`precomputed_critic_data` / `critic_exception`): Validates single-call or remote QLoRA/32B/Gemini Critic JSON payloads.
      - Mode 3 (`use_fast_critic=True`): Runs deterministic `evaluate_inner_critic_fast` with strict action/schema validation.
    Strictly fail-closed (`processing_status='VERIFICATION_FAILED'`, `critic_action='verification_failed'`,
    `confidence='low'`, `claims=[]`) on any exception or schema violation.
    """
    if critic_exception is not None:
        logger.error(f"[Critic] External critic exception (fail-closed): {critic_exception}")
        return _build_verification_failed_result(answer, f"Critic exception: {critic_exception}")

    if use_fast_critic:
        try:
            from core.verifier import evaluate_inner_critic_fast

            fast_out = evaluate_inner_critic_fast(
                query=query,
                draft_answer=answer,
                source_units=list(source_units or []),
                required_entities=list(required_entities or []),
            )
            valid_actions = ("approve", "rewrite", "request_more_context", "lower_confidence")
            if not hasattr(fast_out, "action") or fast_out.action not in valid_actions:
                return _build_verification_failed_result(
                    answer, f"Invalid fast critic action: {getattr(fast_out, 'action', None)!r}"
                )
            return {
                "answer": answer,
                "claims": list(claims or []),
                "critic_action": fast_out.action,
                "critic_feedback": fast_out.to_dict(),
                "processing_status": "SUCCESS",
                "error_stage": None,
                "error_code": None,
                "error_message": None,
                "confidence": "high" if fast_out.action == "approve" else "low",
                "warnings": list(fast_out.unsupported_claims or []),
            }
        except Exception as exc:
            logger.error(f"[Critic] Fast critic exception (fail-closed): {exc}")
            return _build_verification_failed_result(answer, f"Fast critic exception: {exc}")

    if precomputed_critic_data is not None:
        if not isinstance(precomputed_critic_data, dict) or precomputed_critic_data.get("error"):
            reason = f"Invalid precomputed critic payload: {precomputed_critic_data!r}"
            logger.error(f"[Critic] {reason}")
            return _build_verification_failed_result(answer, reason)

        # If payload uses approved_claim_ids schema, delegate to standard list validation below
        if "approved_claim_ids" in precomputed_critic_data and "critic_warnings" in precomputed_critic_data:
            data = precomputed_critic_data
        else:
            valid_actions = ("approve", "accept", "rewrite", "request_more_context", "lower_confidence")
            raw_act = precomputed_critic_data.get("critic_action") or precomputed_critic_data.get("action")
            has_critic_keys = any(
                k in precomputed_critic_data for k in ("critic_action", "action", "is_supported")
            )
            if not has_critic_keys or (
                raw_act is not None and str(raw_act).strip().lower() not in valid_actions
            ):
                reason = f"Critic schema violation in precomputed payload: {precomputed_critic_data!r}"
                logger.error(f"[Critic] {reason}")
                return _build_verification_failed_result(answer, reason)

            from core.schemas import SummarizerCriticOutput

            norm_act = str(raw_act or ("approve" if precomputed_critic_data.get("is_supported", True) else "rewrite")).strip().lower()
            if norm_act == "accept":
                norm_act = "approve"
            critic_out = SummarizerCriticOutput.from_dict(
                {
                    "is_supported": precomputed_critic_data.get("is_supported", norm_act == "approve"),
                    "unsupported_claims": (
                        precomputed_critic_data.get("unsupported_claims")
                        or precomputed_critic_data.get("Unsupported_claims")
                        or []
                    ),
                    "missing_key_points": (
                        precomputed_critic_data.get("missing_key_points")
                        or precomputed_critic_data.get("Missing_key_points")
                        or []
                    ),
                    "action": norm_act,
                    "confidence": precomputed_critic_data.get("confidence", 0.9 if norm_act == "approve" else 0.45),
                    "critique_rationale": (
                        precomputed_critic_data.get("critique_rationale")
                        or precomputed_critic_data.get("critique")
                        or precomputed_critic_data.get("reason")
                        or ""
                    ),
                }
            )
            return {
                "answer": answer,
                "claims": list(claims or []),
                "critic_action": critic_out.action,
                "critic_feedback": critic_out.to_dict(),
                "processing_status": "SUCCESS",
                "error_stage": None,
                "error_code": None,
                "error_message": None,
                "confidence": "high" if critic_out.action == "approve" else "low",
                "warnings": list(critic_out.unsupported_claims or []),
            }
    else:
        data = None

    if not claims:
        return {
            "answer": answer,
            "claims": [],
            "critic_action": "request_more_context",
            "critic_feedback": {
                "is_supported": False,
                "action": "request_more_context",
                "confidence": 0.0,
                "unsupported_claims": [],
                "missing_key_points": ["Không có nhận định nào được tạo ra."],
                "critique_rationale": "Không có nhận định nào được tạo ra.",
            },
            "processing_status": "SUCCESS",
            "error_stage": None,
            "error_code": None,
            "error_message": None,
            "confidence": "low",
            "warnings": ["Không có nhận định nào được tạo ra."],
        }

    if data is None:
        prompt = f"""Bạn là Tác tử Phê bình Độc lập (Agent 3 - Critic).
Nhiệm vụ của bạn là kiểm tra xem bản thảo câu trả lời và các nhận định có bám sát bằng chứng trích dẫn hay không.
Nếu một nhận định không có bằng chứng hỗ trợ rõ ràng hoặc nói quá (overclaim), hãy LOẠI BỎ hoặc CẢNH BÁO.

Bản thảo:
{answer}

Danh sách nhận định:
{claims}

Bằng chứng trích dẫn:
{evidence_spans}

YÊU CẦU TRẢ VỀ JSON:
{{
  "approved_claim_ids": ["claim_1", ...],
  "critic_warnings": ["Lý do nếu có nhận định bị loại..."],
  "action": "approve" | "rewrite" | "request_more_context" | "lower_confidence"
}}
"""
        try:
            data = llm_client.generate_json(prompt, agent="summarizer")
        except Exception as exc:
            logger.error(f"[Critic] Exception during verification (fail-closed): {exc}")
            return _build_verification_failed_result(answer, f"Critic exception: {exc}")

    # Strict schema validation — fail-closed on malformed output
    if not isinstance(data, dict) or data.get("error"):
        reason = f"Invalid critic output schema: {data!r}"
        logger.error(f"[Critic] {reason}")
        return _build_verification_failed_result(answer, reason)

    raw_approved = data.get("approved_claim_ids")
    raw_warnings = data.get("critic_warnings", [])
    if not isinstance(raw_approved, list) or not isinstance(raw_warnings, list):
        reason = "Critic schema violation: 'approved_claim_ids' and 'critic_warnings' must be lists."
        logger.error(f"[Critic] {reason}")
        return _build_verification_failed_result(answer, reason)

    approved_ids = {str(x) for x in raw_approved}
    warnings = [str(w) for w in raw_warnings if str(w).strip()]

    filtered = [
        c
        for i, c in enumerate(claims)
        if str(c.get("id") or c.get("claim_id") or f"c_{i}") in approved_ids
    ]
    rejected_texts = [
        str(c.get("text") or c.get("claim") or "")
        for i, c in enumerate(claims)
        if str(c.get("id") or c.get("claim_id") or f"c_{i}") not in approved_ids
    ]

    raw_action = str(data.get("action") or "").strip().lower()
    if not filtered:
        warnings.append("Toàn bộ các nhận định bị loại bỏ do thiếu căn cứ bằng chứng (Zero Hallucination).")
        critic_action = raw_action if raw_action in ("request_more_context", "rewrite") else "request_more_context"
        confidence_label = "low"
        conf_score = 0.2
    elif rejected_texts:
        critic_action = raw_action if raw_action in ("approve", "rewrite", "lower_confidence", "request_more_context") else "lower_confidence"
        confidence_label = "high" if critic_action == "approve" else "low"
        conf_score = round(len(filtered) / max(len(claims), 1), 2)
    else:
        critic_action = raw_action if raw_action in ("approve", "rewrite", "lower_confidence", "request_more_context") else "approve"
        confidence_label = "high" if critic_action == "approve" else "low"
        conf_score = 0.95

    return {
        "answer": answer,
        "claims": filtered,
        "critic_action": critic_action,
        "critic_feedback": {
            "is_supported": len(filtered) > 0 and len(rejected_texts) == 0,
            "action": critic_action,
            "confidence": conf_score,
            "approved_claim_ids": list(approved_ids),
            "unsupported_claims": rejected_texts,
            "missing_key_points": warnings,
            "critique_rationale": "; ".join(warnings) if warnings else "Approved by Inner Critic.",
        },
        "processing_status": "SUCCESS",
        "error_stage": None,
        "error_code": None,
        "error_message": None,
        "confidence": confidence_label,
        "warnings": warnings,
    }


def critique_summary_and_claims(
    answer: str,
    claims: list[dict[str, Any]],
    evidence_spans: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """
    Backward-compatible 3-tuple wrapper (fail-closed: returns empty claims on verification failure).
    Returns:
      - refined_answer
      - filtered_claims (empty on failure — never auto-approves)
      - critic_warnings
    """
    res = evaluate_critic_contract(answer, claims, evidence_spans)
    return res["answer"], res["claims"], res["warnings"]
