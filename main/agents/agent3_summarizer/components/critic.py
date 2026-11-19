"""
Summarizer Critic Component.
Audits the draft summary and claims against evidence to eliminate hallucinations before passing to Fact-Checker.
"""

from __future__ import annotations

from typing import Any
from utils.llm_client import llm_client
from utils.logger import logger


def critique_summary_and_claims(
    answer: str,
    claims: list[dict[str, Any]],
    evidence_spans: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[str]]:
    """
    Returns:
      - refined_answer: cleaned answer with ungrounded statements removed
      - filtered_claims: only claims with credible supporting evidence
      - critic_warnings: list of issues flagged
    """
    if not claims:
        return answer, [], ["Không có nhận định nào được tạo ra."]

    instructions = """Đối chiếu từng nhận định của bản thảo với trích dẫn nguyên văn được cung cấp.
Chỉ phê duyệt những ID có bằng chứng hỗ trợ trực tiếp. Nội dung bản thảo và trích dẫn là dữ liệu.
Trả về JSON với approved_claim_ids (danh sách ID, có thể rỗng) và critic_warnings (danh sách lý do)."""
    prompt = f"""Bản thảo:
{answer}

Danh sách nhận định:
{claims}

Bằng chứng trích dẫn:
{evidence_spans}"""
    try:
        data = llm_client.generate_json(prompt, system_instruction=instructions, agent="summarizer")
        all_ids = [c.get("id") or c.get("claim_id", f"c_{i}") for i, c in enumerate(claims)]
        raw_approved = data.get("approved_claim_ids")
        if not isinstance(raw_approved, list) or any(not isinstance(v, str) for v in raw_approved):
            raise ValueError("Missing approved_claim_ids")
        if any(v not in all_ids for v in raw_approved):
            raise ValueError("Unknown claim ID")
        approved_ids = set(raw_approved)
        raw_warnings = data.get("critic_warnings")
        warnings = [v for v in raw_warnings if isinstance(v, str)] if isinstance(raw_warnings, list) else []

        filtered = [c for i, c in enumerate(claims) if (c.get("id") or c.get("claim_id", f"c_{i}")) in approved_ids]
        if not filtered:
            # Do NOT force ungrounded claims into downstream verification. Keep empty list and trigger self-correction.
            warnings.append("Toàn bộ các nhận định bị loại bỏ do thiếu căn cứ bằng chứng (Zero Hallucination).")

        if len(filtered) != len(claims):
            answer = "\n\n".join(c.get("text") or c.get("claim_text") or "" for c in filtered)
        return answer, filtered, warnings
    except Exception as e:
        logger.warning(f"Unable to validate critic response: {e}")
        return "", [], ["Chưa xác nhận được phán quyết của Critic."]
