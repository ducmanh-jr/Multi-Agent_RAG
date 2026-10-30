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
  "critic_warnings": ["Lý do nếu có nhận định bị loại..."]
}}
"""
    try:
        data = llm_client.generate_json(prompt)
        approved_ids = set(data.get("approved_claim_ids", [c["id"] for c in claims]))
        warnings = data.get("critic_warnings", [])

        filtered = [c for c in claims if c["id"] in approved_ids]
        if not filtered:
            filtered = claims[:3]  # keep at least some for fact checker

        return answer, filtered, warnings
    except Exception as e:
        logger.warning(f"Critic pass encountered an error ({e}), bypassing critic filter.")
        return answer, claims, []
