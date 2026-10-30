"""
Summarizer Generator Component.
Synthesizes a draft scientific answer and extracts atomic propositions with evidence spans.
"""

from __future__ import annotations

from typing import Any
from core.state import AtomicClaim, EvidenceSpan, compute_context_hash
from utils.llm_client import llm_client
from utils.logger import logger


def generate_summary_and_claims(
    query: str,
    source_units: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
      - draft_answer: comprehensive academic synthesis
      - evidence_spans: verbatim quotations with unit_id & context_hash
      - claims: atomic propositions
    """
    # Prepare top context units
    context_str = ""
    for u in source_units[:6]:
        context_str += f"[Unit ID: {u['id']} | Doc: {u.get('doc_id')} | Trang: {u.get('page_no')}]:\n{u.get('text')}\n\n"

    prompt = f"""Bạn là Tác tử Tóm tắt Khoa học (Agent 3 - Generator) trong hệ thống Evidence-First RAG.
Nhiệm vụ của bạn:
1. Trả lời câu hỏi nghiên cứu dựa TRUYỆT ĐỐI vào các đoạn tài liệu nguồn được cung cấp dưới đây. KHÔNG tự bịa đặt thông tin.
2. Trích xuất chính xác các câu trích dẫn nguyên văn (evidence_spans) làm bằng chứng.
3. Tách câu trả lời thành các nhận định nguyên tử (atomic claims), mỗi nhận định phải gắn với Unit ID tương ứng.

Câu hỏi nghiên cứu:
"{query}"

Tài liệu nguồn:
{context_str}

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON (không kèm markdown):
{{
  "answer": "Nội dung trả lời chi tiết và khoa học...",
  "evidence_spans": [
    {{
      "quote": "trích dẫn nguyên văn từ tài liệu nguồn",
      "unit_id": "Unit ID chính xác chứa trích dẫn",
      "doc_id": "tên doc_id",
      "page_no": 1
    }}
  ],
  "claims": [
    {{
      "id": "claim_1",
      "text": "Nội dung nhận định nguyên tử ngắn gọn, độc lập",
      "evidence_unit_id": "Unit ID hỗ trợ"
    }}
  ]
}}
"""
    try:
        data = llm_client.generate_json(prompt)
        answer = data.get("answer", "Chưa thể tổng hợp câu trả lời.")
        raw_evidence = data.get("evidence_spans", [])
        raw_claims = data.get("claims", [])

        evidence_spans = []
        for i, ev in enumerate(raw_evidence):
            q = ev.get("quote", "").strip()
            if not q:
                continue
            span = EvidenceSpan(
                id=f"ev_{i+1}",
                quote=q,
                source_unit_id=ev.get("unit_id", "unknown"),
                doc_id=ev.get("doc_id", "unknown"),
                page_no=ev.get("page_no", 1),
            )
            evidence_spans.append(span.to_dict())

        claims = []
        for i, c in enumerate(raw_claims):
            if isinstance(c, dict):
                text = c.get("text", "").strip()
                ev_id = c.get("evidence_unit_id", "")
            else:
                text = str(c).strip()
                ev_id = ""
            if not text:
                continue
            claim = AtomicClaim(
                id=c.get("id", f"claim_{i+1}") if isinstance(c, dict) else f"claim_{i+1}",
                text=text,
                evidence_ids=[ev_id] if ev_id else [],
            )
            claims.append(claim.to_dict())

        return answer, evidence_spans, claims

    except Exception as e:
        logger.error(f"Generator failed: {e}")
        fallback_claim = AtomicClaim(
            id="claim_fallback",
            text="Hệ thống đang hoạt động ở chế độ dự phòng do không có phản hồi từ LLM.",
            evidence_ids=[],
        )
        return "Tài liệu đang được xử lý ở chế độ dự phòng.", [], [fallback_claim.to_dict()]
