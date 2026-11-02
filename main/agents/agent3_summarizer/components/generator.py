"""
Summarizer Generator Component.
Synthesizes a draft scientific answer and extracts atomic propositions with evidence spans.
"""

from __future__ import annotations

from typing import Any
from core.state import AtomicClaim, EvidenceSpan, compute_context_hash
from utils.llm_client import llm_client
from utils.logger import logger


def estimate_tokens(text: str) -> int:
    """Rough estimation of token count for mixed Vietnamese/English text (approx ~1.3 tokens per whitespace word)."""
    return int(len(text.split()) * 1.3) + 1


def pack_context_by_token_budget(
    source_units: list[dict[str, Any]],
    max_tokens: int = 3800,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Packs source units into context string until max_tokens budget is reached.
    Ensures long papers (>10 pages) are maximally represented without truncation bias
    or overflowing LLM context windows.
    Returns:
      (packed_context_str, included_units)
    """
    context_parts = []
    included_units = []
    current_tokens = 0

    for u in source_units:
        text = u.get("text", "")
        unit_header = f"[Unit ID: {u.get('id')} | Doc: {u.get('doc_id')} | Trang: {u.get('page_no')}]:\n"
        unit_str = f"{unit_header}{text}\n\n"
        unit_tokens = estimate_tokens(unit_str)

        if current_tokens + unit_tokens > max_tokens:
            if not included_units:
                allowed_words = int((max_tokens - estimate_tokens(unit_header)) / 1.3)
                if allowed_words > 50:
                    truncated_text = " ".join(text.split()[:allowed_words]) + "..."
                    unit_str = f"{unit_header}{truncated_text}\n\n"
                    context_parts.append(unit_str)
                    included_units.append(u)
            break

        context_parts.append(unit_str)
        included_units.append(u)
        current_tokens += unit_tokens

    return "".join(context_parts), included_units


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
    # Prioritize most relevant units via BM25 before packing token budget
    if len(source_units) > 10:
        from agents.agent1_retrieval.tools.bm25 import SimpleBM25
        bm25 = SimpleBM25()
        bm25.fit(source_units)
        relevant_units = bm25.search(query, top_k=15)
        seen_ids = {u.get("id") for u in relevant_units}
        ordered_units = relevant_units + [u for u in source_units if u.get("id") not in seen_ids]
    else:
        ordered_units = source_units

    context_str, _ = pack_context_by_token_budget(ordered_units, max_tokens=3800)

    prompt = f"""Bạn là Trợ lý Nghiên cứu AI thông minh, chuyên nghiệp và chuẩn xác (Agent 3 - Summarizer).
Nhiệm vụ của bạn:
1. Dựa trên tài liệu nguồn dưới đây, hãy trả lời câu hỏi của người dùng một cách tự nhiên, mạch lạc, đầy đủ và sáng rõ như một Chatbot AI hàng đầu (như ChatGPT / NotebookLM).
2. Quy cách trình bày câu trả lời ("answer"):
   - Mở đầu bằng một câu dẫn nhập tự nhiên, trực tiếp vào vấn đề (Ví dụ: "Dưới đây là tóm tắt nội dung tài liệu về..." hoặc "Dựa trên tài liệu nghiên cứu, ...").
   - Trình bày các ý chính thành các gạch đầu dòng rõ ràng, với tiêu đề in đậm (Ví dụ: "- **Tên sự kiện / Khái niệm (Năm)**: Diễn giải chi tiết, số liệu, ý nghĩa...").
   - Nếu có nhiều nhóm chủ đề hoặc sự kiện quốc tế/trong nước, hãy phân nhóm rõ ràng.
   - Giọng văn khách quan, học thuật nhưng tự nhiên, dễ đọc.
3. Trích xuất chính xác các câu trích dẫn nguyên văn (evidence_spans) làm bằng chứng kiểm chứng.
4. Tách các luận điểm thành nhận định nguyên tử (atomic claims) gắn với Unit ID tương ứng.

Câu hỏi nghiên cứu:
"{query}"

Tài liệu nguồn:
{context_str}

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON (không kèm markdown bên ngoài):
{{
  "answer": "Dưới đây là tổng hợp nội dung chi tiết...

- **Tiêu đề 1**: Nội dung...

- **Tiêu đề 2**: Nội dung...",
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
        data = llm_client.generate_json(prompt, agent="summarizer")
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
        logger.warning(f"Generator primary LLM unavailable ({e}), using deterministic extractive synthesis.")
        try:
            import json
            fb_raw = llm_client._mock_fallback(prompt, json_mode=True)
            fb_data = json.loads(fb_raw)
            fb_ans = fb_data.get("answer", "")
            fb_spans = [
                EvidenceSpan(
                    id=f"ev_{i+1}",
                    quote=ev.get("quote", ""),
                    source_unit_id=ev.get("unit_id", "u1"),
                    doc_id=ev.get("doc_id", "doc"),
                    page_no=ev.get("page_no", 1),
                ).to_dict()
                for i, ev in enumerate(fb_data.get("evidence_spans", []))
                if isinstance(ev, dict) and ev.get("quote")
            ]
            fb_claims = [
                AtomicClaim(
                    id=f"claim_{i+1}",
                    text=c.get("text", ""),
                    evidence_ids=[c.get("evidence_unit_id", "u1")],
                ).to_dict()
                for i, c in enumerate(fb_data.get("claims", []))
                if isinstance(c, dict) and c.get("text")
            ]
            if fb_ans:
                return fb_ans, fb_spans, fb_claims
        except Exception:
            pass
        fallback_claim = AtomicClaim(
            id="claim_fallback",
            text="Hệ thống đang hoạt động ở chế độ dự phòng do không có phản hồi từ LLM.",
            evidence_ids=[],
        )
        return "Tài liệu đang được xử lý ở chế độ dự phòng.", [], [fallback_claim.to_dict()]
