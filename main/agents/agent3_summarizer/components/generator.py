"""
Summarizer Generator Component.
Synthesizes a draft scientific answer and extracts atomic propositions with evidence spans.
"""

from __future__ import annotations

import re
from typing import Any
from core.state import AtomicClaim, EvidenceSpan, compute_context_hash
from utils.llm_client import llm_client
from utils.logger import logger


def estimate_tokens(text: str) -> int:
    """Accurate estimation of token count for mixed Vietnamese/English text (accounting for multi-token Vietnamese subwords)."""
    words = text.split()
    return int(len(words) * 2.3) + 1


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


def format_academic_answer(answer: str) -> str:
    """Đảm bảo câu trả lời luôn có ngắt đoạn rõ ràng, không dồn chữ và làm sạch trích dẫn."""
    if not answer:
        return answer
    text = re.sub(r"(\[\d+\](?:\s*\*(?:\(Trang\s*\d+\))\*)?)\s*\.\s*", r"\1. ", answer)
    text = re.sub(r"\s+([,\.])", r"\1", text)
    # Ensure line breaks before markdown headers
    text = re.sub(r"([^\n])\s*(#{1,4}\s+[^\n]+)", r"\1\n\n\2\n", text)
    if "\n\n" not in text and len(text) > 200:
        text = re.sub(r"(\[\d+\](?:\s*\*(?:\(Trang\s*\d+\))\*)?\s*\.)\s+([A-ZÀ-ỸĐ])", r"\1\n\n\2", text)
    text = re.sub(r"\n{3,}", r"\n\n", text)
    return text.strip()


def generate_summary_and_claims(
    query: str,
    source_units: list[dict[str, Any]],
    force_extractive: bool = False,
) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Returns:
      - draft_answer: comprehensive academic synthesis
      - evidence_spans: verbatim quotations with unit_id & context_hash
      - claims: atomic propositions
    """
    if not source_units:
        clean_q = query.strip()
        no_doc_msg = (
            f"Dựa trên các bài báo và tài liệu hiện có trong hệ thống, thông tin về \"{clean_q}\" "
            f"**hoàn toàn không được đề cập**.\n\n"
            f"**Gợi ý**: Bạn có thể tải file PDF hoặc bài báo liên quan đến vấn đề này lên hệ thống "
            f"để tôi có thể đọc và phân tích chi tiết cho bạn."
        )
        return no_doc_msg, [], [{"id": "claim_1", "text": f"Tài liệu hiện có không đề cập đến '{clean_q}'.", "evidence_ids": []}]

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

    is_summary_req = any(kw in query.lower() for kw in ["/tom-tat", "tom-tat", "tóm tắt", "tổng quan", "đóng góp cốt lõi"])
    summary_guidance = ""
    if is_summary_req:
        summary_guidance = """
   - ĐẶC BIỆT KHI CÂU HỎI LÀ TÓM TẮT (/tom-tat hoặc tóm tắt tài liệu): Trả lời ĐẦY ĐỦ, TOÀN DIỆN và TRÌNH BÀY ĐẸP THEO MỤC RÕ RÀNG (tuyệt đối KHÔNG viết dồn thành một đoạn văn dài):
     ### 1. Mục tiêu & Bối cảnh Nghiên cứu
     - Trình bày bối cảnh bài toán, mục đích chính kèm trích dẫn [1] *(Trang X)*.
     ### 2. Các Phương pháp Tiếp cận & Kiến trúc Đánh giá
     - Liệt kê rõ các phương pháp, thuật toán, mô hình (in đậm tên mô hình: **BERT-large**, **RoBERTa-base**, **Llama 3.1 8B**, v.v.).
     ### 3. Kết quả Thực nghiệm & Số liệu Cốt lõi
     - Trình bày chi tiết các chỉ số định lượng then chốt (in đậm độ đo & số liệu: **0,9882**, **F1 macro 0,9881**, **8.453 câu**, v.v.).
     ### 4. Kết luận & Đánh đổi Triển khai (Trade-offs)
     - Tổng kết các ưu/nhược điểm và khuyến nghị thực tế."""

    prompt = f"""Bạn là Trợ lý Nghiên cứu AI thông minh, chuyên nghiệp và chuẩn xác (Agent 3 - Summarizer).
Nhiệm vụ của bạn:
1. Dựa trên tài liệu nguồn dưới đây, hãy tuân thủ nguyên tắc cốt lõi **"ĐÚNG VÀ ĐỦ"**:
   - Trả lời ĐÚNG TRỌNG TÂM, ĐẦY ĐỦ bản chất kỹ thuật, KHÔNG THỪA THÃI và KHÔNG THIẾU.
   - Tuyệt đối KHÔNG dùng câu trả lời nhanh cụt ngủn 1-2 dòng, nhưng cũng KHÔNG chèn lan man các thông tin ngoài lề không liên quan trực tiếp đến câu hỏi.
2. Quy cách trình bày câu trả lời ("answer") - BẮT BUỘC TUÂN THỦ:
   - BẮT BUỘC NGẮT ĐOẠN RÕ RÀNG (\n\n) giữa các ý. Tuyệt đối KHÔNG viết một khối văn bản dài dồn cục (wall of text).
   - BẮT BUỘC IN ĐẬM (**từ khóa**) cho các thực thể quan trọng: tên phương pháp/mô hình (ví dụ: **BERT-large**, **Llama 3.1 8B**, **Nginx**), các độ đo/số liệu định lượng (ví dụ: **0,9882**, **F1 macro 0,9881**, **8.453 câu**), và các phát hiện then chốt.
   - Trích dẫn nguồn chính xác theo định dạng [1] *(Trang X)*. ngay sau luận điểm tương ứng.
   - Không dùng nhãn "> **Câu trả lời trực tiếp:**".
   - Nếu câu hỏi về định nghĩa/khái niệm/cú pháp cụ thể (ví dụ: "upstream là gì"): Trình bày rõ bản chất, khối cấu hình minh họa (dùng khối code ```nginx ... ``` nếu có cấu hình), và 1-2 cơ chế kỹ thuật gắn liền trực tiếp với đối tượng được hỏi. Tuyệt đối KHÔNG kéo các mục giáo trình, bài tập, slide, tài liệu tham khảo hay các thông số cấu hình không liên quan.{summary_guidance}
   - Giọng văn khách quan, học thuật nhưng tự nhiên, mạch lạc, dễ đọc.
3. QUY TẮC KHI TÀI LIỆU KHÔNG CÓ THÔNG TIN (ZERO HALLUCINATION - KỊCH BẢN 2):
   - Nếu câu hỏi hỏi về một sự kiện, người hoặc chủ đề mà tài liệu nguồn KHÔNG hề đề cập:
     + Tuyệt đối KHÔNG tự sáng tác hay suy diễn.
     + Trả lời rõ ràng: "Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{query}\" **hoàn toàn không được đề cập**.\n\nVui lòng hỏi các câu hỏi liên quan đến nội dung tài liệu được cung cấp hoặc tải thêm tài liệu liên quan lên hệ thống để phân tích."
4. QUY TẮC KHI CÂU HỎI MƠ HỒ / THIẾU BỐI CẢNH / PHÂN NHÁNH ĐA NGHĨA (KHỬ MƠ HỒ CHỦ ĐỘNG - KỊCH BẢN 3):
   - Khi câu hỏi hỏi chung chung, quá ngắn, hoặc thiếu ngữ cảnh cụ thể trong khi tài liệu nguồn có nhiều khía cạnh/kết quả/phiên bản khác nhau (ví dụ: "kết quả thế nào", "cấu hình ra sao", "ưu điểm là gì"):
     + Bước 1: Khẳng định tài liệu CÓ đề cập đến chủ đề này nhưng tồn tại nhiều khía cạnh hoặc kịch bản khác nhau.
     + Bước 2: Liệt kê ngắn gọn 2–4 nhánh thông tin chính đang có trong tài liệu kèm trích dẫn [1] *(Trang X)*.
     + Bước 3: Đưa ra chỉ dẫn: "Vui lòng cung cấp thêm chi tiết để tôi hỗ trợ chính xác hơn dựa trên tài liệu."
5. Trích xuất chính xác các câu trích dẫn nguyên văn (evidence_spans) làm bằng chứng kiểm chứng.
6. Tách các luận điểm thành nhận định nguyên tử (atomic claims) gắn với Unit ID tương ứng.

Câu hỏi nghiên cứu:
"{query}"

Tài liệu nguồn:
{context_str}

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON (không kèm markdown bên ngoài):
{{
  "answer": "Dựa trên tài liệu nghiên cứu, ... (câu trả lời chuẩn xác, trực diện bản chất, đủ mã cấu hình/cơ chế kỹ thuật nếu có, trích dẫn [1] rõ ràng, tuyệt đối không chèn bài viết/video/task ngoài lề)",
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
        if force_extractive:
            raise RuntimeError("Fast fallback requested: skipping redundant remote LLM call.")
        data = llm_client.generate_json(prompt, agent="summarizer")
        answer = re.sub(r"[,;\s]*\[\s*$", "", data.get("answer", "Chưa thể tổng hợp câu trả lời.")).strip()
        answer = re.sub(r"^>\s*\*\*Câu trả lời trực tiếp:\*\*\s*", "", answer, flags=re.IGNORECASE).strip()
        answer = format_academic_answer(answer)
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

        if not claims and source_units and not data.get("is_out_of_domain"):
            from modes.cpu_min.summarizer import CPUMinSummarizerAgent
            cpu_res = CPUMinSummarizerAgent().run({"query": query, "source_units": source_units})
            if cpu_res.get("claims"):
                return cpu_res["answer"], cpu_res.get("evidence_spans", []), cpu_res["claims"]

        return answer, evidence_spans, claims

    except Exception as e:
        logger.warning(f"Generator primary LLM unavailable ({e}), using deterministic extractive synthesis.")
        try:
            from modes.cpu_min.summarizer import CPUMinSummarizerAgent
            cpu_res = CPUMinSummarizerAgent().run({"query": query, "source_units": source_units})
            if cpu_res.get("claims") and cpu_res.get("answer"):
                return cpu_res["answer"], cpu_res.get("evidence_spans", []), cpu_res.get("claims", [])
        except Exception:
            pass
        try:
            import json
            fb_raw = llm_client._offline_extractive_fallback(prompt, json_mode=True)
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
