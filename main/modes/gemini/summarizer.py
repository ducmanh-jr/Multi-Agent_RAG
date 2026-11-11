"""
Gemini Mode - Agentic Summarizer + Self-Critic Agent (L3-L4, Quota-Optimized).
Combines Grounded Synthesis, Verbatim Evidence Span Extraction, Atomic Claim Decomposition,
and Self-Critique (approve / rewrite / request_more_context) in a SINGLE structured JSON Gemini Call!
"""

from __future__ import annotations

import json
import re
from typing import Any

from agents.agent3_summarizer.components.generator import pack_context_by_token_budget
from core.base_agent import BaseAgent
from core.schemas import SummarizerCriticOutput
from core.state import AgentState, AtomicClaim, EvidenceSpan
from core.verifier import ensure_inline_citations, evaluate_inner_critic_fast
from utils.llm_client import llm_client
from utils.domain_guard import check_out_of_domain, build_out_of_domain_response


class GeminiSummarizerAgent(BaseAgent):
    """Agentic Summarizer + Self-Critic powered by Gemini API (1 Call)."""

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="GeminiSummarizerAgent", max_internal_retry=max_internal_retry)

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query") or state.get("question", "")
        source_units = state.get("source_units", []) or []
        required_entities = state.get("required_entities", []) or []

        clean_q = query.strip()
        is_ood, ood_reason = check_out_of_domain(clean_q, source_units)
        if is_ood:
            self.logger.info(f"[Gemini Summarizer] Zero-Hallucination Guard triggered: {ood_reason}")
            res = build_out_of_domain_response(clean_q, model_family="gemini_flash")
            trace = self.log_action("gemini_summarization_completed", {"status": "out_of_domain", "reason": ood_reason})
            res["reasoning_trace"] = [trace]
            return res

        self.logger.info(f"[Gemini Summarizer] Calling Gemini Flash for: '{query[:65]}' ({len(source_units)} units)")

        context_str, _ = pack_context_by_token_budget(source_units, max_tokens=6000)

        chat_context = ""
        chat_history = state.get("chat_history") or []
        if chat_history:
            turns = []
            for t in chat_history[-3:]:
                r = "Người dùng" if t.get("role") == "user" else "Trợ lý AI"
                c = (t.get("content") or "").strip()[:200]
                if c:
                    turns.append(f"{r}: {c}")
            if turns:
                chat_context = "\n[Ngữ cảnh trao đổi đa lượt trước đó]:\n" + "\n".join(turns) + "\n"

        is_summary_request = any(
            kw in clean_q.lower()
            for kw in [
                "/tom-tat",
                "tom-tat",
                "/tomtat",
                "tóm tắt",
                "tổng quan nội dung",
                "đóng góp cốt lõi",
                "kết quả chính",
            ]
        )

        summary_instruction = ""
        if is_summary_request:
            summary_instruction = """
ĐẶC BIỆT KHI CÂU HỎI LÀ YÊU CẦU TÓM TẮT (/tom-tat, tóm tắt tổng quan, đóng góp cốt lõi):
- BẮT BUỘC trả lời ĐẦY ĐỦ, CHI TIẾT và TOÀN DIỆN tất cả các phần cốt lõi của tài liệu (tối thiểu 4 - 6 mục, độ dài khoảng 400 - 800 từ).
- Tuyệt đối KHÔNG trả lời tóm tắt bằng 1 câu ngắn cụt ngủn!
- Trình bày thành các mục lớn rõ ràng với tiêu đề in đậm. BẮT BUỘC XUỐNG DÒNG (dùng ký tự \\n\\n) giữa các mục và các gạch đầu dòng, TUYỆT ĐỐI KHÔNG VIẾT DỒN TRÊN CÙNG MỘT DÒNG:
  1. **Mục tiêu & Phạm vi bài học / nghiên cứu**: Giới thiệu mục đích chính và bài toán tài liệu giải quyết.
  2. **Kiến trúc cốt lõi & Khái niệm nền tảng**: Phân tích các thành phần kiến trúc, giao thức, định nghĩa trọng tâm (Web Server, Reverse Proxy, Port, Domain...).
  3. **Cấu hình & Giải pháp Kỹ thuật Thực tế**: Trình bày chi tiết các cơ chế, tham số cấu hình, ví dụ hoặc phương pháp triển khai (định tuyến, proxy_set_header, client_max_body_size...).
  4. **Cân bằng tải / Bảo mật / Tối ưu hóa**: Đi sâu vào các giải pháp nâng cao (upstream round-robin/least_conn, SSL/TLS, Certbot...).
  5. **Kế hoạch triển khai & Đánh giá kết quả**: Bảng phân công, khối lượng công việc (points) hoặc kết quả kiểm nghiệm thực tế.
  6. **Lưu ý & Khuyến nghị Quan trọng**: Các cảnh báo vận hành (kiểm tra cú pháp với nginx -t, log debug, rủi ro).
"""

        prompt = f"""Bạn là Trợ lý AI Nghiên cứu Cấp cao & Thẩm định Bằng chứng (Gemini Intelligence Engine).{chat_context}
Nhiệm vụ: Dựa hoàn toàn vào Tài liệu nguồn dưới đây, hãy trả lời câu hỏi sau một cách chuẩn xác, trực diện, mạch lạc, đầy đủ và tự nhiên như một Chatbot AI hàng đầu (như ChatGPT / NotebookLM):
"{query}"

QUY TẮC TRẢ LỜI CỐT LÕI (BẮT BUỘC - NGUYÊN TẮC 'ĐÚNG VÀ ĐỦ'):
1. Chuẩn mực Đúng và Đủ: Trả lời ĐÚNG TRỌNG TÂM, ĐẦY ĐỦ bản chất kỹ thuật, KHÔNG THỪA THÃI và KHÔNG THIẾU SÓT. Tuyệt đối KHÔNG trả lời cụt ngủn 1 dòng, nhưng cũng KHÔNG chèn lan man rác hành chính (Task, slide, người làm), danh mục tham khảo đọc thêm (Bài viết, video), hay các cấu hình không liên quan.
2. Khái niệm / Kỹ thuật / Mã lệnh: Nếu câu hỏi hỏi về định nghĩa hoặc cấu hình kỹ thuật (ví dụ "upstream là gì"):
   - Mở đầu bằng đoạn giải thích trực diện bản chất và mục đích của nó.
   - Cung cấp cú pháp cấu hình mẫu chuẩn xác đặt trong khối mã lệnh (ví dụ ```nginx ... ```).
   - Nêu 1-2 cơ chế vận hành cốt lõi gắn liền trực tiếp với đối tượng được hỏi (ví dụ: thuật toán cân bằng tải, cách dùng proxy_pass).
3. Trình bày danh sách & in đậm thực thể (Structured Formatting):
   - Nếu câu hỏi hỏi về các nhóm vấn đề, nội dung, mục tiêu, hoặc danh sách: Hãy dùng các gạch đầu dòng rõ ràng hoặc số thứ tự, in đậm tiêu đề từng ý (ví dụ: "1. **Tên ý chính**: Diễn giải cụ thể..."). Mỗi mục/gạch đầu dòng BẮT BUỘC nằm trên một dòng riêng biệt (dùng ký tự \\n hoặc \\n\\n), tuyệt đối không viết gộp trên cùng một dòng.
   - Nếu câu hỏi về mốc thời gian, ngày tháng, tên riêng: Nêu chính xác con số, ngày tháng năm và sự kiện đi kèm.
4. Gắn trích dẫn học thuật: Gắn ký hiệu [1], [2] tương ứng ngay sau mỗi luận điểm có bằng chứng.
{summary_instruction}
5. KỊCH BẢN KHI TÀI LIỆU KHÔNG CÓ THÔNG TIN (ZERO HALLUCINATION):
   - Nếu câu hỏi hỏi về một sự việc, nhân vật, mốc thời gian hoặc chủ đề mà Tài liệu nguồn KHÔNG hề đề cập đến:
     + Tuyệt đối KHÔNG tự suy đoán, bịa đặt hoặc gán ghép là tài liệu có nói.
     + Trả lời rõ ràng, minh bạch ngay dòng đầu tiên theo mẫu:
       "Dựa trên các bài báo/tài liệu hiện có trong hệ thống, thông tin về \"{query}\" **hoàn toàn không được đề cập**."
     + Thêm gợi ý: "**Gợi ý**: Bạn có thể tải thêm file PDF hoặc bài báo liên quan đến vấn đề này lên hệ thống để tôi đọc và phân tích cho bạn."
     + Đặt "is_supported": true, "critic_action": "approve", "evidence_spans": [], "claims": [{{"text": "Tài liệu không đề cập đến thông tin này", "unit_id": None}}].
5. KỊCH BẢN SO SÁNH / TỔNG HỢP NHIỀU TÀI LIỆU (MULTI-DOCUMENT COMPARISON):
   - Khi ngữ cảnh có nhiều tài liệu và yêu cầu là so sánh, đối chiếu (/so-sanh) hoặc tóm tắt (/tom-tat):
     + Nếu các tài liệu thuộc các chủ đề độc lập (ví dụ NGINX vs Môi trường): KHÔNG từ chối. Mở đầu bằng nhận định: "> **Đánh giá tương quan:** Các tài liệu được chọn thuộc **các chủ đề/lĩnh vực nghiên cứu độc lập**. Vì nội dung không cùng một hệ quy chiếu chuyên môn, hệ thống không áp đặt so sánh giả tạo, mà đối chiếu khách quan phạm vi và nội dung cốt lõi của từng tài liệu:".
     + Tiếp theo, trình bày mạch lạc từng tài liệu: Tiêu đề, Nội dung cốt lõi, Chi tiết thực nghiệm/pháp lý, kèm số trang và trích dẫn [1], [2].

Tài liệu nguồn:
{context_str}

YÊU CẦU TRẢ VỀ ĐÚNG ĐỊNH DẠNG JSON HỢP LỆ (không kèm markdown ngoài):
{{
  "answer": "Câu trả lời trực tiếp đầy đủ, mạch lạc, chi tiết có in đậm các luận điểm, xuống dòng rõ ràng (\\n\\n) giữa các mục và gắn trích dẫn [1], [2]...",
  "evidence_spans": [
    {{"quote": "câu trích dẫn nguyên văn từ tài liệu nguồn", "unit_id": "Unit ID", "doc_id": "Doc ID", "page_no": 1}}
  ],
  "claims": [
    {{"text": "Nhận định nguyên tử độc lập đã được kiểm chứng", "unit_id": "Unit ID"}}
  ],
  "is_supported": true,
  "missing_key_points": [],
  "critic_action": "approve"
}}
"""
        response_text = ""
        try:
            response_text = llm_client._call_gemini(
                prompt=prompt,
                temperature=0.15,
                max_tokens=4096,
                json_mode=True,
            )
        except Exception as e:
            self.logger.warning(f"[Gemini Summarizer] API call failed ({e}), using deterministic fallback.")

        answer = ""
        raw_claims: list[Any] = []
        raw_spans: list[Any] = []
        critic_out = SummarizerCriticOutput()

        if response_text:
            clean_json = re.sub(r"^```(?:json)?|```$", "", response_text.strip(), flags=re.MULTILINE).strip()
            try:
                # Fix unescaped LaTeX backslashes before json.loads
                sanitized_json = re.sub(r'\\(?![/"\\bfnrtu])', r"\\\\", clean_json)
                data = json.loads(sanitized_json)
                answer = str(data.get("answer", "")).strip()
                raw_spans = data.get("evidence_spans", []) or []
                raw_claims = data.get("claims", []) or []
                critic_out = SummarizerCriticOutput.from_dict(
                    {
                        "is_supported": data.get("is_supported", True),
                        "missing_key_points": data.get("missing_key_points", []),
                        "action": data.get("critic_action", "approve"),
                    }
                )
            except Exception:
                # Robust fallback for unescaped inline quotes or truncated JSON
                m_ans = re.search(r'"answer"\s*:\s*"(.*?)(?:"\s*,\s*"(?:evidence_spans|claims|is_supported|missing_key_points|critic_action)"|\s*"\}|\s*$)', clean_json, flags=re.DOTALL)
                if m_ans:
                    answer = m_ans.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
                else:
                    m_ans_alt = re.search(r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)', clean_json, flags=re.DOTALL)
                    if m_ans_alt:
                        answer = m_ans_alt.group(1).replace('\\"', '"').replace("\\n", "\n").strip()
                    else:
                        answer = re.sub(r'^\s*\{\s*"answer"\s*:\s*"?', "", clean_json).strip()

                # Attempt to rescue evidence_spans from clean_json
                m_spans = re.search(r'"evidence_spans"\s*:\s*(\[.*?\])', clean_json, flags=re.DOTALL)
                if m_spans:
                    try:
                        raw_spans = json.loads(m_spans.group(1))
                    except Exception:
                        pass

                # Attempt to rescue claims from clean_json
                m_claims = re.search(r'"claims"\s*:\s*(\[.*?\])', clean_json, flags=re.DOTALL)
                if m_claims:
                    try:
                        raw_claims = json.loads(m_claims.group(1))
                    except Exception:
                        pass

        # Clean trailing unclosed brackets or punctuation cut-offs
        if answer:
            answer = re.sub(r"[,;\s]*\[\s*$", "", answer).strip()

        if not answer:
            self.logger.warning("[Gemini Summarizer] Primary Gemini unavailable, engaging deterministic engine.")
            from modes.cpu_min.summarizer import CPUMinSummarizerAgent
            cpu_agent = CPUMinSummarizerAgent()
            cpu_res = cpu_agent.run(state)
            return cpu_res

        # Normalize claims and evidence_spans to strict state contracts
        claims = [
            AtomicClaim(
                id=f"claim_{idx + 1}",
                text=c.get("text") or c.get("claim", "") if isinstance(c, dict) else str(c),
                unit_id=c.get("unit_id") if isinstance(c, dict) else None,
            ).to_dict()
            for idx, c in enumerate(raw_claims)
            if (c.get("text") or c.get("claim", "") if isinstance(c, dict) else str(c)).strip()
        ]
        evidence_spans = [
            EvidenceSpan(
                id=f"ev_{idx + 1}",
                quote=e.get("quote", "") if isinstance(e, dict) else str(e),
                unit_id=e.get("unit_id", "u1") if isinstance(e, dict) else "u1",
                doc_id=e.get("doc_id", "doc") if isinstance(e, dict) else "doc",
                page_no=e.get("page_no", 1) if isinstance(e, dict) else 1,
            ).to_dict()
            for idx, e in enumerate(raw_spans)
        ]

        if not claims and answer and source_units:
            from core.verifier import split_answer_into_atomic_claims
            claims = split_answer_into_atomic_claims(answer, source_units)

        if not evidence_spans and source_units:
            for idx, u in enumerate(source_units[:5]):
                txt = str(u.get("text") or u.get("content") or "").strip()
                if txt:
                    evidence_spans.append(
                        EvidenceSpan(
                            id=f"ev_{idx + 1}",
                            quote=txt[:250],
                            unit_id=u.get("id", f"u{idx+1}"),
                            doc_id=u.get("doc_id", "doc"),
                            page_no=u.get("page_no", 1),
                        ).to_dict()
                    )

        answer = ensure_inline_citations(answer, source_units, evidence_spans)
        answer = re.sub(r"[,;\s]*\[\s*$", "", answer).strip()

        trace = self.log_action(
            "gemini_summarization_completed",
            {
                "engine": "Gemini_API_SingleCall_Critic",
                "claims_count": len(claims),
                "evidence_count": len(evidence_spans),
                "critic_action": critic_out.action,
            },
        )

        return {
            "answer": answer,
            "claims": claims,
            "evidence_spans": evidence_spans,
            "critic_action": critic_out.action,
            "critic_feedback": critic_out.to_dict(),
            "generator_model_family": "gemini_flash",
            "reasoning_trace": [trace],
        }
