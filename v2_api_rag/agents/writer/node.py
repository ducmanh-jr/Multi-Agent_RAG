from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState
from utils.llm_client import call_text_llm

class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="WriterAgent")

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        answer = state.get("answer", "")
        evidence_spans = state.get("evidence_spans", [])
        verified_claims = state.get("verified_claims", [])
        self.logger.info(f"[{self.name}] Đang tổng hợp báo cáo nghiên cứu hoàn chỉnh...")

        spans_text = "\n".join([f"> \"{s}\"" for s in evidence_spans])
        claims_text = "\n".join([f"- {c.get('claim')} ({c.get('status')}): {c.get('explanation')}" for c in verified_claims])

        system_instruction = (
            "Bạn là một biên tập viên nghiên cứu học thuật cao cấp. Nhiệm vụ của bạn là soạn thảo "
            "một Báo Cáo Nghiên Cứu toàn diện, trang trọng theo chuẩn Markdown dựa trên câu trả lời "
            "và các bằng chứng đã được kiểm duyệt.\n"
            "Cấu trúc báo cáo bắt buộc gồm:\n"
            "# BÁO CÁO NGHIÊN CỨU: [Tiêu đề phù hợp]\n"
            "## 1. Tóm tắt kết quả (Executive Summary)\n"
            "## 2. Phân tích chi tiết & Luận điểm khoa học\n"
            "## 3. Căn cứ bằng chứng & Trích dẫn nguyên văn (Citations)\n"
            "## 4. Kết luận & Khuyến nghị"
        )
        user_prompt = (
            f"Câu hỏi nghiên cứu: {query}\n\n"
            f"Kết quả sơ bộ: {answer}\n\n"
            f"Bằng chứng hỗ trợ:\n{spans_text}\n\n"
            f"Đối chiếu luận điểm:\n{claims_text}"
        )

        try:
            report = call_text_llm(
                prompt=user_prompt,
                system_instruction=system_instruction,
                temperature=0.3
            )
        except Exception as e:
            self.logger.error(f"[{self.name}] Soạn thảo thất bại: {e}")
            report = f"# BÁO CÁO NGHIÊN CỨU\n\n### Kết quả:\n{answer}"

        trace = {
            "node": self.name,
            "action": "synthesize_final_report",
            "report_length": len(report)
        }
        return {
            "final_report": report,
            "confidence": "high",
            "reasoning_trace": [trace]
        }
