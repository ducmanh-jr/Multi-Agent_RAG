from typing import Any
from pydantic import BaseModel, Field
from core.base_agent import BaseAgent
from core.state import AgentState
from utils.llm_client import call_structured_llm

class SummarizerOutput(BaseModel):
    answer: str = Field(description="Câu trả lời súc tích, giải đáp chính xác câu hỏi dựa trên context")
    evidence_spans: list[str] = Field(description="Các câu hoặc đoạn trích NGUYÊN VĂN (verbatim) làm căn cứ")
    confidence: str = Field(description="'high' hoặc 'low'")

class SummarizerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SummarizerAgent")

    def run(self, state: AgentState) -> dict[str, Any]:
        query = state.get("query", "")
        retrieved_docs = state.get("retrieved_docs", [])
        doc_schema = state.get("document_schema", {})
        
        self.logger.info(f"[{self.name}] Đang tổng hợp câu trả lời cho: '{query}'")

        # Chuẩn bị context tổng hợp
        context_parts = []
        for i, d in enumerate(retrieved_docs):
            context_parts.append(f"[Tài liệu {i+1} - ID: {d.get('doc_id')}]:\n{d.get('text')}")
        
        # Nếu có từ PDF
        if doc_schema and "pages" in doc_schema:
            for p in doc_schema["pages"]:
                for s in p.get("sections", [])[:5]:
                    context_parts.append(f"[Trang PDF {p.get('page_no')}]: {s.get('text')}")

        full_context = "\n\n".join(context_parts) if context_parts else "Không có tài liệu nào được tìm thấy."

        system_instruction = (
            "Bạn là một chuyên gia nghiên cứu khoa học. Nhiệm vụ của bạn là đọc kỹ ngữ cảnh tài liệu "
            "và trả lời câu hỏi của người dùng. "
            "BẮT BUỘC: Mọi khẳng định phải có căn cứ trong tài liệu. Cung cấp danh sách các đoạn trích dẫn "
            "nguyên văn (evidence_spans) từ tài liệu hỗ trợ cho câu trả lời."
        )
        user_prompt = f"Câu hỏi: {query}\n\nNgữ cảnh tài liệu:\n{full_context}"

        try:
            res = call_structured_llm(
                prompt=user_prompt,
                schema=SummarizerOutput,
                system_instruction=system_instruction,
                temperature=0.2
            )
            answer = res.answer
            evidence_spans = res.evidence_spans
            confidence = res.confidence
        except Exception as e:
            self.logger.error(f"[{self.name}] Gọi LLM thất bại: {e}")
            answer = "Đã xảy ra lỗi trong quá trình tổng hợp thông tin."
            evidence_spans = []
            confidence = "low"

        trace = {
            "node": self.name,
            "action": "summarize_and_extract_evidence",
            "evidence_count": len(evidence_spans),
            "confidence": confidence
        }
        return {
            "answer": answer,
            "evidence_spans": evidence_spans,
            "confidence": confidence,
            "reasoning_trace": [trace]
        }
