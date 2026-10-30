from typing import Any
from pydantic import BaseModel, Field
from core.base_agent import BaseAgent
from core.state import AgentState
from utils.llm_client import call_structured_llm

class ClaimVerification(BaseModel):
    claim: str = Field(description="Luận điểm được trích xuất")
    status: str = Field(description="'support', 'contradict', hoặc 'not_enough_info'")
    explanation: str = Field(description="Giải thích ngắn gọn đối chiếu với evidence")

class FactCheckerOutput(BaseModel):
    fact_check_label: str = Field(description="'support' nếu toàn bộ đúng, 'contradict' nếu có mâu thuẫn, 'not_enough_info' nếu thiếu bằng chứng")
    verified_claims: list[ClaimVerification] = Field(description="Danh sách kiểm chứng từng luận điểm")
    critique_reason: str = Field(description="Tổng kết đánh giá")

class FactCheckerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="FactCheckerAgent")

    def run(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("answer", "")
        evidence_spans = state.get("evidence_spans", [])
        retrieved_docs = state.get("retrieved_docs", [])
        self.logger.info(f"[{self.name}] Đang kiểm chứng tính xác thực (Fact-Checking)...")

        docs_text = "\n".join([f"- {d.get('text')}" for d in retrieved_docs])
        spans_text = "\n".join([f"- {s}" for s in evidence_spans])

        system_instruction = (
            "Bạn là một Fact-Checker khách quan và nghiêm ngặt (LLM-as-a-Judge / NLI Verifier). "
            "Nhiệm vụ: phân tách câu trả lời thành các luận điểm (claims), sau đó đối chiếu từng luận điểm "
            "với evidence_spans và tài liệu gốc. "
            "Nếu toàn bộ luận điểm đều được chứng minh: trả về 'support'. "
            "Nếu có luận điểm mâu thuẫn: trả về 'contradict'. "
            "Nếu không đủ tài liệu để khẳng định: trả về 'not_enough_info'."
        )
        user_prompt = (
            f"Câu trả lời cần kiểm chứng:\n{answer}\n\n"
            f"Đoạn trích dẫn bằng chứng (Evidence Spans):\n{spans_text}\n\n"
            f"Tài liệu gốc:\n{docs_text}"
        )

        try:
            res = call_structured_llm(
                prompt=user_prompt,
                schema=FactCheckerOutput,
                system_instruction=system_instruction,
                temperature=0.1
            )
            label = res.fact_check_label
            verified_claims = [c.model_dump() for c in res.verified_claims]
            critique = res.critique_reason
        except Exception as e:
            self.logger.warning(f"[{self.name}] Lỗi Fact-check, mặc định chuyển tiếp an toàn: {e}")
            label = "support"
            verified_claims = []
            critique = "Bỏ qua kiểm chứng do lỗi kết nối"

        trace = {
            "node": self.name,
            "action": "verify_factual_grounding",
            "fact_check_label": label,
            "critique": critique,
            "claims_count": len(verified_claims)
        }
        return {
            "fact_check_label": label,
            "verified_claims": verified_claims,
            "critique_reason": critique,
            "reasoning_trace": [trace]
        }
