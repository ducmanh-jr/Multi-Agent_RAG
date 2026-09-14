from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState


class FactCheckerAgent(BaseAgent):
    """
    Agent 4: Fact-Checker Agent (Trọng tài kiểm chứng sự thật)
    Xem README.md cùng thư mục để biết chi tiết kiến trúc, I/O contract và hướng dẫn hiện thực.
    """

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="FactCheckerAgent", max_internal_retry=max_internal_retry)
        # TODO: Thành viên tự khởi tạo các thành phần cần thiết (ClaimExtractor, FactCheckJudge, ...)
        #       Tham khảo README.md để biết chi tiết.

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Input từ state:
            - answer (str): Câu trả lời từ Summarizer
            - evidence_spans (list[str]): Chứng cứ trích dẫn
            - retrieved_docs (list[dict]): Tài liệu gốc
            - retry_count (int): Số lần retry tầng Orchestrator
            - query (str): Câu hỏi gốc

        Output trả về (dict):
            - claims (list[str]): Danh sách luận điểm đã trích xuất
            - verified_claims (list[dict]): Kết quả kiểm chứng từng claim
            - fact_check_label (str): "support" | "contradict" | "not_enough_info"
            - reasoning_trace (list[dict]): Log nội bộ
        """
        # TODO: Thành viên hiện thực logic Claim Extraction + NLI Judging tại đây.
        #       Xem README.md để biết chi tiết về SciBERT, LLM-as-judge, conditional edge.
        raise NotImplementedError("Thành viên cần hiện thực hàm này theo README.md")
