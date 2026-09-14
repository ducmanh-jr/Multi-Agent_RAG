from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState


class WriterAgent(BaseAgent):
    """
    Agent 5: Writer Agent (Báo cáo & Phê bình cấp câu)
    Xem README.md cùng thư mục để biết chi tiết kiến trúc, I/O contract và hướng dẫn hiện thực.
    """

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="WriterAgent", max_internal_retry=max_internal_retry)
        # TODO: Thành viên tự khởi tạo các thành phần cần thiết (ReportBuilder, SentenceCritic, ...)
        #       Tham khảo README.md để biết chi tiết.

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Input từ state:
            - query (str): Câu hỏi gốc
            - answer (str): Câu trả lời đã tổng hợp
            - verified_claims (list[dict]): Kết quả kiểm chứng từ Fact-Checker
            - retrieved_docs (list[dict]): Tài liệu gốc
            - document_schema (dict | None): Schema từ Reader Agent

        Output trả về (dict):
            - final_report (str): Báo cáo Markdown hoàn chỉnh
            - reasoning_trace (list[dict]): Log nội bộ
        """
        # TODO: Thành viên hiện thực logic Report Building + Sentence-level Critique tại đây.
        #       Xem README.md để biết chi tiết về cấu trúc báo cáo, orphan detection.
        raise NotImplementedError("Thành viên cần hiện thực hàm này theo README.md")
