from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState


class SummarizerAgent(BaseAgent):
    """
    Agent 3: Summarizer Agent (Generator + Critic)
    Xem README.md cùng thư mục để biết chi tiết kiến trúc, I/O contract và hướng dẫn hiện thực.
    """

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="SummarizerAgent", max_internal_retry=max_internal_retry)
        # TODO: Thành viên tự khởi tạo các thành phần cần thiết (SummarizerGenerator, SummarizerCritic, ...)
        #       Tham khảo README.md để biết chi tiết.

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Input từ state:
            - query (str): Câu hỏi nghiên cứu
            - retrieved_docs (list[dict]): Tài liệu đã truy xuất
            - document_schema (dict | None): Schema từ Reader Agent (nếu có PDF)

        Output trả về (dict):
            - answer (str): Câu trả lời tổng hợp
            - evidence_spans (list[str]): Danh sách đoạn trích dẫn chứng cứ
            - confidence (str): "high" | "low"
            - reasoning_trace (list[dict]): Log nội bộ
        """
        # TODO: Thành viên hiện thực logic Generator + Critic loop tại đây.
        #       Xem README.md để biết chi tiết về Dual LoRA, 3 case types, verbatim matching.
        raise NotImplementedError("Thành viên cần hiện thực hàm này theo README.md")
