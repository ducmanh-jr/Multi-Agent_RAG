from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState


class RetrievalAgent(BaseAgent):
    """
    Agent 1: Retrieval Agent (Router & Hybrid Search)
    Xem README.md cùng thư mục để biết chi tiết kiến trúc, I/O contract và hướng dẫn hiện thực.
    """

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="RetrievalAgent", max_internal_retry=max_internal_retry)
        # TODO: Thành viên tự khởi tạo các thành phần cần thiết (BM25Index, DenseIndex, HybridSearcher, ...)
        #       Tham khảo README.md để biết chi tiết.

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Input từ state:
            - query (str): Câu hỏi nghiên cứu
            - retry_count (int): Số lần Orchestrator đã retry

        Output trả về (dict):
            - retrieved_docs (list[dict]): Danh sách tài liệu đã truy xuất
            - sub_queries (list[str]): Danh sách sub-query (nếu dùng decompose)
            - reasoning_trace (list[dict]): Log nội bộ
        """
        # TODO: Thành viên hiện thực logic Dynamic Routing + Hybrid Search tại đây.
        #       Xem README.md để biết chi tiết về 5 chiến lược (bm25/dense/hybrid/hyde/decompose).
        raise NotImplementedError("Thành viên cần hiện thực hàm này theo README.md")
