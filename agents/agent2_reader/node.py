from typing import Any
from core.base_agent import BaseAgent
from core.state import AgentState


class ReaderAgent(BaseAgent):
    """
    Agent 2: Reader Agent (Document Layout Reader)
    Xem README.md cùng thư mục để biết chi tiết kiến trúc, I/O contract và hướng dẫn hiện thực.
    """

    def __init__(self, max_internal_retry: int = 2):
        super().__init__(name="ReaderAgent", max_internal_retry=max_internal_retry)
        # TODO: Thành viên tự khởi tạo các thành phần cần thiết (DocumentLayoutParser, ...)
        #       Tham khảo README.md để biết chi tiết.

    def run(self, state: AgentState) -> dict[str, Any]:
        """
        Input từ state:
            - pdf_path (str): Đường dẫn tới file PDF
            - retrieved_docs (list[dict]): Tài liệu từ Retrieval Agent

        Output trả về (dict):
            - document_schema (dict): Schema DocLayNet đã phân tích
            - reasoning_trace (list[dict]): Log nội bộ
        """
        # TODO: Thành viên hiện thực logic Document Layout Parsing tại đây.
        #       Xem README.md để biết chi tiết về DocLayNet schema, PyMuPDF, OCR fallback.
        raise NotImplementedError("Thành viên cần hiện thực hàm này theo README.md")
