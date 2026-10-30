from typing import TypedDict, Annotated, Optional, Any
import operator

class AgentState(TypedDict, total=False):
    """
    AgentState dùng chung cho toàn bộ LangGraph Pipeline v2.
    Được cập nhật tuần tự qua các node theo cơ chế Reducer.
    """
    # 1. Truy vấn & Ngữ cảnh ban đầu
    query: str
    intent: str
    pdf_path: Optional[str]

    # 2. Retrieval State
    sub_queries: list[str]
    retrieved_docs: list[dict[str, Any]]

    # 3. Reader State (Bóc tách cấu trúc tài liệu PDF)
    document_schema: dict[str, Any]

    # 4. Summarizer & Generation State
    answer: str
    evidence_spans: list[str]

    # 5. Fact-Checker & Verification State
    claims: list[str]
    verified_claims: list[dict[str, Any]]
    fact_check_label: str  # "support" | "contradict" | "not_enough_info"
    critique_reason: str

    # 6. Final Synthesis State
    final_report: str
    confidence: str        # "high" | "low"

    # 7. Audit & Self-Correction Trace
    retry_count: int       # Số lần thử lại ở Orchestrator tầng ngoài (tối đa 3)
    reasoning_trace: Annotated[list[dict[str, Any]], operator.add]
