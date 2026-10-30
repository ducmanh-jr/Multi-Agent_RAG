from typing import Literal, Any
from langgraph.graph import StateGraph, START, END

from core.state import AgentState
from core.config import settings
from utils.logger import logger

from agents.retrieval.node import RetrievalAgent
from agents.reader.node import ReaderAgent
from agents.summarizer.node import SummarizerAgent
from agents.fact_checker.node import FactCheckerAgent
from agents.writer.node import WriterAgent

_agents: dict[str, Any] = {}

def orchestrator_router_node(state: AgentState) -> dict[str, Any]:
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    logger.info(f"[Orchestrator] Bắt đầu luồng cho query: '{query}' (retry_count: {retry_count})")
    return {
        "intent": "academic_research",
        "retry_count": retry_count,
        "reasoning_trace": [{"node": "Orchestrator", "action": "route_to_retrieval", "retry_count": retry_count}]
    }

def retrieval_node(state: AgentState) -> dict[str, Any]:
    return _agents["retrieval"].run(state)

def reader_node(state: AgentState) -> dict[str, Any]:
    return _agents["reader"].run(state)

def summarizer_node(state: AgentState) -> dict[str, Any]:
    return _agents["summarizer"].run(state)

def fact_checker_node(state: AgentState) -> dict[str, Any]:
    return _agents["fact_checker"].run(state)

def retry_loop_node(state: AgentState) -> dict[str, Any]:
    current_retry = state.get("retry_count", 0) + 1
    logger.warning(f"[Orchestrator] Fact-Checker yêu cầu kiểm tra lại! Tăng retry lên {current_retry}/3")
    return {
        "retry_count": current_retry,
        "reasoning_trace": [{"node": "Orchestrator_SelfCorrection", "action": "retry_increment", "new_retry": current_retry}]
    }

def escalation_node(state: AgentState) -> dict[str, Any]:
    logger.error("[Escalation] Đã chạm giới hạn retry tối đa! Xuất báo cáo kèm cảnh báo.")
    answer = state.get("answer", "Không đủ tài liệu hỗ trợ.")
    warning_report = (
        f"# CẢNH BÁO ĐỘ TIN CẬY THẤP (LOW CONFIDENCE REPORT)\n\n"
        f"> **Lưu ý:** Câu trả lời dưới đây chưa vượt qua được vòng kiểm chứng Fact-Check sau 3 lần thử lại.\n\n"
        f"### Nội dung sơ bộ:\n{answer}\n\n"
        f"### Khuyến nghị:\nCần bổ sung thêm tài liệu chuyên ngành vào corpus hoặc thu hẹp câu hỏi."
    )
    return {
        "confidence": "low",
        "final_report": warning_report,
        "reasoning_trace": [{"node": "EscalationNode", "action": "escalate_with_warning", "confidence": "low"}]
    }

def writer_node(state: AgentState) -> dict[str, Any]:
    return _agents["writer"].run(state)

# ===================== CONDITIONAL EDGES =====================

def route_after_retrieval(state: AgentState) -> Literal["reader_node", "summarizer_node"]:
    if state.get("pdf_path"):
        logger.info("[Routing] Phát hiện file PDF -> Chuyển sang Reader Agent.")
        return "reader_node"
    logger.info("[Routing] Không có PDF -> Chuyển thẳng sang Summarizer Agent.")
    return "summarizer_node"

def route_after_fact_check(state: AgentState) -> Literal["writer_node", "retry_loop_node", "escalation_node"]:
    label = state.get("fact_check_label", "not_enough_info")
    retry_count = state.get("retry_count", 0)
    
    if label == "support":
        logger.info("[Routing] Fact-check: SUPPORT -> Tiến hành viết báo cáo cuối cùng.")
        return "writer_node"
        
    if retry_count >= settings.MAX_ORCHESTRATOR_RETRY:
        logger.warning(f"[Routing] Đã retry {retry_count} lần -> Kích hoạt Escalation Node.")
        return "escalation_node"
        
    logger.info(f"[Routing] Fact-check: {label} (retry {retry_count}/3) -> Quay lại Retrieval Agent.")
    return "retry_loop_node"

# ===================== BUILD GRAPH =====================

def build_research_rag_graph():
    _agents["retrieval"] = RetrievalAgent()
    _agents["reader"] = ReaderAgent()
    _agents["summarizer"] = SummarizerAgent()
    _agents["fact_checker"] = FactCheckerAgent()
    _agents["writer"] = WriterAgent()

    workflow = StateGraph(AgentState)
    
    workflow.add_node("orchestrator_router", orchestrator_router_node)
    workflow.add_node("retrieval_node", retrieval_node)
    workflow.add_node("reader_node", reader_node)
    workflow.add_node("summarizer_node", summarizer_node)
    workflow.add_node("fact_checker_node", fact_checker_node)
    workflow.add_node("retry_loop_node", retry_loop_node)
    workflow.add_node("escalation_node", escalation_node)
    workflow.add_node("writer_node", writer_node)
    
    workflow.add_edge(START, "orchestrator_router")
    workflow.add_edge("orchestrator_router", "retrieval_node")
    
    workflow.add_conditional_edges(
        "retrieval_node",
        route_after_retrieval,
        {
            "reader_node": "reader_node",
            "summarizer_node": "summarizer_node"
        }
    )
    
    workflow.add_edge("reader_node", "summarizer_node")
    workflow.add_edge("summarizer_node", "fact_checker_node")
    
    workflow.add_conditional_edges(
        "fact_checker_node",
        route_after_fact_check,
        {
            "writer_node": "writer_node",
            "retry_loop_node": "retry_loop_node",
            "escalation_node": "escalation_node"
        }
    )
    
    workflow.add_edge("retry_loop_node", "retrieval_node")
    workflow.add_edge("writer_node", END)
    workflow.add_edge("escalation_node", END)
    
    return workflow.compile()
