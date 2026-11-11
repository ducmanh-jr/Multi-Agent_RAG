from typing import Any, Literal, Optional
from langgraph.graph import END, START, StateGraph

from agents.agent0_scout.node import ScoutAgent
from core.config import settings
from core.state import AgentState
from core.supervisor import supervisor_planner
from modes.registry import get_active_agents, get_mode_agents, normalize_mode
from utils.logger import logger


# ===================== NODE FUNCTIONS =====================

def orchestrator_router_node(state: AgentState) -> dict[str, Any]:
    """
    Agentic Supervisor Router Node (L3-L4):
    Phân tích query, phân loại intent + độ phức tạp (LOW/MEDIUM/HIGH),
    tách sub_queries đa bước và trích xuất thực thể bắt buộc.
    """
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    evidence_gaps = state.get("evidence_gaps") or []

    plan = supervisor_planner.plan_query(
        query=query,
        evidence_gaps=evidence_gaps,
        retry_count=retry_count,
    )
    logger.info(
        f"[Supervisor Planner] Query='{query[:70]}' | Intent={plan.intent} | "
        f"Complexity={plan.complexity} | SubQueries={len(plan.sub_queries)} | Retry={retry_count}"
    )

    intent_labels = {
        "FACTOID": "Tra cứu số liệu / Thực thể",
        "results": "Tra cứu số liệu / Thực thể",
        "COMPARISON": "So sánh đối chiếu đa tài liệu",
        "compare": "So sánh đối chiếu đa tài liệu",
        "SYNTHESIS": "Tổng quan & Phân tích chuyên sâu",
        "summarize": "Tóm tắt & Tổng quan tài liệu",
        "methodology": "Phân tích phương pháp & Kiến trúc",
        "limitation": "Đánh giá hạn chế & Thách thức",
        "academic_research": "Nghiên cứu khoa học chuyên sâu",
        "OUT_OF_DOMAIN": "Kiểm tra phạm vi dữ liệu",
    }
    intent_desc = intent_labels.get(plan.intent, plan.intent)
    trace = {
        "node": "SupervisorPlanner",
        "stage": "Thought",
        "action": f"Phân tích mục tiêu: Ý định '{intent_desc}' • Độ phức tạp '{plan.complexity}' • Tự động phân rã {len(plan.sub_queries)} sub-queries",
        "intent": plan.intent,
        "complexity": plan.complexity,
        "sub_queries": plan.sub_queries,
        "required_entities": plan.required_entities,
        "alpha": plan.alpha,
        "top_k": plan.top_k,
        "retry_count": retry_count,
    }
    return {
        "intent": plan.intent,
        "complexity": plan.complexity,
        "sub_queries": plan.sub_queries,
        "required_entities": plan.required_entities,
        "is_numeric_query": plan.is_numeric_query,
        "is_comparative_query": plan.is_comparative_query,
        "alpha": plan.alpha,
        "top_k": plan.top_k,
        "retry_count": retry_count,
        "reasoning_trace": [trace],
    }


def _get_fallback_agents():
    global _agents
    if not _agents:
        _agents.update(get_active_agents())
    return _agents


def scout_node(state: AgentState) -> dict[str, Any]:
    """Standalone node runner for Scout Agent (for unit testing and direct invocation)."""
    return _get_fallback_agents()["scout"].run(state)


def retrieval_node(state: AgentState) -> dict[str, Any]:
    """Standalone node runner for Retrieval Agent (for unit testing and direct invocation)."""
    return _get_fallback_agents()["retrieval"].run(state)


def reader_node(state: AgentState) -> dict[str, Any]:
    """Standalone node runner for Reader Agent (for unit testing and direct invocation)."""
    return _get_fallback_agents()["reader"].run(state)


def summarizer_node(state: AgentState) -> dict[str, Any]:
    """Standalone node runner for Summarizer Agent (for unit testing and direct invocation)."""
    return _get_fallback_agents()["summarizer"].run(state)


def fact_checker_node(state: AgentState) -> dict[str, Any]:
    """Standalone node runner for Fact Checker Agent (for unit testing and direct invocation)."""
    return _get_fallback_agents()["fact_checker"].run(state)


def retry_loop_node(state: AgentState) -> dict[str, Any]:
    """
    Node trung gian xử lý vòng lặp tự sửa lỗi (Two-Tier Self-Correction):
    Tăng retry_count += 1 và sinh bộ sub_queries bù lỗ hổng thông tin (Gap-Driven Re-Query).
    """
    current_retry = state.get("retry_count", 0) + 1
    query = state.get("query", "")
    gaps = state.get("evidence_gaps") or []
    raw_critic = state.get("critic_feedback")
    critic_fb = raw_critic if isinstance(raw_critic, dict) else {}

    missing_points = [
        str(g.get("missing_aspect") or g.get("suggested_query") or "")
        for g in gaps
        if isinstance(g, dict)
    ] + list(critic_fb.get("missing_key_points") or [])
    unsupported = list(critic_fb.get("unsupported_claims") or [])

    new_sub_queries = supervisor_planner.build_gap_requeries(
        original_query=query,
        missing_points=missing_points,
        unsupported_claims=unsupported,
    )

    logger.warning(
        f"[Orchestrator Self-Correction] Kích hoạt vòng tìm kiếm bổ sung {current_retry}/{settings.MAX_ORCHESTRATOR_RETRY} "
        f"với sub_queries={new_sub_queries}"
    )

    max_loops = state.get("max_reflection_loops") or state.get("max_iterations") or settings.MAX_ORCHESTRATOR_RETRY
    trace = {
        "node": "Orchestrator_SelfCorrection",
        "stage": "Reflection",
        "action": f"Kích hoạt vòng tự sửa sai ({current_retry}/{max_loops}): Tự động phát hiện lỗ hổng & truy hồi bù khuyết cho {len(missing_points)} điểm thiếu",
        "new_retry_count": current_retry,
        "new_sub_queries": new_sub_queries,
    }
    return {
        "retry_count": current_retry,
        "sub_queries": new_sub_queries,
        "critic_action": "approve",  # Reset flag before re-entering retrieval
        "reasoning_trace": [trace],
    }


def escalation_node(state: AgentState) -> dict[str, Any]:
    """
    Escalation Node (Lối thoát an toàn):
    Khi vượt quá số vòng retry nhưng vẫn còn mệnh đề chưa kiểm chứng,
    chuyển sang Writer với cờ confidence='low' hoặc xuất báo cáo có cảnh báo rõ ràng.
    """
    logger.warning("[Escalation] Đã chạm giới hạn retry. Chuyển sang Writer với cảnh báo độ tin cậy.")
    state_copy = dict(state)
    state_copy["confidence"] = "low"
    try:
        writer_out = _get_fallback_agents()["writer"].run(state_copy)  # type: ignore[arg-type]
        writer_out["confidence"] = "low"
        return writer_out
    except Exception:
        answer = state.get("answer", "Không tìm thấy đủ bằng chứng trong tài liệu.")
        warning_report = (
            f"# BÁO CÁO NGHIÊN CỨU (ĐỘ TIN CẬY THẤP)\n\n"
            f"> **Lưu ý:** Một số mệnh đề chưa tìm thấy bằng chứng nguyên văn sau {state.get('retry_count', 1)} vòng truy xuất.\n\n"
            f"{answer}"
        )
        return {
            "confidence": "low",
            "final_report": warning_report,
            "reasoning_trace": [{"node": "EscalationNode", "action": "escalate_with_warning"}],
        }


def writer_node(state: AgentState) -> dict[str, Any]:
    res = _get_fallback_agents()["writer"].run(state)
    if "confidence" not in res:
        res["confidence"] = "high" if state.get("verified_claims") else "low"
    return res


# ===================== CONDITIONAL EDGES =====================

def route_after_summarizer(state: AgentState) -> Literal["fact_checker_node", "retry_loop_node"]:
    """
    Inner Critic Gate:
    Nếu Summarizer Critic phát hiện ngữ cảnh hiện tại hoàn toàn thiếu thông tin ('request_more_context')
    và còn ngân sách retry -> Quay thẳng về retry_loop_node -> retrieval_node để lấy thêm chunks!
    """
    critic_action = state.get("critic_action", "approve")
    retry_count = state.get("retry_count", 0)
    no_new_evidence = state.get("no_new_evidence", False)
    max_loops = state.get("max_reflection_loops") or state.get("max_iterations") or settings.MAX_ORCHESTRATOR_RETRY

    if (
        critic_action == "request_more_context"
        and retry_count < max_loops
        and not no_new_evidence
    ):
        logger.info(f"[Routing] Inner Critic yêu cầu 'request_more_context' (vòng {retry_count+1}/{max_loops}) -> Quay về Retrieval lấy thêm tài liệu.")
        return "retry_loop_node"

    return "fact_checker_node"


def route_after_fact_check(state: AgentState) -> Literal["writer_node", "retry_loop_node", "escalation_node"]:
    """
    Outer Fact-Checker Gate:
    - Nếu support -> Sang Writer Agent
    - Nếu contradict / not_enough_info:
        + Nếu retry_count >= max_loops hoặc no_new_evidence -> Sang Escalation Node
        + Ngược lại -> Sang retry_loop_node -> Retrieval Agent
    """
    label = state.get("fact_check_label", "not_enough_info")
    retry_count = state.get("retry_count", 0)
    no_new_evidence = state.get("no_new_evidence", False)
    max_loops = state.get("max_reflection_loops") or state.get("max_iterations") or settings.MAX_ORCHESTRATOR_RETRY

    if label in ["support", "supported"]:
        logger.info("[Routing] Fact-check: SUPPORT -> Tiến hành viết báo cáo cuối cùng.")
        return "writer_node"

    if retry_count >= max_loops or no_new_evidence:
        logger.warning(
            f"[Routing] Dừng lặp (retry={retry_count}/{max_loops}, no_new_evidence={no_new_evidence}) -> Chuyển sang Escalation/Writer."
        )
        return "escalation_node"

    logger.info(f"[Routing] Fact-check: {label} (retry {retry_count}/{max_loops}) -> Quay lại Retrieval.")
    return "retry_loop_node"


# ===================== BUILD GRAPH =====================

_agents: dict[str, Any] = {}


def build_research_rag_graph(mode: Optional[str] = None):
    """
    Xây dựng và biên dịch đồ thị LangGraph Two-Tier Self-Correction (L3-L4).
    Sử dụng factory closures để liên kết trực tiếp agent instance vào các node của đồ thị,
    tránh xung đột race condition trên biến toàn cục _agents khi chạy đa luồng/đa chế độ.
    """
    local_agents: dict[str, Any] = {"scout": ScoutAgent()}
    if mode:
        norm_mode = normalize_mode(mode)
        settings.EXECUTION_MODE = norm_mode
        for a in ["retrieval", "reader", "summarizer", "factchecker", "writer"]:
            settings.set_agent_mode(a, norm_mode)
        local_agents.update(get_mode_agents(norm_mode))
    else:
        local_agents.update(get_active_agents())

    # Đồng bộ với _agents toàn cục để tương thích ngược nếu có module khác đọc
    _agents.update(local_agents)

    def _scout_node(state: AgentState) -> dict[str, Any]:
        return local_agents["scout"].run(state)

    def _retrieval_node(state: AgentState) -> dict[str, Any]:
        return local_agents["retrieval"].run(state)

    def _reader_node(state: AgentState) -> dict[str, Any]:
        return local_agents["reader"].run(state)

    def _summarizer_node(state: AgentState) -> dict[str, Any]:
        return local_agents["summarizer"].run(state)

    def _fact_checker_node(state: AgentState) -> dict[str, Any]:
        return local_agents["fact_checker"].run(state)

    def _writer_node(state: AgentState) -> dict[str, Any]:
        res = local_agents["writer"].run(state)
        if "confidence" not in res:
            res["confidence"] = "high" if state.get("verified_claims") else "low"
        return res

    def _escalation_node(state: AgentState) -> dict[str, Any]:
        logger.warning("[Escalation] Đã chạm giới hạn retry. Chuyển sang Writer với cảnh báo độ tin cậy.")
        state_copy = dict(state)
        state_copy["confidence"] = "low"
        try:
            writer_out = local_agents["writer"].run(state_copy)
            writer_out["confidence"] = "low"
            return writer_out
        except Exception:
            answer = state.get("answer", "Không tìm thấy đủ bằng chứng trong tài liệu.")
            warning_report = (
                f"# BÁO CÁO NGHIÊN CỨU (ĐỘ TIN CẬY THẤP)\n\n"
                f"> **Lưu ý:** Một số mệnh đề chưa tìm thấy bằng chứng nguyên văn sau {state.get('retry_count', 1)} vòng truy xuất.\n\n"
                f"{answer}"
            )
            return {
                "confidence": "low",
                "final_report": warning_report,
                "reasoning_trace": [{"node": "EscalationNode", "action": "escalate_with_warning"}],
            }

    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator_router", orchestrator_router_node)
    workflow.add_node("scout_node", _scout_node)
    workflow.add_node("retrieval_node", _retrieval_node)
    workflow.add_node("reader_node", _reader_node)
    workflow.add_node("summarizer_node", _summarizer_node)
    workflow.add_node("fact_checker_node", _fact_checker_node)
    workflow.add_node("retry_loop_node", retry_loop_node)
    workflow.add_node("escalation_node", _escalation_node)
    workflow.add_node("writer_node", _writer_node)

    workflow.add_edge(START, "orchestrator_router")
    workflow.add_edge("orchestrator_router", "scout_node")
    workflow.add_edge("scout_node", "retrieval_node")
    workflow.add_edge("retrieval_node", "reader_node")
    workflow.add_edge("reader_node", "summarizer_node")

    # Tier 1 Gate: Summarizer Inner Critic -> (Fact-Checker or Direct Re-Query)
    workflow.add_conditional_edges(
        "summarizer_node",
        route_after_summarizer,
        {
            "fact_checker_node": "fact_checker_node",
            "retry_loop_node": "retry_loop_node",
        },
    )

    # Tier 2 Gate: Fact-Checker -> (Writer, Retry Loop, or Escalation)
    workflow.add_conditional_edges(
        "fact_checker_node",
        route_after_fact_check,
        {
            "writer_node": "writer_node",
            "retry_loop_node": "retry_loop_node",
            "escalation_node": "escalation_node",
        },
    )

    workflow.add_edge("retry_loop_node", "retrieval_node")
    workflow.add_edge("writer_node", END)
    workflow.add_edge("escalation_node", END)

    return workflow.compile()


build_graph = build_research_rag_graph

