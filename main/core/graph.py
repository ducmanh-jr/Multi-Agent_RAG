from functools import partial
from typing import Any, Callable, Optional
from langgraph.graph import END, START, StateGraph

from agents.agent0_scout.node import ScoutAgent
from core.citation_audit import write_with_citation_audit
from core.config import settings
from core.coordination import decide_handoff, fact_check_accepted
from core.figure_qa import (
    add_figure_evidence, attach_answer_figures, check_visual_answer,
    draft_visual_answer, figure_context, write_visual_answer,
)
from core.handoff_inputs import prepare_handoff_input
from core.state import AgentState
from core.supervisor import supervisor_planner
from modes.registry import get_active_agents, get_agent_runners, normalize_mode
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
        _agents.update({"scout": ScoutAgent(), **get_active_agents()})
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


def escalation_node(state: AgentState) -> dict[str, Any]:
    """
    Escalation Node (Lối thoát an toàn):
    Khi vượt quá số vòng retry nhưng vẫn còn mệnh đề chưa kiểm chứng,
    chuyển sang Writer với cờ confidence='low' hoặc xuất báo cáo có cảnh báo rõ ràng.
    """
    return _write_escalated_report(state, _get_fallback_agents()["writer"].run)


def _write_escalated_report(
    state: AgentState, write: Callable[[AgentState], dict[str, Any]]
) -> dict[str, Any]:
    logger.warning("[Escalation] Bằng chứng chưa đủ sau khi thẩm định. Chuyển sang Writer với độ tin cậy thấp.")
    state_copy = state.copy()
    state_copy["confidence"] = "low"

    def write_or_warn(writer_state: AgentState) -> dict[str, Any]:
        try:
            return write(writer_state)
        except Exception:
            answer = "Chưa thể xuất báo cáo đã đối chiếu với nguồn."
            return {
                "final_report": answer,
                "reasoning_trace": [{"node": "EscalationNode", "action": "escalate_with_warning"}],
            }

    result = write_with_citation_audit(state_copy, write_or_warn)
    result["confidence"] = "low"
    result["final_report"] = (
        "> **Độ tin cậy thấp.** Bằng chứng hiện có chưa đủ để xác nhận toàn bộ câu trả lời.\n\n"
        + (result.get("final_report") or "")
    )
    result["reasoning_trace"].append({
        "node": "EscalationNode", "action": "unresolved_review", "retry_count": state.get("retry_count", 0),
    })
    return result


def writer_node(state: AgentState) -> dict[str, Any]:
    res = write_with_citation_audit(state, _get_fallback_agents()["writer"].run)
    if "confidence" not in res:
        res["confidence"] = "high" if state.get("verified_claims") else "low"
    return res


# ===================== CONDITIONAL EDGES =====================

def _handoff_for(state: AgentState, role: str, key: str) -> dict[str, Any] | None:
    trace = state.get("reasoning_trace") or []
    if trace:
        last = trace[-1]
        if last.get("action") == "agent_handoff" and last.get(key) == role:
            return last
    return None


def route_after_scout(state: AgentState) -> str:
    trace = state.get("reasoning_trace") or []
    if trace and trace[-1].get("action") in {"discovery_clarify", "discovery_rejected"}:
        return END
    return "retrieval_node"


def route_after_summarizer(state: AgentState) -> str:
    request = _handoff_for(state, "summarizer", "sender")
    return f"{request['recipient']}_node" if request else "fact_checker_node"


def route_after_fact_check(state: AgentState) -> str:
    request = _handoff_for(state, "fact_checker", "sender")
    if request:
        return f"{request['recipient']}_node"
    if figure_context(state).get("explicit") and not state.get("rejected_claims"):
        return "writer_node"
    return "writer_node" if fact_check_accepted(state) else "escalation_node"


_agents: dict[str, Any] = {}


def build_research_rag_graph(mode: Optional[str] = None):
    """Build a run-local team. Only the two reviewing roles can initiate corrective work."""
    run_agents = get_agent_runners(mode)
    reader_mode = normalize_mode(mode or settings.get_agent_mode("reader") or settings.EXECUTION_MODE)
    scout = ScoutAgent()

    def run_specialist(role: str, state: AgentState) -> dict[str, Any]:
        request = _handoff_for(state, role, "recipient")
        inputs = prepare_handoff_input(state, request) if request else state
        visual = figure_context(inputs).get("explicit", False)
        if visual and role == "summarizer":
            update = draft_visual_answer(inputs)
        elif visual and role == "fact_checker":
            update = check_visual_answer(inputs)
        else:
            update = run_agents[role](inputs)
        if role == "reader":
            update = add_figure_evidence(inputs, update, reader_mode)
        if role == "reader" and state.get("retry_count") and state.get("source_units"):
            # A corrective read supplements the evidence already gathered for this question.
            units = []
            seen = set()
            for unit in [*(update.get("source_units") or []), *state["source_units"]]:
                key = (unit.get("doc_id"), unit.get("page_no"), unit.get("id") or unit.get("text"))
                if key not in seen:
                    seen.add(key)
                    units.append(unit)
            update = {**update, "source_units": units}
        trace = [
            entry for entry in update.get("reasoning_trace") or []
            if entry.get("action") not in {"agent_handoff", "agent_handoff_received"}
        ]
        if request:
            trace.insert(0, {
                "action": "agent_handoff_received",
                "agent": role,
                "sender": request["sender"],
                "recipient": role,
                "reason": request["reason"],
                "signature": request["signature"],
                "retry_count": request["retry_count"],
            })

        if role == "summarizer" and state.get("retry_count", 0) > 0:
            # Judgements and citations from the previous draft cannot certify its replacement.
            update = {
                "answer": "", "claims": [], "evidence_spans": [],
                "critic_action": "approve", "critic_feedback": {},
                "verified_claims": [], "rejected_claims": [], "evidence_gaps": [],
                "short_answer": None, "answer_status": None, "fact_check_label": "not_enough_info",
                "statistical_audit": None, "code_verification": None,
                **update,
            }

        if not visual and role in {"summarizer", "fact_checker"}:
            reviewed = {**state, **update, "reasoning_trace": [*(state.get("reasoning_trace") or []), *trace]}
            decision = decide_handoff(role, reviewed)
            if decision is not None:
                attempt = state.get("retry_count", 0) + 1
                trace.append({**decision.as_trace(), "agent": role, "retry_count": attempt})
                update["retry_count"] = attempt
        update["reasoning_trace"] = trace
        return update

    def _writer_node(state: AgentState) -> dict[str, Any]:
        writer = write_visual_answer if figure_context(state).get("explicit") else run_agents["writer"]
        result = write_with_citation_audit(state, writer)
        if state.get("critic_action") == "lower_confidence":
            result["confidence"] = "low"
        else:
            result.setdefault("confidence", "high" if state.get("verified_claims") else "low")
        return attach_answer_figures(state, result)

    workflow = StateGraph(AgentState)

    workflow.add_node("orchestrator_router", orchestrator_router_node)
    workflow.add_node("scout_node", scout.run)
    for role in ("retrieval", "reader", "summarizer", "fact_checker"):
        workflow.add_node(f"{role}_node", partial(run_specialist, role))
    workflow.add_node("escalation_node", lambda state: attach_answer_figures(state, _write_escalated_report(state, run_agents["writer"])))
    workflow.add_node("writer_node", _writer_node)

    workflow.add_edge(START, "orchestrator_router")
    workflow.add_edge("orchestrator_router", "scout_node")
    workflow.add_conditional_edges("scout_node", route_after_scout, ["retrieval_node", END])
    workflow.add_edge("retrieval_node", "reader_node")
    workflow.add_edge("reader_node", "summarizer_node")

    correction_targets = ["retrieval_node", "reader_node", "summarizer_node"]
    workflow.add_conditional_edges(
        "summarizer_node", route_after_summarizer, [*correction_targets, "fact_checker_node"]
    )
    workflow.add_conditional_edges(
        "fact_checker_node", route_after_fact_check, [*correction_targets, "writer_node", "escalation_node"]
    )
    workflow.add_edge("writer_node", END)
    workflow.add_edge("escalation_node", END)

    return workflow.compile()


build_graph = build_research_rag_graph

