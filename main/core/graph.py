from typing import Any, Callable, Literal, Optional
from langgraph.graph import END, START, StateGraph

from agents.agent0_scout.node import ScoutAgent
from core.config import settings
from core.model_gateway import ProviderConfigurationError, ProviderPolicyError
from core.schemas import map_evidence_to_legacy_fact_check_label
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.tool_registry import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolPermissionDeniedError,
    ToolRegistry,
    tool_registry,
)
from modes.registry import get_active_agents, get_mode_agents, normalize_mode
from utils.llm_client import llm_client
from utils.logger import logger


# ===================== STANDARDIZED TEMPLATES (AGENT.md Section 11) =====================

TEMPLATE_NO_DOCUMENT_SELECTED = (
    "Câu hỏi này cần được trả lời dựa trên nội dung tài liệu, nhưng hiện tại chưa có tài liệu "
    "phù hợp được chọn. Vui lòng tải lên hoặc chọn tài liệu để tôi kiểm tra."
)

TEMPLATE_UNSUPPORTED_CAPABILITY = (
    "Tôi chưa có công cụ hoặc dữ liệu cần thiết để thực hiện chính xác yêu cầu này. "
    "Tôi có thể hỗ trợ giải thích lý thuyết hoặc hướng dẫn quy trình thực hiện."
)

TEMPLATE_SAFETY_RESTRICTED = (
    "Tôi không thể hỗ trợ yêu cầu này vì vi phạm nguyên tắc an toàn và bảo mật hệ thống."
)

TEMPLATE_WEB_RESEARCH_DISABLED = (
    "Yêu cầu này cần tìm kiếm nghiên cứu hoặc thông tin bên ngoài, nhưng chế độ tìm kiếm ngoài "
    "(auto-scout) hiện chưa được bật. Vui lòng bật auto-scout hoặc tải tài liệu PDF liên quan lên hệ thống."
)


# ===================== NODE FUNCTIONS =====================

def orchestrator_router_node(state: AgentState) -> dict[str, Any]:
    """
    Agentic Supervisor & General-Purpose Intent Router Node:
    Phân loại 9 nhóm Primary Intent (AGENT.md Mục 7), kiểm tra InputStatus / QueryResolution,
    quyết định nhánh thực thi (Direct Assistant / Task Execution / Template / Scout / Retrieval)
    và lập kế hoạch sub-queries khi cần RAG.
    """
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    evidence_gaps = state.get("evidence_gaps") or []
    pdf_path = state.get("pdf_path")
    pdf_paths = state.get("pdf_paths")
    auto_scout = bool(state.get("auto_scout", False))
    tool_req = state.get("tool_request") or {}
    requested_tool = tool_req.get("name") if isinstance(tool_req, dict) else None

    plan = supervisor_planner.plan_query(
        query=query,
        evidence_gaps=evidence_gaps,
        retry_count=retry_count,
        pdf_path=pdf_path,
        pdf_paths=pdf_paths,
        auto_scout=auto_scout,
        requested_tool=requested_tool,
    )
    logger.info(
        f"[Supervisor Planner] Query='{query[:70]}' | PrimaryIntent={plan.primary_intent} | "
        f"Intent={plan.intent} | InputStatus={plan.input_status} | "
        f"Scout={plan.needs_scout} | Retrieval={plan.needs_retrieval}"
    )

    trace = {
        "node": "SupervisorPlanner",
        "stage": "Thought",
        "action": (
            f"Phân loại ý định: '{plan.primary_intent}' ({plan.sub_intent or plan.intent}) • "
            f"Trạng thái đầu vào: {plan.input_status} • Độ phức tạp: {plan.complexity}"
        ),
        "primary_intent": plan.primary_intent,
        "sub_intent": plan.sub_intent,
        "intent": plan.intent,
        "input_status": plan.input_status,
        "query_resolution": plan.query_resolution,
        "needs_scout": plan.needs_scout,
        "needs_retrieval": plan.needs_retrieval,
        "complexity": plan.complexity,
        "sub_queries": plan.sub_queries,
        "required_entities": plan.required_entities,
        "alpha": plan.alpha,
        "top_k": plan.top_k,
        "retry_count": retry_count,
    }
    return {
        "primary_intent": plan.primary_intent,
        "sub_intent": plan.sub_intent,
        "input_status": plan.input_status,
        "query_resolution": plan.query_resolution,
        "processing_status": "RUNNING",
        "needs_scout": plan.needs_scout,
        "needs_retrieval": plan.needs_retrieval,
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


def direct_assistant_node(state: AgentState) -> dict[str, Any]:
    """
    Handles GENERAL_QA and TASK_ASSISTED_RESPONSE directly without Scout or RAG Retrieval.
    Respects ModelGateway policies via llm_client.generate().
    """
    query = (state.get("query") or "").strip()
    primary_intent = state.get("primary_intent", "GENERAL_QA")
    system_prompt = (
        "Bạn là Trợ lý AI đa năng. Trả lời trực diện, ĐÚNG VÀ ĐỦ, không thừa, không thiếu. "
        "Đây là câu hỏi kiến thức chung hoặc hỗ trợ viết code/hướng dẫn kỹ thuật — không giả vờ trích dẫn PDF."
    )
    try:
        answer = llm_client.generate(
            prompt=query,
            system_instruction=system_prompt,
            temperature=0.2,
            json_mode=False,
        )
        answer = (answer or "").strip()
        if not answer:
            raise RuntimeError("Empty response from direct assistant LLM.")
    except (ProviderPolicyError, ProviderConfigurationError) as exc:
        err_msg = f"Không thể tạo câu trả lời trực tiếp do lỗi cấu hình/chính sách mô hình: {exc}"
        trace = {
            "node": "DirectAssistant",
            "stage": "Error",
            "action": "direct_assistant_policy_or_config_error",
            "error": str(exc),
        }
        return {
            "answer": err_msg,
            "final_report": err_msg,
            "processing_status": "LLM_FAILED",
            "evidence_status": "NOT_APPLICABLE",
            "fact_check_label": "not_enough_info",
            "error_stage": "direct_assistant",
            "error_code": type(exc).__name__,
            "error_message": str(exc),
            "confidence": "low",
            "reasoning_trace": [trace],
        }
    except Exception as exc:
        err_msg = f"Không thể xử lý yêu cầu trực tiếp lúc này do lỗi mô hình: {exc}"
        trace = {
            "node": "DirectAssistant",
            "stage": "Error",
            "action": "direct_assistant_llm_failed",
            "error": str(exc),
        }
        return {
            "answer": err_msg,
            "final_report": err_msg,
            "processing_status": "LLM_FAILED",
            "evidence_status": "NOT_APPLICABLE",
            "fact_check_label": "not_enough_info",
            "error_stage": "direct_assistant",
            "error_code": "LLM_FAILED",
            "error_message": str(exc),
            "confidence": "low",
            "reasoning_trace": [trace],
        }

    trace = {
        "node": "DirectAssistant",
        "stage": "Response",
        "action": f"Trả lời trực tiếp ({primary_intent}) không qua Scout/RAG",
        "primary_intent": primary_intent,
    }
    return {
        "answer": answer,
        "final_report": answer,
        "processing_status": "SUCCESS",
        "evidence_status": "NOT_APPLICABLE",
        "fact_check_label": map_evidence_to_legacy_fact_check_label("NOT_APPLICABLE"),
        "confidence": "high",
        "reasoning_trace": [trace],
    }


def _run_task_execution(state: AgentState, active_registry: ToolRegistry) -> dict[str, Any]:
    """Executes TASK_EXECUTION strictly through ToolRegistry + PermissionGuard."""
    tool_req = state.get("tool_request") or {}
    tool_name = (
        str(tool_req.get("name") or "").strip().lower()
        if isinstance(tool_req, dict)
        else ""
    )
    if not tool_name:
        sub = str(state.get("sub_intent") or "").strip().lower()
        tool_name = "" if sub in ("", "tool_operation") else sub

    payload = tool_req.get("payload") if isinstance(tool_req, dict) else {}
    granted_level = (
        tool_req.get("granted_level", "L0") if isinstance(tool_req, dict) else "L0"
    )
    approved_raw = (
        tool_req.get("approved_tools") if isinstance(tool_req, dict) else None
    )
    approved_tools = set(approved_raw) if approved_raw else set()

    try:
        result = active_registry.execute(
            name=tool_name,
            payload=payload or {},
            granted_level=granted_level,
            approved_tools=approved_tools,
        )
    except ToolNotFoundError as exc:
        trace = {
            "node": "TaskExecution",
            "stage": "ToolCheck",
            "action": "tool_not_found",
            "tool": tool_name,
            "error": str(exc),
        }
        return {
            "primary_intent": "UNSUPPORTED_CAPABILITY",
            "answer": TEMPLATE_UNSUPPORTED_CAPABILITY,
            "final_report": TEMPLATE_UNSUPPORTED_CAPABILITY,
            "processing_status": "TOOL_FAILED",
            "evidence_status": "NOT_APPLICABLE",
            "fact_check_label": "support",
            "error_stage": "task_execution",
            "error_code": exc.error_code,
            "error_message": str(exc),
            "confidence": "low",
            "reasoning_trace": [trace],
        }
    except ToolPermissionDeniedError as exc:
        msg = f"Không đủ quyền hoặc chưa có phê duyệt để thực thi công cụ '{tool_name}': {exc}"
        trace = {
            "node": "TaskExecution",
            "stage": "PermissionGuard",
            "action": "tool_permission_denied",
            "tool": tool_name,
            "error": str(exc),
        }
        return {
            "answer": msg,
            "final_report": msg,
            "processing_status": "TOOL_FAILED",
            "evidence_status": "NOT_APPLICABLE",
            "fact_check_label": "support",
            "error_stage": "task_execution",
            "error_code": exc.error_code,
            "error_message": str(exc),
            "confidence": "low",
            "reasoning_trace": [trace],
        }
    except ToolExecutionError as exc:
        msg = f"Công cụ '{tool_name}' gặp lỗi khi thực thi: {exc}"
        trace = {
            "node": "TaskExecution",
            "stage": "Error",
            "action": "tool_execution_failed",
            "tool": tool_name,
            "error": str(exc),
        }
        return {
            "answer": msg,
            "final_report": msg,
            "processing_status": "TOOL_FAILED",
            "evidence_status": "NOT_APPLICABLE",
            "fact_check_label": "not_enough_info",
            "error_stage": "task_execution",
            "error_code": exc.error_code,
            "error_message": str(exc),
            "confidence": "low",
            "reasoning_trace": [trace],
        }

    output_str = str(result.get("output", ""))
    trace = {
        "node": "TaskExecution",
        "stage": "Action",
        "action": f"Thực thi công cụ '{tool_name}' thành công",
        "tool": tool_name,
    }
    return {
        "tool_result": result,
        "answer": output_str,
        "final_report": output_str,
        "processing_status": "SUCCESS",
        "evidence_status": "NOT_APPLICABLE",
        "fact_check_label": map_evidence_to_legacy_fact_check_label("NOT_APPLICABLE"),
        "confidence": "high",
        "reasoning_trace": [trace],
    }


def template_response_node(state: AgentState) -> dict[str, Any]:
    """
    Produces standardized, honest responses (AGENT.md Section 11) for:
      - NO_DOCUMENT_SELECTED
      - CLARIFICATION_REQUIRED
      - SAFETY_RESTRICTED
      - UNSUPPORTED_CAPABILITY
      - WEB_RESEARCH with auto_scout disabled
      - Scout REJECT / CLARIFY / no_relevant_papers / download_failed
    """
    primary_intent = state.get("primary_intent", "GENERAL_QA")
    input_status = state.get("input_status", "OK")
    scout_decision = str(state.get("scout_decision") or "").strip().lower()
    scout_message = state.get("scout_message")

    processing_status: str = "SUCCESS"
    query_resolution = state.get("query_resolution", "CLEAR")
    error_stage: Optional[str] = None
    error_code: Optional[str] = None

    if scout_decision in ("reject", "clarify", "no_relevant_papers", "download_failed"):
        msg = scout_message or TEMPLATE_UNSUPPORTED_CAPABILITY
        if scout_decision == "clarify":
            query_resolution = "CLARIFICATION_REQUIRED"
        elif scout_decision == "download_failed":
            processing_status = "RETRIEVAL_FAILED"
            error_stage = "scout"
            error_code = "SCOUT_DOWNLOAD_FAILED"
        reason = f"scout_{scout_decision}"
    elif primary_intent == "SAFETY_RESTRICTED":
        msg = TEMPLATE_SAFETY_RESTRICTED
        reason = "safety_restricted"
    elif primary_intent == "CLARIFICATION_REQUIRED":
        query_resolution = "CLARIFICATION_REQUIRED"
        msg = (
            "Câu hỏi của bạn hiện còn ngắn hoặc có nhiều cách hiểu khác nhau. "
            "Vui lòng làm rõ một trong các hướng sau để tôi trả lời chính xác:\n"
            "1. Bạn muốn hỏi về nội dung của một tài liệu PDF cụ thể?\n"
            "2. Bạn muốn giải thích kiến thức chung hay hướng dẫn kỹ thuật?\n"
            "3. Bạn muốn tìm kiếm nghiên cứu học thuật mới về chủ đề cụ thể nào?"
        )
        reason = "clarification_required"
    elif input_status == "NO_DOCUMENT_SELECTED":
        msg = TEMPLATE_NO_DOCUMENT_SELECTED
        reason = "no_document_selected"
    elif primary_intent == "WEB_RESEARCH" and not state.get("needs_scout", False):
        msg = TEMPLATE_WEB_RESEARCH_DISABLED
        reason = "web_research_scout_disabled"
    else:
        msg = TEMPLATE_UNSUPPORTED_CAPABILITY
        reason = "unsupported_capability"

    trace = {
        "node": "TemplateResponse",
        "stage": "Response",
        "action": reason,
        "primary_intent": primary_intent,
        "input_status": input_status,
        "query_resolution": query_resolution,
    }
    return {
        "answer": msg,
        "final_report": msg,
        "query_resolution": query_resolution,
        "processing_status": processing_status,
        "evidence_status": "NOT_APPLICABLE",
        "fact_check_label": map_evidence_to_legacy_fact_check_label("NOT_APPLICABLE"),
        "error_stage": error_stage,
        "error_code": error_code,
        "confidence": "high",
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

def route_after_orchestrator(
    state: AgentState,
) -> Literal[
    "direct_assistant_node",
    "task_execution_node",
    "template_response_node",
    "scout_node",
    "retrieval_node",
]:
    """
    Primary Intent Router Gate (before Scout & Retrieval):
    Ensures RAG is a capability, never a mandatory default for general or task queries.
    """
    primary_intent = state.get("primary_intent", "DOCUMENT_QA")
    input_status = state.get("input_status", "OK")
    needs_scout = bool(state.get("needs_scout", False))
    needs_retrieval = bool(state.get("needs_retrieval", False))

    if primary_intent in ("SAFETY_RESTRICTED", "CLARIFICATION_REQUIRED", "UNSUPPORTED_CAPABILITY"):
        return "template_response_node"

    if input_status == "NO_DOCUMENT_SELECTED":
        return "template_response_node"

    if primary_intent in ("GENERAL_QA", "TASK_ASSISTED_RESPONSE"):
        return "direct_assistant_node"

    if primary_intent == "TASK_EXECUTION":
        return "task_execution_node"

    if needs_scout:
        return "scout_node"

    if needs_retrieval:
        return "retrieval_node"

    return "template_response_node"


def route_after_scout(
    state: AgentState,
) -> Literal["retrieval_node", "template_response_node"]:
    """Routes Scout REJECT / CLARIFY / failure decisions directly to TemplateResponse instead of Retrieval."""
    decision = str(state.get("scout_decision") or "").strip().lower()
    if decision in ("reject", "clarify", "no_relevant_papers", "download_failed"):
        return "template_response_node"
    return "retrieval_node"


def route_after_summarizer(
    state: AgentState,
) -> Literal["fact_checker_node", "retry_loop_node", "escalation_node"]:
    """
    Inner Critic Gate (Fail-Closed):
    - Nếu Critic thất bại (`VERIFICATION_FAILED` hoặc `critic_action == 'verification_failed'`)
      -> Chuyển thẳng sang `escalation_node` (KHÔNG tự duyệt và KHÔNG đi tiếp như kết quả đã xác minh).
    - Nếu Critic yêu cầu thêm ngữ cảnh (`request_more_context`) và còn ngân sách retry
      -> Quay về `retry_loop_node` -> `retrieval_node`.
    - Ngược lại -> Sang `fact_checker_node`.
    """
    processing_status = state.get("processing_status", "RUNNING")
    critic_action = state.get("critic_action", "approve")
    retry_count = state.get("retry_count", 0)
    no_new_evidence = state.get("no_new_evidence", False)
    max_loops = state.get("max_reflection_loops") or state.get("max_iterations") or settings.MAX_ORCHESTRATOR_RETRY

    if processing_status == "VERIFICATION_FAILED" or critic_action == "verification_failed":
        logger.warning(
            "[Routing] Inner Critic VERIFICATION_FAILED -> Fail-closed sang EscalationNode (không phê duyệt)."
        )
        return "escalation_node"

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


def build_research_rag_graph(
    mode: Optional[str] = None,
    agent_overrides: Optional[dict[str, Any]] = None,
    registry: Optional[ToolRegistry] = None,
):
    """
    Xây dựng và biên dịch đồ thị LangGraph General-Purpose Multi-Agent Assistant.
    Hỗ trợ phân nhánh Intent trước Scout/Retrieval và dependency injection (agent_overrides, registry)
    cho kiểm thử cô lập mà không gọi dịch vụ ngoài.
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

    if agent_overrides:
        local_agents.update(agent_overrides)

    active_registry = registry if registry is not None else tool_registry

    # Đồng bộ với _agents toàn cục để tương thích ngược nếu có module khác đọc
    _agents.update(local_agents)

    def _scout_node(state: AgentState) -> dict[str, Any]:
        # MIXED_QA preservation: keep user's PDF(s) while also discovering external sources
        if state.get("primary_intent") == "MIXED_QA" and (state.get("pdf_path") or state.get("pdf_paths")):
            orig_pdf = state.get("pdf_path")
            orig_pdfs = list(state.get("pdf_paths") or ([orig_pdf] if orig_pdf else []))
            scout_input = dict(state)
            scout_input["pdf_path"] = None
            res = local_agents["scout"].run(scout_input)
            ext_docs = []
            for d in res.get("retrieved_docs") or []:
                if isinstance(d, dict):
                    tagged = dict(d)
                    tagged.setdefault("source_type", "external_research")
                    ext_docs.append(tagged)
            merged_pdfs = list(orig_pdfs)
            scouted_pdf = res.get("pdf_path")
            if scouted_pdf and scouted_pdf not in merged_pdfs:
                merged_pdfs.append(scouted_pdf)
            res["pdf_path"] = orig_pdf or (merged_pdfs[0] if merged_pdfs else None)
            res["pdf_paths"] = merged_pdfs
            if ext_docs:
                res["retrieved_docs"] = ext_docs
            return res
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

    def _direct_assistant_node(state: AgentState) -> dict[str, Any]:
        override = local_agents.get("direct_assistant")
        if override is not None and hasattr(override, "run"):
            return override.run(state)
        if callable(override):
            return override(state)
        return direct_assistant_node(state)

    def _task_execution_node(state: AgentState) -> dict[str, Any]:
        return _run_task_execution(state, active_registry)

    def _escalation_node(state: AgentState) -> dict[str, Any]:
        is_verif_failed = (
            state.get("processing_status") == "VERIFICATION_FAILED"
            or state.get("critic_action") == "verification_failed"
        )
        if is_verif_failed:
            logger.warning(
                "[Escalation] VERIFICATION_FAILED — xuất cảnh báo CHƯA ĐƯỢC XÁC MINH (không gọi Writer như kết quả đã duyệt)."
            )
            err_detail = state.get("error_message") or "Bộ kiểm chứng gặp lỗi kỹ thuật hoặc đầu ra không hợp lệ."
            draft_answer = state.get("answer", "")
            unverified_report = (
                "# CẢNH BÁO: NỘI DUNG CHƯA ĐƯỢC XÁC MINH (VERIFICATION_FAILED)\n\n"
                "> **Cảnh báo kiểm chứng:** Hệ thống không thể hoàn tất bước kiểm chứng bằng chứng cho câu trả lời này. "
                f"Nội dung bên dưới **chưa được xác minh** và không nên sử dụng như kết quả đã kiểm chứng.\n>\n"
                f"> **Chi tiết lỗi:** {err_detail}\n\n"
                f"{draft_answer}"
            )
            return {
                "confidence": "low",
                "processing_status": "VERIFICATION_FAILED",
                "error_stage": state.get("error_stage") or "summarizer_critic",
                "error_code": state.get("error_code") or "CRITIC_VERIFICATION_FAILED",
                "fact_check_label": "not_enough_info",
                "final_report": unverified_report,
                "reasoning_trace": [
                    {
                        "node": "EscalationNode",
                        "action": "escalate_verification_failed_unverified",
                        "processing_status": "VERIFICATION_FAILED",
                    }
                ],
            }

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
    workflow.add_node("direct_assistant_node", _direct_assistant_node)
    workflow.add_node("task_execution_node", _task_execution_node)
    workflow.add_node("template_response_node", template_response_node)
    workflow.add_node("scout_node", _scout_node)
    workflow.add_node("retrieval_node", _retrieval_node)
    workflow.add_node("reader_node", _reader_node)
    workflow.add_node("summarizer_node", _summarizer_node)
    workflow.add_node("fact_checker_node", _fact_checker_node)
    workflow.add_node("retry_loop_node", retry_loop_node)
    workflow.add_node("escalation_node", _escalation_node)
    workflow.add_node("writer_node", _writer_node)

    workflow.add_edge(START, "orchestrator_router")

    # Primary Intent Router Gate (before Scout & Retrieval)
    workflow.add_conditional_edges(
        "orchestrator_router",
        route_after_orchestrator,
        {
            "direct_assistant_node": "direct_assistant_node",
            "task_execution_node": "task_execution_node",
            "template_response_node": "template_response_node",
            "scout_node": "scout_node",
            "retrieval_node": "retrieval_node",
        },
    )

    workflow.add_edge("direct_assistant_node", END)
    workflow.add_edge("task_execution_node", END)
    workflow.add_edge("template_response_node", END)

    # Scout Gate: REJECT / CLARIFY / failure -> TemplateResponse; success -> Retrieval
    workflow.add_conditional_edges(
        "scout_node",
        route_after_scout,
        {
            "retrieval_node": "retrieval_node",
            "template_response_node": "template_response_node",
        },
    )

    workflow.add_edge("retrieval_node", "reader_node")
    workflow.add_edge("reader_node", "summarizer_node")

    # Tier 1 Gate: Summarizer Inner Critic -> (Fact-Checker, Direct Re-Query, or Fail-Closed Escalation)
    workflow.add_conditional_edges(
        "summarizer_node",
        route_after_summarizer,
        {
            "fact_checker_node": "fact_checker_node",
            "retry_loop_node": "retry_loop_node",
            "escalation_node": "escalation_node",
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

