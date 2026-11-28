import asyncio
from functools import partial
import hashlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable, Optional
from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, START, StateGraph

from agents.agent0_scout.node import ScoutAgent
from core.agent_decisions import decide_for_role
from core.agent_protocol import (
    AgentResult,
    AgentTask,
    compact_result_summary,
    make_agent_message,
    normalize_decision_dict,
)
from core.citation_audit import citation_location, get_citation_audit, write_with_citation_audit
from core.comparison_plan import (
    compare_documents, comparison_schedule, get_comparison_budget, incomplete_comparison, review_comparison,
    run_coro_sync,
)
from core.config import RunConfig, agent_inference_mode, get_run_settings, use_run_config
from core.coordination import (
    decide_handoff,
    evaluate_agent_handoff,
    fact_check_accepted,
    next_handoff_budgets,
)
from core.evidence_store import EvidenceStore
from core.figure_qa import (
    add_figure_evidence, attach_answer_figures, check_visual_answer,
    draft_visual_answer, figure_context, write_visual_answer,
)
from core.handoff_inputs import prepare_handoff_input
from core.provider_router import is_gemini_permitted
from core.research_dag import (
    build_research_dag_plan,
    execute_document_fanout_dag,
    execute_retrieval_dag,
    mark_dag_stage_status,
    pending_dag_investigation_queries,
    replan_research_dag_for_gaps,
    sync_dag_manifest_from_retrieval,
)
from core.research_tasks import ResearchTask, TaskStatus
from core.task_scheduler import TaskPolicy, TaskScheduler
from core.schemas import RouterPlanOutput
from core.state import AgentState
from core.supervisor import supervisor_planner
from core.supervisor_hub import route_from_supervisor_hub, supervisor_hub_step
from modes.registry import get_active_agents, get_agent_runners, normalize_mode, resolve_agent_modes
from utils.language import detect_corpus_language, localized_status_message, resolve_language_bundle
from utils.logger import logger


# ===================== NODE FUNCTIONS =====================

def orchestrator_router_node(
    state: AgentState,
    *,
    planner_llm_fn: Optional[Callable[[str], Any]] = None,
    execution_mode: Optional[str] = None,
) -> dict[str, Any]:
    """
    Agentic Supervisor Router Node (L3-L4):
    Phân tích query, phân loại intent + độ phức tạp (LOW/MEDIUM/HIGH),
    tách sub_queries đa bước (kết hợp LLM DAG Planner khi chế độ cho phép)
    và dựng kế hoạch ResearchPlan DAG đã kiểm chứng.
    """
    query = state.get("query", "")
    retry_count = state.get("retry_count", 0)
    evidence_gaps = state.get("evidence_gaps") or []
    chat_history = state.get("chat_history") or []

    coref = supervisor_planner.resolve_conversational_query(
        query,
        chat_history,
        existing_resolved=state.get("resolved_query") or state.get("query_with_context"),
    )
    resolved_query = coref["resolved_query"] or query
    inherited_entities = list(
        dict.fromkeys([*(state.get("inherited_entities") or []), *(coref.get("inherited_entities") or [])])
    )[:8]

    base_plan = supervisor_planner.plan_query(
        query=resolved_query,
        evidence_gaps=evidence_gaps,
        retry_count=retry_count,
    )
    if inherited_entities:
        base_plan.required_entities = list(dict.fromkeys([*base_plan.required_entities, *inherited_entities]))[:8]
        if coref.get("is_followup") and query.strip() and query.strip() not in base_plan.sub_queries:
            base_plan.sub_queries = list(dict.fromkeys([resolved_query, query.strip(), *base_plan.sub_queries]))[:4]
    plan = base_plan
    planning_source = "deterministic"

    custom_llm_fn = planner_llm_fn or state.get("planner_llm_fn")
    if callable(custom_llm_fn):
        refined = supervisor_planner.plan_query_with_llm(
            query=resolved_query,
            evidence_gaps=evidence_gaps,
            retry_count=retry_count,
            llm_fn=custom_llm_fn,
        )
        if refined is not base_plan and refined.reasoning.startswith("LLM-Refined"):
            plan = refined
            planning_source = "llm_refined"
    else:
        run_settings = get_run_settings()
        active_mode = (
            execution_mode
            or run_settings.get_inference_mode("retrieval")
            or run_settings.EXECUTION_MODE
            or "cpu_min"
        )
        should_refine_gemini = (
            retry_count == 0
            and base_plan.complexity in {"MEDIUM", "HIGH"}
            and is_gemini_permitted(active_mode, role="planner")
            and bool(getattr(run_settings, "GEMINI_API_KEY", ""))
        )
        if should_refine_gemini:
            from utils.llm_client import llm_client

            def _planner_llm(prompt: str) -> Any:
                with agent_inference_mode("gemini"):
                    return llm_client.generate_json(prompt, agent="retrieval")

            refined = supervisor_planner.plan_query_with_llm(
                query=resolved_query,
                evidence_gaps=evidence_gaps,
                retry_count=retry_count,
                llm_fn=_planner_llm,
            )
            if refined.reasoning.startswith("LLM-Refined"):
                plan = refined
                planning_source = "llm_refined"

        # Single-Planner consolidation: when Gemini didn't refine and Kaggle router tool is available
        # in gpu_finetune/hybrid mode, Supervisor invokes `/api/router` ONLY for HIGH-complexity queries
        # (multi-hop / multi-document comparison / synthesis) so simple/medium single-doc queries stay fast.
        if (
            planning_source == "deterministic"
            and retry_count == 0
            and base_plan.complexity == "HIGH"
            and active_mode in {"gpu_finetune", "hybrid"}
            and bool(getattr(run_settings, "REMOTE_INFERENCE_URL", ""))
        ):
            try:
                from utils.llm_client import llm_client

                router_res = llm_client.call_kaggle_specialist(
                    endpoint="api/router",
                    payload={"query": resolved_query, "adapter": "router", "max_new_tokens": 160},
                    adapter="router",
                    timeout=25.0,
                    allow_generate_fallback=False,
                )
                if isinstance(router_res, dict) and router_res:
                    lora_plan = RouterPlanOutput.from_dict(router_res, default_query=resolved_query)
                    merged_subs = list(dict.fromkeys([*lora_plan.sub_queries, *plan.sub_queries]))[:4]
                    if merged_subs:
                        plan.sub_queries = merged_subs
                    plan.alpha = lora_plan.alpha
                    plan.top_k = lora_plan.top_k
                    if lora_plan.intent:
                        plan.intent = lora_plan.intent
                    planning_source = "kaggle_router_tool"
            except Exception as exc:
                logger.warning("[Supervisor Planner] Kaggle /api/router tool fallback to deterministic plan: %s", exc)

    dag_round_index = int(state.get("dag_round_index") or 0)
    dag = build_research_dag_plan(
        plan,
        evidence_gaps=evidence_gaps,
        round_index=dag_round_index,
    )
    dag_manifest = dag.to_manifest()
    existing_schema = dict(state.get("document_schema") or {})
    existing_schema["research_dag"] = dag_manifest
    existing_schema["planner_source"] = planning_source

    logger.info(
        f"[Supervisor Planner] Query='{query[:70]}' | Intent={plan.intent} | "
        f"Complexity={plan.complexity} | SubQueries={len(plan.sub_queries)} | "
        f"Source={planning_source} | DAGTasks={dag_manifest['total_tasks']} | Retry={retry_count}"
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
        "planning_source": planning_source,
        "sub_queries": plan.sub_queries,
        "required_entities": plan.required_entities,
        "alpha": plan.alpha,
        "top_k": plan.top_k,
        "dag_task_count": dag_manifest["total_tasks"],
        "retry_count": retry_count,
    }
    lang_bundle = resolve_language_bundle(
        query,
        explicit_answer_language=state.get("answer_language"),
        units_or_docs=state.get("source_units") or state.get("retrieved_docs"),
        existing_source_language=state.get("source_language"),
    )
    doc_scope = list(state.get("document_scope") or [])
    if not doc_scope:
        raw_paths = list(dict.fromkeys(state.get("pdf_paths") or ([state["pdf_path"]] if state.get("pdf_path") else [])))
        for rp in raw_paths:
            if isinstance(rp, str) and rp.strip():
                doc_scope.append({"path": rp.strip()})

    return {
        "query_language": lang_bundle["query_language"],
        "answer_language": lang_bundle["answer_language"],
        "source_language": lang_bundle["source_language"],
        "resolved_query": resolved_query,
        "query_with_context": coref["query_with_context"],
        "inherited_entities": inherited_entities,
        "document_scope": doc_scope,
        "intent": plan.intent,
        "complexity": plan.complexity,
        "sub_queries": plan.sub_queries,
        "required_entities": plan.required_entities,
        "is_numeric_query": plan.is_numeric_query,
        "is_comparative_query": plan.is_comparative_query,
        "alpha": plan.alpha,
        "top_k": plan.top_k,
        "retry_count": retry_count,
        "dag_round_index": dag_round_index,
        "handoff_budgets": dict(state.get("handoff_budgets") or {}),
        "draft_revision_needed": bool(state.get("draft_revision_needed", False)),
        "document_schema": existing_schema,
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
            lang = writer_state.get("answer_language") or "vi"
            answer = (
                "Unable to compile a source-audited report."
                if lang == "en"
                else "Chưa thể xuất báo cáo đã đối chiếu với nguồn."
            )
            return {
                "final_report": answer,
                "reasoning_trace": [{"node": "EscalationNode", "action": "escalate_with_warning"}],
            }

    result = write_with_citation_audit(state_copy, write_or_warn)
    result["confidence"] = "low"
    banner = localized_status_message(
        "LOW_CONFIDENCE_BANNER",
        language=str(state.get("answer_language") or ""),
        query=str(state.get("query") or ""),
    )
    result["final_report"] = f"{banner}\n\n" + (result.get("final_report") or "")
    result["reasoning_trace"].append({
        "node": "EscalationNode", "action": "unresolved_review", "retry_count": state.get("retry_count", 0),
    })
    return result


def writer_node(state: AgentState) -> dict[str, Any]:
    res = write_with_citation_audit(state, _get_fallback_agents()["writer"].run)
    if "confidence" not in res:
        res["confidence"] = "high" if state.get("verified_claims") else "low"
    return res


# ===================== UNIFIED CONDITIONAL DISPATCH =====================

def _handoff_for(state: AgentState, role: str, key: str) -> dict[str, Any] | None:
    trace = state.get("reasoning_trace") or []
    if trace:
        last = trace[-1]
        if last.get("action") == "agent_handoff" and last.get(key) == role:
            return last
    return None


def dispatch_next_node(
    completed_node: str | None,
    state: AgentState,
    *,
    comparison_plan: bool = False,
    supervisor_mode: str | None = None,
) -> str:
    """
    Single unified dispatch policy shared by fresh LangGraph runs, retry loops, and checkpoint resume.
    Routes every completed node across all 6 specialist roles + Supervisor consistently in both
    `pipeline` and `hub` orchestration modes.
    """
    node = (completed_node or "").strip()
    effective_sup_mode = str(supervisor_mode or state.get("supervisor_mode") or "pipeline").strip().lower()
    if effective_sup_mode == "hub":
        if not node:
            return "orchestrator_router"
        if node == "supervisor_node":
            return route_from_supervisor_hub(state)
        if node in {"escalation_node", END, "__end__"}:
            return END
        if node in {
            "orchestrator_router",
            "scout_node",
            "retrieval_node",
            "reader_node",
            "summarizer_node",
            "fact_checker_node",
            "writer_node",
            "comparison_review_node",
        }:
            return "supervisor_node"
        return "orchestrator_router"

    if not node:
        return "orchestrator_router"
    if node == "orchestrator_router":
        request = _handoff_for(state, "supervisor", "sender")
        return f"{request['recipient']}_node" if request else "scout_node"
    if node == "scout_node":
        trace = state.get("reasoning_trace") or []
        if trace and trace[-1].get("action") in {
            "discovery_clarify", "discovery_rejected", "discovery_failed", "download_failed", "waiting_for_document",
        }:
            return END
        request = _handoff_for(state, "scout", "sender")
        return f"{request['recipient']}_node" if request else "retrieval_node"
    if node == "retrieval_node":
        request = _handoff_for(state, "retrieval", "sender")
        if request:
            return f"{request['recipient']}_node"
        if comparison_plan:
            schedule = comparison_schedule(state.get("reasoning_trace") or [])
            if schedule is not None:
                return "comparison_review_node" if schedule["status"] == "COMPLETED" else END
        return "reader_node"
    if node == "reader_node":
        if comparison_plan:
            schedule = comparison_schedule(state.get("reasoning_trace") or [])
            if schedule is not None:
                return "comparison_review_node" if schedule["status"] == "COMPLETED" else END
        request = _handoff_for(state, "reader", "sender")
        return f"{request['recipient']}_node" if request else "summarizer_node"
    if node == "summarizer_node":
        request = _handoff_for(state, "summarizer", "sender")
        return f"{request['recipient']}_node" if request else "fact_checker_node"
    if node == "fact_checker_node":
        request = _handoff_for(state, "fact_checker", "sender")
        if request:
            return f"{request['recipient']}_node"
        if figure_context(state).get("explicit") and not state.get("rejected_claims"):
            return "writer_node"
        return "writer_node" if fact_check_accepted(state) else "escalation_node"
    if node == "comparison_review_node":
        trace = state.get("reasoning_trace") or []
        last = trace[-1] if trace else {}
        if last.get("action") != "agent_handoff":
            return END
        return "comparison_review_node" if last.get("recipient") == "summarizer" else f"{last['recipient']}_node"
    if node == "writer_node":
        request = _handoff_for(state, "writer", "sender")
        return f"{request['recipient']}_node" if request else END
    if node in {"escalation_node", END, "__end__"}:
        return END
    return "orchestrator_router"


def route_after_router(state: AgentState) -> str:
    return dispatch_next_node("orchestrator_router", state)


def route_after_scout(state: AgentState) -> str:
    return dispatch_next_node("scout_node", state)


def route_after_retrieval(state: AgentState) -> str:
    return dispatch_next_node("retrieval_node", state)


def route_after_reader(state: AgentState) -> str:
    return dispatch_next_node("reader_node", state)


def route_after_summarizer(state: AgentState) -> str:
    return dispatch_next_node("summarizer_node", state)


def route_after_fact_check(state: AgentState) -> str:
    return dispatch_next_node("fact_checker_node", state)


def route_after_writer(state: AgentState) -> str:
    return dispatch_next_node("writer_node", state)


def resolve_next_node_after_checkpoint(
    completed_node: str | None,
    state: AgentState,
    *,
    comparison_plan: bool = False,
    supervisor_mode: str | None = None,
) -> str:
    """Resolve the exact successor node to execute when resuming after `completed_node`."""
    return dispatch_next_node(
        completed_node,
        state,
        comparison_plan=comparison_plan,
        supervisor_mode=supervisor_mode,
    )


_agents: dict[str, Any] = {}


def build_research_rag_graph(
    mode: Optional[str] = None, *, run_config: RunConfig | None = None, auto_scout: bool = True,
    allow_scout_handoff: Optional[bool] = None,
    parallel_retrieval: bool = False, comparison_plan: bool = False,
    planner_llm_fn: Optional[Callable[[str], Any]] = None,
    arbiter_llm_fn: Optional[Callable[[str], Any]] = None,
    start_at_node: Optional[str] = None,
    enable_reader_handoff: bool = True,
    supervisor_mode: Optional[str] = None,
    disabled_agent_handoffs: Optional[list[str]] = None,
):
    """Build a run-local team supporting both `hub` (default Hub-and-Spoke) and `pipeline` (legacy baseline) modes."""
    run_config = run_config or RunConfig.capture()
    effective_supervisor_mode = str(
        supervisor_mode
        if supervisor_mode is not None
        else (getattr(run_config, "SUPERVISOR_MODE", "hub") or "hub")
    ).strip().lower()
    hub_mode = effective_supervisor_mode != "pipeline"
    graph_disabled_handoffs = [
        str(r).strip().lower()
        for r in (disabled_agent_handoffs or [])
        if isinstance(r, str) and r.strip()
    ]
    with use_run_config(run_config):
        run_agents = get_agent_runners(mode)
        reader_mode = normalize_mode(mode or run_config.get_agent_mode("reader") or run_config.EXECUTION_MODE)
        scout = ScoutAgent(auto_scout=auto_scout)
        scout_handoff_allowed = bool(auto_scout) if allow_scout_handoff is None else bool(allow_scout_handoff)
        scout_mode = normalize_mode(mode or run_config.EXECUTION_MODE)
        agent_modes = resolve_agent_modes(run_config, mode)
        pilot_enabled = parallel_retrieval and agent_modes["retrieval"] == "cpu_min"
        is_cpu_min_team = set(agent_modes.values()) == {"cpu_min"}
        comparison_enabled = bool(comparison_plan)
    if comparison_enabled:
        from modes.cpu_min.reader import read_comparison_document

    def _read_comparison_generic(query: str, path: str, ranking: list[dict[str, Any]]) -> dict[str, Any]:
        from core.citation_audit import evidence_quote
        from core.state import compute_context_hash
        ranked_locs = {citation_location(chunk) for chunk in ranking if citation_location(chunk)}
        ranked_ids = {chunk.get("id") for chunk in ranking if chunk.get("id")}
        result = run_agents["reader"]({
            "query": query, "pdf_path": path, "pdf_paths": [path], "retrieved_docs": ranking,
        })
        units = []
        for unit in result.get("source_units") or []:
            location = citation_location(unit)
            text = evidence_quote(unit)
            if not location or not text.strip() or location not in ranked_locs:
                continue
            if unit.get("id") not in ranked_ids and not str(unit.get("id") or "").startswith("table_"):
                continue
            unit_hash = unit.get("context_hash") or compute_context_hash(text, *location)
            units.append({**unit, "context_hash": unit_hash, "source_path": path})
        if not units:
            raise ValueError("Reader returned no traceable evidence from the selected document.")
        return {**result, "source_units": units}

    def _accepts_kwarg(fn: Callable[..., Any], kwarg: str) -> bool:
        try:
            params = inspect.signature(fn).parameters
            return kwarg in params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            return False

    def run_comparison_step(role, inputs, plan, previous):
        use_cpu_min_reader = is_cpu_min_team and _accepts_kwarg(run_agents["reader"], "allow_pdf_fallback")
        reader = (
            partial(read_comparison_document, run_agents["reader"], inputs.get("query", ""))
            if use_cpu_min_reader
            else partial(_read_comparison_generic, inputs.get("query", ""))
        )
        top_k = max(len(plan.paths), min(int(inputs.get("top_k") or 8), 10))
        if role == "retrieval":
            if is_cpu_min_team and _accepts_kwarg(run_agents[role], "comparison"):
                update = run_agents[role](inputs, comparison=plan, comparison_reader=reader)
            else:
                per_doc_k = max(2, min(top_k, 8))

                def rank_single_doc(path: str) -> list[dict[str, Any]]:
                    res = run_agents["retrieval"]({
                        **inputs,
                        "pdf_path": path,
                        "pdf_paths": [path],
                        "top_k": per_doc_k,
                    })
                    return list(res.get("retrieved_docs") or [])

                update = run_coro_sync(compare_documents(
                    plan, rank_single_doc, reader, top_k,
                    retry_count=inputs.get("retry_count", 0), read_only=False,
                ))
        else:
            source_by_path = dict(zip(plan.paths, previous["required_sources"], strict=True))

            def reuse_ranking(path):
                return [chunk for chunk in inputs.get("retrieved_docs") or [] if chunk.get("doc_id") == source_by_path[path]]

            update = run_coro_sync(compare_documents(
                plan, reuse_ranking, reader, top_k,
                retry_count=inputs.get("retry_count", 0), read_only=True,
            ))
        if comparison_schedule(update["reasoning_trace"])["status"] == "COMPLETED":
            budget = get_comparison_budget()
            if budget:
                budget.remaining()
            update = add_figure_evidence(inputs, update, reader_mode)
        return update

    def _path_identity(raw_path: str) -> tuple[str, str | None]:
        try:
            resolved = os.path.normcase(str(Path(raw_path).resolve()))
        except Exception:
            resolved = os.path.normcase(os.path.abspath(str(raw_path)))
        digest: str | None = None
        try:
            if os.path.isfile(raw_path):
                h = hashlib.sha256()
                with open(raw_path, "rb") as fh:
                    for block in iter(lambda: fh.read(65536), b""):
                        h.update(block)
                digest = h.hexdigest()
        except OSError:
            digest = None
        return resolved, digest

    def _active_task_for(state: AgentState, role: str) -> AgentTask | None:
        pending = state.get("pending_tasks") or []
        if pending and isinstance(pending[0], dict) and pending[0].get("role") == role:
            return AgentTask.from_dict(pending[0])
        return None

    def _attach_protocol_envelope(
        role: str,
        state: AgentState,
        update: dict[str, Any],
        task: AgentTask | None,
        *,
        explicit_decision: Any = None,
    ) -> dict[str, Any]:
        """Wrap specialist update with mandatory `AgentDecision`, compact `AgentResult`, and blackboard message."""
        merged_for_decide = {
            **state,
            "allow_scout_handoff": scout_handoff_allowed,
            "enable_reader_handoff": False if (not enable_reader_handoff or state.get("enable_reader_handoff") is False) else True,
        }
        decision = (
            explicit_decision
            if explicit_decision is not None
            else decide_for_role(role, merged_for_decide, update, task.to_dict() if task else None)
        )
        round_idx = int(update.get("retry_count") if update.get("retry_count") is not None else (state.get("retry_count") or 0))
        dec_dict = normalize_decision_dict(
            role,
            decision,
            round_index=round_idx,
            task_id=task.task_id if task else None,
        )
        summary = compact_result_summary(role, state, update)
        raw_tools = update.get("tools_used")
        tools_used = tuple(str(t) for t in raw_tools if isinstance(t, str) and t.strip()) if isinstance(raw_tools, (list, tuple)) else ()
        result_obj = AgentResult(
            role=role,
            status=str(dec_dict.get("status") or "completed"),
            decision=dec_dict,
            task_id=task.task_id if task else None,
            summary=summary,
            tools_used=tools_used,
            round_index=round_idx,
        )
        msg_kind = "decision_proposed" if dec_dict.get("action") not in {"proceed", ""} else "result_reported"
        msg = make_agent_message(
            sender=role,
            recipient="supervisor" if hub_mode else str(dec_dict.get("target_agent") or "supervisor"),
            kind=msg_kind,
            action=str(dec_dict.get("action") or "proceed"),
            reason=str(dec_dict.get("reason") or ""),
            task_id=task.task_id if task else None,
            focus=(dec_dict.get("parameters") or {}).get("focus") if isinstance(dec_dict.get("parameters"), dict) else (),
            summary=summary,
            round_index=round_idx,
        )
        existing_msgs = list(update.get("agent_messages") or [])
        existing_msgs.append(msg)
        wrapped = {
            **update,
            "last_result": result_obj.to_dict(),
            "agent_messages": existing_msgs,
        }
        if hub_mode:
            wrapped["pending_tasks"] = []
        return wrapped

    def _run_scout(state: AgentState) -> dict[str, Any]:
        task = _active_task_for(state, "scout") if hub_mode else None
        request = (task.to_handoff_request() if task else None) or _handoff_for(state, "scout", "recipient")
        if not request:
            base_out = scout.run(state)
            return _attach_protocol_envelope("scout", state, base_out, task)
        existing_paths = list(dict.fromkeys(
            state.get("pdf_paths") or ([state["pdf_path"]] if state.get("pdf_path") else [])
        ))
        focus = [item for item in request.get("focus") or [] if isinstance(item, str) and item.strip()]
        discovery_query = focus[0] if focus else state.get("query", "")
        discovery_state = {**state, "query": discovery_query, "pdf_path": None, "pdf_paths": []}
        active_scout = scout if (scout.auto_scout or not scout_handoff_allowed) else ScoutAgent(auto_scout=True)
        update = active_scout.run(discovery_state)
        receipt = {
            "action": "agent_handoff_received",
            "agent": "scout",
            "sender": request["sender"],
            "recipient": "scout",
            "reason": request["reason"],
            "signature": request.get("signature"),
            "retry_count": request.get("retry_count", int(state.get("retry_count") or 0)),
        }
        if request.get("decision_id"):
            receipt["decision_id"] = request["decision_id"]
        if request.get("task_id"):
            receipt["task_id"] = request["task_id"]
        candidate_paths = list(dict.fromkeys(
            p for p in (update.get("pdf_paths") or ([update["pdf_path"]] if update.get("pdf_path") else []))
            if isinstance(p, str) and p.strip()
        ))
        seen_resolved: set[str] = set()
        seen_hashes: set[str] = set()
        for existing in existing_paths:
            res_path, file_hash = _path_identity(existing)
            seen_resolved.add(res_path)
            if file_hash:
                seen_hashes.add(file_hash)
        truly_new_paths: list[str] = []
        for candidate in candidate_paths:
            res_path, file_hash = _path_identity(candidate)
            if res_path in seen_resolved or (file_hash and file_hash in seen_hashes):
                continue
            seen_resolved.add(res_path)
            if file_hash:
                seen_hashes.add(file_hash)
            truly_new_paths.append(candidate)

        merged_paths = [*existing_paths, *truly_new_paths]
        outcome_event = {
            "action": "handoff_outcome_recorded",
            "agent": "scout",
            "sender": request["sender"],
            "recipient": "scout",
            "signature": request.get("signature"),
            "decision_id": request.get("decision_id"),
            "task_id": request.get("task_id"),
            "outcome_status": "discovered_new_documents" if truly_new_paths else "no_new_documents",
            "new_documents_count": len(truly_new_paths),
        }
        trace = [
            receipt,
            *(update.get("reasoning_trace") or []),
            outcome_event,
        ]
        if truly_new_paths and merged_paths:
            out = {
                **update,
                "pdf_path": merged_paths[0],
                "pdf_paths": merged_paths,
                "no_new_evidence": False,
                "agent_decision": None,
                "reasoning_trace": trace,
            }
            return _attach_protocol_envelope("scout", state, out, task)
        if not existing_paths and update.get("answer_status"):
            out = {
                **update,
                "pdf_path": None,
                "pdf_paths": [],
                "no_new_evidence": True,
                "agent_decision": None,
                "reasoning_trace": [receipt, *(update.get("reasoning_trace") or [])],
            }
            return _attach_protocol_envelope("scout", state, out, task)
        # If discovery found no genuinely new PDF, preserve existing paths and mark no_new_evidence=True
        out = {
            "pdf_path": existing_paths[0] if existing_paths else None,
            "pdf_paths": existing_paths,
            "no_new_evidence": True,
            "agent_decision": None,
            "reasoning_trace": trace,
        }
        return _attach_protocol_envelope("scout", state, out, task)

    def run_specialist(role: str, state: AgentState, *, inline_handoff: bool | None = None) -> dict[str, Any]:
        use_inline_handoff = (not hub_mode) if inline_handoff is None else bool(inline_handoff)
        task = _active_task_for(state, role) if hub_mode else None
        request = (task.to_handoff_request() if task else None) or _handoff_for(state, role, "recipient")
        inputs = prepare_handoff_input(state, request) if request else state
        if role == "reader" and (not enable_reader_handoff or state.get("enable_reader_handoff") is False):
            inputs = {**inputs, "enable_reader_handoff": False}
        existing_schema = dict(inputs.get("document_schema") or state.get("document_schema") or {})
        dag_manifest = existing_schema.get("research_dag") if isinstance(existing_schema.get("research_dag"), dict) else None
        current_dag_round = int(
            state.get("dag_round_index")
            if state.get("dag_round_index") is not None
            else (dag_manifest.get("round_index", 0) if isinstance(dag_manifest, dict) else 0)
        )
        gap_queries = (
            pending_dag_investigation_queries(dag_manifest)
            if role == "retrieval" and current_dag_round > 0
            else []
        )
        if gap_queries:
            merged_sub_queries = list(dict.fromkeys([
                *gap_queries,
                *(q.strip() for q in (inputs.get("sub_queries") or []) if isinstance(q, str) and q.strip()),
            ]))
            inputs = {**inputs, "sub_queries": merged_sub_queries}

        visual = figure_context(inputs).get("explicit", False)
        previous = comparison_schedule(state.get("reasoning_trace") or []) if comparison_enabled and role == "reader" else None
        comparison = (
            supervisor_planner.plan_document_comparison(inputs)
            if comparison_enabled and (role == "retrieval" or (previous and request)) else None
        )
        dag_trace: dict[str, Any] | None = None
        if visual and role == "summarizer":
            update = draft_visual_answer(inputs, mode=agent_modes["summarizer"])
        elif visual and role == "fact_checker":
            update = check_visual_answer(inputs)
        elif comparison is not None:
            update = run_comparison_step(role, inputs, comparison, previous)
        elif pilot_enabled and role == "retrieval":
            update = run_agents[role](inputs, parallel_search=True)
        elif (parallel_retrieval or bool(gap_queries)) and role == "retrieval":
            sub_queries = list(dict.fromkeys(
                q.strip() for q in (inputs.get("sub_queries") or [])
                if isinstance(q, str) and q.strip()
            ))
            selected_pdfs = list(dict.fromkeys(
                p for p in (inputs.get("pdf_paths") or ([inputs["pdf_path"]] if inputs.get("pdf_path") else []))
                if isinstance(p, str) and p.strip()
            ))
            if len(sub_queries) >= 2:
                def _run_branch(branch_query: str) -> dict[str, Any]:
                    branch_state = {
                        **inputs,
                        "query": branch_query,
                        "sub_queries": [branch_query],
                    }
                    return run_agents["retrieval"](branch_state)

                prev_seen = inputs.get("seen_chunks") or state.get("seen_chunks") or []
                try:
                    dag_update, dag_trace = execute_retrieval_dag(
                        sub_queries,
                        _run_branch,
                        top_k=int(inputs.get("top_k") or 10),
                        round_index=current_dag_round,
                        dag_manifest=dag_manifest,
                        previous_seen_chunks=prev_seen,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[ResearchDAG] Parallel retrieval DAG fell back to single-pass retrieval: {exc}"
                    )
                    dag_update, dag_trace = {}, {
                        "agent": "task_scheduler",
                        "action": "research_dag_fallback",
                        "status": "fallback",
                        "reason": str(exc),
                    }
                if dag_trace.get("status") != "fallback" and dag_update:
                    update = {
                        **dag_update,
                        "reasoning_trace": [dag_trace],
                    }
                else:
                    fallback_update = run_agents[role](inputs)
                    fallback_trace = list(fallback_update.get("reasoning_trace") or [])
                    if dag_trace:
                        fallback_trace.insert(0, dag_trace)
                    update = {**fallback_update, "reasoning_trace": fallback_trace}
            elif parallel_retrieval and len(selected_pdfs) >= 2:
                def _run_doc_branch(doc_path: str) -> dict[str, Any]:
                    branch_state = {
                        **inputs,
                        "pdf_path": doc_path,
                        "pdf_paths": [doc_path],
                    }
                    return run_agents["retrieval"](branch_state)

                prev_seen = inputs.get("seen_chunks") or state.get("seen_chunks") or []
                try:
                    fanout_update, fanout_trace = execute_document_fanout_dag(
                        selected_pdfs,
                        _run_doc_branch,
                        top_k=int(inputs.get("top_k") or 10),
                        round_index=current_dag_round,
                        previous_seen_chunks=prev_seen,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[ResearchDAG] Per-document fan-out fell back to single-pass retrieval: {exc}"
                    )
                    fanout_update, fanout_trace = {}, {
                        "agent": "task_scheduler",
                        "action": "document_fanout_fallback",
                        "status": "fallback",
                        "reason": str(exc),
                    }
                if fanout_trace.get("status") != "fallback" and fanout_update:
                    dag_trace = fanout_trace
                    update = {
                        **fanout_update,
                        "reasoning_trace": [fanout_trace],
                    }
                else:
                    fallback_update = run_agents[role](inputs)
                    fallback_trace = list(fallback_update.get("reasoning_trace") or [])
                    if fanout_trace:
                        fallback_trace.insert(0, fanout_trace)
                    update = {**fallback_update, "reasoning_trace": fallback_trace}
            else:
                update = run_agents[role](inputs)
        else:
            update = run_agents[role](inputs)

        if role == "retrieval":
            # Track cumulative seen_chunks & truthful evidence growth across retries for ALL retrieval modes
            prev_seen_list = list(state.get("seen_chunks") or [])
            if "retrieved_docs" in update and "no_new_evidence" not in update:
                new_docs = [d for d in (update.get("retrieved_docs") or []) if isinstance(d, dict)]
                prev_ids = {
                    (d.get("id") or d.get("chunk_id") or (d.get("doc_id"), d.get("page_no"), str(d.get("text") or "")[:80]))
                    for d in prev_seen_list
                    if isinstance(d, dict)
                }
                fresh_docs = [
                    d for d in new_docs
                    if (d.get("id") or d.get("chunk_id") or (d.get("doc_id"), d.get("page_no"), str(d.get("text") or "")[:80])) not in prev_ids
                ]
                if "seen_chunks" not in update:
                    update = {**update, "seen_chunks": [*prev_seen_list, *fresh_docs]}
                if state.get("retry_count", 0) > 0 and prev_seen_list and not fresh_docs:
                    update = {**update, "no_new_evidence": True}

        if role == "retrieval" and dag_manifest is not None:
            schema = dict(update.get("document_schema") or existing_schema)
            sync_event = dag_trace if (isinstance(dag_trace, dict) and dag_trace.get("action") == "research_dag_executed") else None
            if isinstance(dag_trace, dict) and dag_trace.get("action") == "document_fanout_dag_executed":
                schema["document_fanout"] = dag_trace
            schema["research_dag"] = sync_dag_manifest_from_retrieval(
                dag_manifest,
                sync_event,
                retrieved_count=len(update.get("retrieved_docs") or []),
                round_index=current_dag_round,
            )
            update = {**update, "document_schema": schema}

        if role == "reader" and comparison is None:
            update = add_figure_evidence(inputs, update, reader_mode)
        if role == "reader" and comparison is None and state.get("retry_count") and state.get("source_units"):
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
            receipt = {
                "action": "agent_handoff_received",
                "agent": role,
                "sender": request["sender"],
                "recipient": role,
                "reason": request["reason"],
                "signature": request.get("signature"),
                "retry_count": request.get("retry_count", int(state.get("retry_count") or 0)),
            }
            if request.get("decision_id"):
                receipt["decision_id"] = request["decision_id"]
            if request.get("task_id"):
                receipt["task_id"] = request["task_id"]
            trace.insert(0, receipt)
            outcome_status = "completed"
            if role == "retrieval":
                outcome_status = "no_new_evidence" if update.get("no_new_evidence") else "evidence_retrieved"
            elif role == "reader":
                outcome_status = "units_extracted" if update.get("source_units") else "no_units"
            elif role == "summarizer":
                outcome_status = "draft_revised" if (update.get("answer") or "").strip() else "draft_empty"
            elif role == "fact_checker":
                outcome_status = str(update.get("fact_check_label") or "evaluated")
            trace.append({
                "action": "handoff_outcome_recorded",
                "agent": role,
                "sender": request["sender"],
                "recipient": role,
                "signature": request["signature"],
                "decision_id": request.get("decision_id"),
                "task_id": request.get("task_id"),
                "outcome_status": outcome_status,
            })

        if role == "reader":
            schema = {
                **(state.get("document_schema") or {}),
                **(update.get("document_schema") or {}),
            }
            schema["evidence_store"] = EvidenceStore.from_state({**state, **update}).to_dict()
            src_lang = detect_corpus_language(
                update.get("source_units") or state.get("source_units") or state.get("retrieved_docs"),
                default=state.get("source_language") or "en",
            )
            update = {**update, "document_schema": schema, "source_language": src_lang}

        should_reset_draft = role == "summarizer" and (
            bool(state.get("draft_revision_needed"))
            or (request is not None and request.get("sender") in {"summarizer", "fact_checker", "writer"})
            or bool(state.get("verified_claims") or state.get("rejected_claims"))
        )
        if should_reset_draft:
            # Event-driven draft reset: clear stale verification/claims only when revising an earlier draft,
            # never falsely on a pre-draft reader->retrieval handoff.
            update = {
                "answer": "", "claims": [], "evidence_spans": [],
                "critic_action": "approve", "critic_feedback": {},
                "verified_claims": [], "rejected_claims": [], "evidence_gaps": [],
                "short_answer": None, "answer_status": None, "fact_check_label": "not_enough_info",
                "statistical_audit": None, "code_verification": None,
                "draft_revision_needed": False,
                **update,
            }
        elif role == "summarizer":
            update = {**update, "draft_revision_needed": False}

        if role == "summarizer":
            schema = dict(update.get("document_schema") or state.get("document_schema") or {})
            schema["evidence_store"] = EvidenceStore.from_state({**state, **update}).to_dict()
            if isinstance(schema.get("research_dag"), dict):
                has_answer = bool((update.get("answer") or state.get("answer") or "").strip())
                schema["research_dag"] = mark_dag_stage_status(
                    schema.get("research_dag"),
                    "synthesize",
                    TaskStatus.COMPLETED if has_answer else TaskStatus.FAILED,
                    round_index=current_dag_round,
                    metadata_update={"claims_count": len(update.get("claims") or [])},
                )
            update = {**update, "document_schema": schema}

        merged_disabled = list(dict.fromkeys([
            *graph_disabled_handoffs,
            *(str(r).strip().lower() for r in (state.get("disabled_agent_handoffs") or []) if isinstance(r, str) and r.strip()),
        ]))
        reviewed_snapshot = {
            **state,
            **update,
            "agent_decision": update.get("agent_decision"),
            "allow_scout_handoff": scout_handoff_allowed,
            "enable_reader_handoff": False if (not enable_reader_handoff or state.get("enable_reader_handoff") is False) else True,
            "disabled_agent_handoffs": merged_disabled,
            "reasoning_trace": [*(state.get("reasoning_trace") or []), *trace],
        }

        if role == "fact_checker" and not visual:
            schema = dict(update.get("document_schema") or state.get("document_schema") or {})
            store = EvidenceStore.from_state(reviewed_snapshot)
            schema["evidence_store"] = store.to_dict()
            if isinstance(schema.get("research_dag"), dict):
                if fact_check_accepted(reviewed_snapshot):
                    verify_outcome = "PASSED"
                else:
                    label = str(reviewed_snapshot.get("fact_check_label") or "").lower()
                    rejected = reviewed_snapshot.get("rejected_claims") or []
                    has_contradiction = label in {"contradict", "contradicted", "refuted"} or any(
                        str(item.get("status", "")).lower() in {"contradict", "contradicted", "refuted"}
                        for item in rejected if isinstance(item, dict)
                    )
                    verify_outcome = "CONTRADICTED" if has_contradiction else "NEEDS_MORE_EVIDENCE"
                schema["research_dag"] = mark_dag_stage_status(
                    schema.get("research_dag"),
                    "verify",
                    TaskStatus.COMPLETED,
                    round_index=current_dag_round,
                    review_outcome=verify_outcome,
                    metadata_update={
                        "fact_check_label": reviewed_snapshot.get("fact_check_label"),
                        "verified_count": len(reviewed_snapshot.get("verified_claims") or []),
                        "rejected_count": len(reviewed_snapshot.get("rejected_claims") or []),
                    },
                )
            update["document_schema"] = schema

        # Every specialist computes its own mandatory `AgentDecision` via `decide_for_role`.
        role_decision = (
            decide_for_role(role, reviewed_snapshot, update, task.to_dict() if task else None)
            if not visual
            else None
        )

        if use_inline_handoff:
            should_evaluate_decision = not visual and (
                role in {"summarizer", "fact_checker"}
                or (role in {"reader", "retrieval"} and (update.get("agent_decision") is not None or (role_decision is not None and role_decision.action != "proceed")))
            )
            if should_evaluate_decision:
                reviewed_inline = {
                    **reviewed_snapshot,
                    "agent_decision": (
                        update.get("agent_decision")
                        if update.get("agent_decision") is not None
                        else (role_decision.to_dict() if role_decision is not None and role_decision.action != "proceed" else None)
                    ),
                }
                decision, audit_event = evaluate_agent_handoff(role, reviewed_inline)
                if audit_event is not None:
                    trace.append(audit_event)
                if role == "fact_checker":
                    schema = dict(update.get("document_schema") or state.get("document_schema") or {})
                    gaps = reviewed_inline.get("evidence_gaps") or []
                    if gaps and decision is not None and decision.recipient in {"retrieval", "scout", "reader"}:
                        replanned = replan_research_dag_for_gaps(schema.get("research_dag"), gaps)
                        schema["research_dag"] = replanned
                        update["dag_round_index"] = int(replanned.get("round_index", current_dag_round))
                        if replanned.get("added_gap_tasks", 0) > 0:
                            trace.append({
                                "agent": "task_scheduler",
                                "action": "research_dag_replanned",
                                "round_index": replanned["round_index"],
                                "added_gap_tasks": replanned["added_gap_tasks"],
                                "total_tasks": replanned["total_tasks"],
                            })
                    update["document_schema"] = schema
                if decision is not None:
                    attempt = state.get("retry_count", 0) + 1
                    budget_cat = str((audit_event or {}).get("budget_category") or "revise")
                    updated_budgets = next_handoff_budgets(
                        state.get("handoff_budgets"),
                        budget_cat,
                        fallback_total=int(state.get("retry_count") or 0),
                    )
                    handoff_event = {
                        **decision.as_trace(),
                        "agent": role,
                        "retry_count": attempt,
                        "budget_category": budget_cat,
                    }
                    if audit_event and audit_event.get("decision_id"):
                        handoff_event["decision_id"] = audit_event["decision_id"]
                    trace.append(handoff_event)
                    update["retry_count"] = attempt
                    update["handoff_budgets"] = updated_budgets
                    if role in {"summarizer", "fact_checker", "writer"}:
                        update["draft_revision_needed"] = True

        update["agent_decision"] = None
        update["reasoning_trace"] = trace
        return _attach_protocol_envelope(role, state, update, task, explicit_decision=role_decision)

    def _mark_write_stage(state: AgentState, result: dict[str, Any]) -> dict[str, Any]:
        schema = dict(result.get("document_schema") or state.get("document_schema") or {})
        if isinstance(schema.get("research_dag"), dict):
            write_round = int(
                result.get("dag_round_index")
                if result.get("dag_round_index") is not None
                else (
                    state.get("dag_round_index")
                    if state.get("dag_round_index") is not None
                    else schema["research_dag"].get("round_index", 0)
                )
            )
            schema["research_dag"] = mark_dag_stage_status(
                schema.get("research_dag"),
                "write",
                TaskStatus.COMPLETED,
                round_index=write_round,
                metadata_update={"confidence": result.get("confidence")},
            )
            result = {**result, "document_schema": schema}
        return result

    def _writer_node(state: AgentState, *, inline_handoff: bool | None = None) -> dict[str, Any]:
        use_inline_handoff = (not hub_mode) if inline_handoff is None else bool(inline_handoff)
        task = _active_task_for(state, "writer") if hub_mode else None
        writer = write_visual_answer if figure_context(state).get("explicit") else run_agents["writer"]
        result = write_with_citation_audit(state, writer)
        comparison = comparison_schedule(state.get("reasoning_trace") or []) if comparison_enabled else None
        if comparison:
            references = get_citation_audit(result["reasoning_trace"]).get("references", [])
            cited = {location[0] for ref in references if (location := citation_location(ref))}
            if not cited.issuperset(comparison["required_sources"]):
                inc = incomplete_comparison("missing_audited_source")
                return _attach_protocol_envelope("writer", state, inc, task)
        if state.get("critic_action") == "lower_confidence":
            result["confidence"] = "low"
        else:
            result.setdefault("confidence", "high" if state.get("verified_claims") else "low")

        writer_decision = decide_for_role("writer", state, result, task.to_dict() if task else None)
        if use_inline_handoff:
            raw_proposal = (
                result.get("agent_decision")
                if result.get("agent_decision") is not None
                else (writer_decision.to_dict() if writer_decision.action != "proceed" else None)
            )
            if raw_proposal is not None:
                w_trace = list(result.get("reasoning_trace") or [])
                reviewed_w = {
                    **state,
                    **result,
                    "agent_decision": raw_proposal,
                    "allow_scout_handoff": scout_handoff_allowed,
                    "disabled_agent_handoffs": list(dict.fromkeys([
                        *graph_disabled_handoffs,
                        *(str(r).strip().lower() for r in (state.get("disabled_agent_handoffs") or []) if isinstance(r, str) and r.strip()),
                    ])),
                    "reasoning_trace": [*(state.get("reasoning_trace") or []), *w_trace],
                }
                w_decision, w_audit = evaluate_agent_handoff("writer", reviewed_w)
                if w_audit is not None:
                    w_trace.append(w_audit)
                if w_decision is not None:
                    attempt = state.get("retry_count", 0) + 1
                    w_cat = str((w_audit or {}).get("budget_category") or "verify")
                    w_budgets = next_handoff_budgets(
                        state.get("handoff_budgets"),
                        w_cat,
                        fallback_total=int(state.get("retry_count") or 0),
                    )
                    w_handoff = {
                        **w_decision.as_trace(),
                        "agent": "writer",
                        "retry_count": attempt,
                        "budget_category": w_cat,
                    }
                    if w_audit and w_audit.get("decision_id"):
                        w_handoff["decision_id"] = w_audit["decision_id"]
                    w_trace.append(w_handoff)
                    result = {
                        **result,
                        "retry_count": attempt,
                        "handoff_budgets": w_budgets,
                        "draft_revision_needed": True,
                        "agent_decision": None,
                        "reasoning_trace": w_trace,
                    }
                else:
                    result = {**result, "agent_decision": None, "reasoning_trace": w_trace}
            else:
                result = {**result, "agent_decision": None}
        else:
            result = {**result, "agent_decision": None}
        finalized = _mark_write_stage(state, attach_answer_figures(state, result))
        return _attach_protocol_envelope("writer", state, finalized, task, explicit_decision=writer_decision)

    def _escalation(state: AgentState) -> dict[str, Any]:
        out = _mark_write_stage(
            state,
            attach_answer_figures(state, _write_escalated_report(state, run_agents["writer"])),
        )
        if hub_mode:
            out = {**out, "pending_tasks": []}
        return out

    def _comparison_review(state: AgentState) -> dict[str, Any]:
        task = _active_task_for(state, "comparison_review") if hub_mode else None
        evidence = comparison_schedule(state.get("reasoning_trace") or [])
        out = run_coro_sync(
            review_comparison(
                state,
                evidence["required_sources"],
                lambda r, v: run_specialist(r, v, inline_handoff=True),
                lambda v: _writer_node(v, inline_handoff=True),
            )
        )
        if hub_mode:
            comp_dec = normalize_decision_dict(
                "comparison_review",
                None,
                round_index=int(out.get("retry_count") or state.get("retry_count") or 0),
                task_id=task.task_id if task else None,
            )
            comp_summary = compact_result_summary("comparison_review", state, out)
            out = {
                **out,
                "pending_tasks": [],
                "last_result": AgentResult(
                    role="comparison_review",
                    status="completed",
                    decision=comp_dec,
                    task_id=task.task_id if task else None,
                    summary=comp_summary,
                    round_index=int(out.get("retry_count") or state.get("retry_count") or 0),
                ).to_dict(),
            }
        return out

    def _after_comparison_review(state: AgentState) -> str:
        return dispatch_next_node("comparison_review_node", state, comparison_plan=True)

    def _after_retrieval(state: AgentState) -> str:
        return dispatch_next_node("retrieval_node", state, comparison_plan=comparison_enabled)

    def _after_reader(state: AgentState) -> str:
        return dispatch_next_node("reader_node", state, comparison_plan=comparison_enabled)

    def _parallel_dispatch_node(state: AgentState) -> dict[str, Any]:
        """Execute a Supervisor-dispatched batch of independent retrieval `AgentTask`s via `TaskScheduler`."""
        pending_raw = [dict(t) for t in (state.get("pending_tasks") or []) if isinstance(t, dict)]
        if not pending_raw:
            return {"pending_tasks": []}
        tasks = [AgentTask.from_dict(item) for item in pending_raw]
        batch_kind = str((tasks[0].metadata or {}).get("batch_kind") or "sub_query")
        top_k = max(4, int(state.get("top_k") or 10))
        round_idx = int(state.get("retry_count") or 0)

        def _run_one_task(t: AgentTask) -> dict[str, Any]:
            meta = t.metadata or {}
            branch_state: dict[str, Any] = {**state, "pending_tasks": [t.to_dict()]}
            if meta.get("target_pdf"):
                branch_state["pdf_path"] = str(meta["target_pdf"])
                branch_state["pdf_paths"] = [str(meta["target_pdf"])]
            elif meta.get("target_query"):
                branch_state["query"] = str(meta["target_query"])
                branch_state["sub_queries"] = [str(meta["target_query"])]
            with use_run_config(run_config):
                if pilot_enabled and _accepts_kwarg(run_agents["retrieval"], "parallel_search"):
                    return run_agents["retrieval"](branch_state, parallel_search=True)
                return run_agents["retrieval"](branch_state)

        async def _execute_batch() -> dict[str, Any]:
            plan_tasks = [ResearchTask(id=t.task_id) for t in tasks]
            by_id = {t.task_id: t for t in tasks}
            policies = {t.task_id: TaskPolicy(timeout_seconds=75.0, max_retries=0) for t in tasks}
            scheduler = TaskScheduler(
                plan_tasks,
                max_concurrency=min(2, len(tasks)),
                task_policies=policies,
                run_timeout_seconds=150.0,
            )

            async def _exec(rtask: ResearchTask, _deps: Any) -> dict[str, Any]:
                return await asyncio.to_thread(_run_one_task, by_id[rtask.id])

            return await scheduler.run(_exec)

        outcomes = run_coro_sync(_execute_batch())

        # Deterministic RRF merge + round-robin per-branch coverage guarantee across completed parallel branches
        rrf_k = 60.0
        merged_scores: dict[str, float] = {}
        merged_by_key: dict[str, dict[str, Any]] = {}
        per_branch_top_keys: list[str] = []
        completed_ids: list[str] = []
        failed_ids: list[str] = []
        branch_messages: list[dict[str, Any]] = []
        branch_schedule_traces: list[dict[str, Any]] = []
        branch_tools_used: list[str] = ["task_scheduler_parallel_batch"]
        branch_summaries: list[dict[str, Any]] = []

        for t in tasks:
            outcome = outcomes.get(t.task_id)
            if outcome and outcome.status == TaskStatus.COMPLETED and isinstance(outcome.output, dict):
                completed_ids.append(t.task_id)
                for ev in outcome.output.get("reasoning_trace") or []:
                    if isinstance(ev, dict) and ev.get("action") == "retrieval_task_schedule":
                        branch_schedule_traces.append(ev)
                for tool_name in outcome.output.get("tools_used") or []:
                    if isinstance(tool_name, str) and tool_name.strip() and tool_name not in branch_tools_used:
                        branch_tools_used.append(tool_name)
                branch_docs = [d for d in (outcome.output.get("retrieved_docs") or []) if isinstance(d, dict)]
                branch_keys: list[str] = []
                for rank_idx, doc in enumerate(branch_docs):
                    key = str(
                        doc.get("id")
                        or doc.get("chunk_id")
                        or f"{doc.get('doc_id')}:{doc.get('page_no')}:{str(doc.get('text') or '')[:80]}"
                    )
                    merged_scores[key] = merged_scores.get(key, 0.0) + (1.0 / (rrf_k + rank_idx + 1.0))
                    if key not in merged_by_key:
                        merged_by_key[key] = dict(doc)
                    branch_keys.append(key)
                if branch_keys:
                    per_branch_top_keys.append(branch_keys[0])
                branch_summaries.append({
                    "task_id": t.task_id,
                    "status": "COMPLETED",
                    "retrieved_count": len(branch_docs),
                    "target_pdf": (t.metadata or {}).get("target_pdf"),
                    "target_query": (t.metadata or {}).get("target_query"),
                })
                branch_messages.append(make_agent_message(
                    sender=t.role,
                    recipient="supervisor",
                    kind="parallel_task_completed",
                    action="parallel_branch_done",
                    reason=t.reason,
                    task_id=t.task_id,
                    focus=t.focus,
                    summary={"retrieved_count": len(branch_docs), "batch_kind": batch_kind},
                    round_index=round_idx,
                ))
            else:
                failed_ids.append(t.task_id)
                branch_summaries.append({
                    "task_id": t.task_id,
                    "status": "FAILED",
                    "retrieved_count": 0,
                    "target_pdf": (t.metadata or {}).get("target_pdf"),
                    "target_query": (t.metadata or {}).get("target_query"),
                })
                branch_messages.append(make_agent_message(
                    sender=t.role,
                    recipient="supervisor",
                    kind="parallel_task_failed",
                    action="parallel_branch_failed",
                    reason=str(getattr(outcome, "error", "failed"))[:180],
                    task_id=t.task_id,
                    focus=t.focus,
                    summary={"batch_kind": batch_kind},
                    round_index=round_idx,
                ))

        effective_limit = max(top_k, len(tasks) * 2)
        ranked_keys = sorted(merged_scores.keys(), key=lambda k: merged_scores[k], reverse=True)
        # Guarantee at least the top-1 chunk from every completed branch is retained (essential for 6-PDF coverage)
        selected_keys: list[str] = []
        for bk in per_branch_top_keys:
            if bk in merged_by_key and bk not in selected_keys:
                selected_keys.append(bk)
        for rk in ranked_keys:
            if len(selected_keys) >= effective_limit:
                break
            if rk not in selected_keys:
                selected_keys.append(rk)
        merged_docs = [merged_by_key[k] for k in selected_keys[:effective_limit]]

        prev_seen_list = list(state.get("seen_chunks") or [])
        prev_ids = {
            (d.get("id") or d.get("chunk_id") or (d.get("doc_id"), d.get("page_no"), str(d.get("text") or "")[:80]))
            for d in prev_seen_list
            if isinstance(d, dict)
        }
        fresh_docs = [
            d for d in merged_docs
            if (d.get("id") or d.get("chunk_id") or (d.get("doc_id"), d.get("page_no"), str(d.get("text") or "")[:80])) not in prev_ids
        ]

        batch_trace = {
            "agent": "task_scheduler",
            "action": "supervisor_parallel_batch_completed",
            "batch_kind": batch_kind,
            "total_tasks": len(tasks),
            "completed_tasks": len(completed_ids),
            "failed_tasks": len(failed_ids),
            "task_ids": [t.task_id for t in tasks],
            "merged_docs_count": len(merged_docs),
            "round_index": round_idx,
        }
        canonical_dag_trace: dict[str, Any] = (
            {
                "agent": "task_scheduler",
                "action": "document_fanout_dag_executed",
                "status": "completed" if completed_ids else "failed",
                "round_index": round_idx,
                "document_count": len(tasks),
                "completed_documents": len(completed_ids),
                "failed_documents": len(failed_ids),
                "merged_chunks": len(merged_docs),
                "branches": branch_summaries,
            }
            if batch_kind == "document"
            else {
                "agent": "task_scheduler",
                "action": "research_dag_executed",
                "status": "completed" if completed_ids else "failed",
                "round_index": round_idx,
                "branch_count": len(tasks),
                "completed_branches": len(completed_ids),
                "failed_branches": len(failed_ids),
                "merged_chunks": len(merged_docs),
                "task_outcomes": {b["task_id"]: b["status"] for b in branch_summaries},
                "branches": branch_summaries,
            }
        )
        batch_update: dict[str, Any] = {
            "retrieved_docs": merged_docs,
            "seen_chunks": [*prev_seen_list, *fresh_docs],
            "no_new_evidence": bool(state.get("retry_count", 0) > 0 and prev_seen_list and not fresh_docs),
            "tools_used": branch_tools_used,
            "agent_messages": branch_messages,
            "reasoning_trace": [*branch_schedule_traces, canonical_dag_trace, batch_trace],
        }
        existing_schema = dict(state.get("document_schema") or {})
        if batch_kind == "document":
            existing_schema["document_fanout"] = canonical_dag_trace
        if isinstance(existing_schema.get("research_dag"), dict):
            existing_schema["research_dag"] = sync_dag_manifest_from_retrieval(
                existing_schema["research_dag"],
                canonical_dag_trace if batch_kind == "sub_query" else None,
                retrieved_count=len(merged_docs),
                round_index=int(state.get("dag_round_index") or 0),
            )
            batch_update["document_schema"] = existing_schema
        elif batch_kind == "document":
            batch_update["document_schema"] = existing_schema
        return _attach_protocol_envelope("retrieval", state, batch_update, tasks[0])

    def _make_graph_node(
        fn: Callable[[AgentState], dict[str, Any]],
        *,
        inference_mode: str | None = None,
    ) -> RunnableLambda:
        """Bind `RunConfig` (and optional `agent_inference_mode`) for BOTH sync `.invoke()` and non-blocking async `.astream()`/`.ainvoke()`."""
        def _sync(state: AgentState) -> dict[str, Any]:
            with use_run_config(run_config):
                if inference_mode:
                    with agent_inference_mode(inference_mode):
                        return fn(state)
                return fn(state)

        async def _async(state: AgentState) -> dict[str, Any]:
            return await asyncio.to_thread(_sync, state)

        return RunnableLambda(_sync, afunc=_async)

    workflow = StateGraph(AgentState)

    # Bind the snapshot at the node boundary, supporting both sync `.invoke()` and non-blocking async `.astream()`.
    def _router_wrapper(state: AgentState) -> dict[str, Any]:
        out = orchestrator_router_node(
            state,
            planner_llm_fn=planner_llm_fn,
            execution_mode=mode,
        )
        out["supervisor_mode"] = "hub" if hub_mode else "pipeline"
        if graph_disabled_handoffs:
            out["disabled_agent_handoffs"] = list(dict.fromkeys([
                *graph_disabled_handoffs,
                *(str(r).strip().lower() for r in (state.get("disabled_agent_handoffs") or []) if isinstance(r, str) and r.strip()),
            ]))
        if hub_mode:
            out["last_result"] = {}
            out["pending_tasks"] = []
        return out

    workflow.add_node("orchestrator_router", _make_graph_node(_router_wrapper))
    workflow.add_node("scout_node", _make_graph_node(_run_scout, inference_mode=scout_mode))
    for role in ("retrieval", "reader", "summarizer", "fact_checker"):
        workflow.add_node(f"{role}_node", _make_graph_node(partial(run_specialist, role)))
    workflow.add_node("escalation_node", _make_graph_node(_escalation))
    workflow.add_node("writer_node", _make_graph_node(_writer_node))
    if comparison_enabled:
        workflow.add_node("comparison_review_node", _make_graph_node(_comparison_review))

    valid_entry_nodes = [
        "orchestrator_router",
        "scout_node",
        "retrieval_node",
        "reader_node",
        "summarizer_node",
        "fact_checker_node",
        "writer_node",
        "escalation_node",
    ]
    if comparison_enabled:
        valid_entry_nodes.append("comparison_review_node")
    if hub_mode:
        def _hub_wrapper(state: AgentState) -> dict[str, Any]:
            scoped = state
            if graph_disabled_handoffs:
                merged_dis = list(dict.fromkeys([
                    *graph_disabled_handoffs,
                    *(str(r).strip().lower() for r in (state.get("disabled_agent_handoffs") or []) if isinstance(r, str) and r.strip()),
                ]))
                scoped = {**state, "disabled_agent_handoffs": merged_dis}
            return supervisor_hub_step(
                scoped,
                comparison_plan=comparison_enabled,
                allow_scout_handoff=scout_handoff_allowed,
                parallel_retrieval=parallel_retrieval,
                arbiter_llm_fn=arbiter_llm_fn,
                execution_mode=mode,
            )

        workflow.add_node("supervisor_node", _make_graph_node(_hub_wrapper))
        workflow.add_node("parallel_dispatch_node", _make_graph_node(_parallel_dispatch_node))
        valid_entry_nodes.extend(["supervisor_node", "parallel_dispatch_node"])

    if start_at_node and start_at_node in valid_entry_nodes and start_at_node != "orchestrator_router":
        workflow.add_conditional_edges(
            START,
            lambda _state, _target=start_at_node: _target,
            valid_entry_nodes,
        )
    else:
        workflow.add_edge(START, "orchestrator_router")

    if hub_mode:
        # True Hub-and-Spoke Multi-Agent Topology:
        # Planner -> Supervisor Hub <-> all specialist nodes (+ controlled parallel batch dispatch);
        # ZERO hardcoded edges between specialists!
        hub_targets = [
            "scout_node",
            "retrieval_node",
            "parallel_dispatch_node",
            "reader_node",
            "summarizer_node",
            "fact_checker_node",
            "writer_node",
            "escalation_node",
            END,
        ]
        if comparison_enabled:
            hub_targets.append("comparison_review_node")
        workflow.add_edge("orchestrator_router", "supervisor_node")
        workflow.add_conditional_edges("supervisor_node", route_from_supervisor_hub, hub_targets)
        for specialist_node in (
            "scout_node",
            "retrieval_node",
            "parallel_dispatch_node",
            "reader_node",
            "summarizer_node",
            "fact_checker_node",
            "writer_node",
        ):
            workflow.add_edge(specialist_node, "supervisor_node")
        if comparison_enabled:
            workflow.add_edge("comparison_review_node", "supervisor_node")
        workflow.add_edge("escalation_node", END)
        return workflow.compile()

    workflow.add_conditional_edges("orchestrator_router", route_after_router, ["scout_node", "retrieval_node"])
    workflow.add_conditional_edges("scout_node", route_after_scout, ["retrieval_node", END])
    if comparison_enabled:
        workflow.add_conditional_edges(
            "retrieval_node",
            _after_retrieval,
            ["reader_node", "retrieval_node", "scout_node", "comparison_review_node", END],
        )
        workflow.add_conditional_edges(
            "reader_node",
            _after_reader,
            ["summarizer_node", "retrieval_node", "reader_node", "scout_node", "comparison_review_node", END],
        )
        workflow.add_conditional_edges(
            "comparison_review_node", _after_comparison_review,
            ["retrieval_node", "reader_node", "comparison_review_node", "scout_node", END],
        )
    else:
        workflow.add_conditional_edges(
            "retrieval_node",
            route_after_retrieval,
            ["reader_node", "retrieval_node", "scout_node"],
        )
        workflow.add_conditional_edges(
            "reader_node",
            route_after_reader,
            ["summarizer_node", "retrieval_node", "reader_node", "scout_node"],
        )

    correction_targets = ["retrieval_node", "reader_node", "summarizer_node"]
    workflow.add_conditional_edges(
        "summarizer_node", route_after_summarizer, [*correction_targets, "fact_checker_node"]
    )
    workflow.add_conditional_edges(
        "fact_checker_node", route_after_fact_check, [*correction_targets, "scout_node", "writer_node", "escalation_node"]
    )
    workflow.add_conditional_edges(
        "writer_node", route_after_writer, ["summarizer_node", "fact_checker_node", END]
    )
    workflow.add_edge("escalation_node", END)

    return workflow.compile()


build_graph = build_research_rag_graph

