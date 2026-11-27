import asyncio
from functools import partial
import inspect
from typing import Any, Callable, Optional
from langgraph.graph import END, START, StateGraph

from agents.agent0_scout.node import ScoutAgent
from core.citation_audit import citation_location, get_citation_audit, write_with_citation_audit
from core.comparison_plan import (
    compare_documents, comparison_schedule, get_comparison_budget, incomplete_comparison, review_comparison,
    run_coro_sync,
)
from core.config import RunConfig, agent_inference_mode, get_run_settings, use_run_config
from core.coordination import decide_handoff, evaluate_agent_handoff, fact_check_accepted
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
from core.research_tasks import TaskStatus
from core.state import AgentState
from core.supervisor import supervisor_planner
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
        should_refine = (
            retry_count == 0
            and base_plan.complexity in {"MEDIUM", "HIGH"}
            and is_gemini_permitted(active_mode, role="planner")
            and bool(getattr(run_settings, "GEMINI_API_KEY", ""))
        )
        if should_refine:
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

    dag = build_research_dag_plan(
        plan,
        evidence_gaps=evidence_gaps,
        round_index=retry_count,
    )
    dag_manifest = dag.to_manifest()
    existing_schema = dict(state.get("document_schema") or {})
    existing_schema["research_dag"] = dag_manifest

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
) -> str:
    """
    Single unified dispatch policy shared by fresh LangGraph runs, retry loops, and checkpoint resume.
    Routes every completed node across all 6 specialist roles + Supervisor consistently.
    """
    node = (completed_node or "").strip()
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
) -> str:
    """Resolve the exact successor node to execute when resuming after `completed_node`."""
    return dispatch_next_node(completed_node, state, comparison_plan=comparison_plan)


_agents: dict[str, Any] = {}


def build_research_rag_graph(
    mode: Optional[str] = None, *, run_config: RunConfig | None = None, auto_scout: bool = True,
    allow_scout_handoff: Optional[bool] = None,
    parallel_retrieval: bool = False, comparison_plan: bool = False,
    planner_llm_fn: Optional[Callable[[str], Any]] = None,
    start_at_node: Optional[str] = None,
):
    """Build a run-local team with unified 6-role supervised handoff dispatch."""
    run_config = run_config or RunConfig.capture()
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

    def _run_scout(state: AgentState) -> dict[str, Any]:
        request = _handoff_for(state, "scout", "recipient")
        if not request:
            return scout.run(state)
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
            "signature": request["signature"],
            "retry_count": request["retry_count"],
        }
        if request.get("decision_id"):
            receipt["decision_id"] = request["decision_id"]
        if request.get("task_id"):
            receipt["task_id"] = request["task_id"]
        new_paths = list(dict.fromkeys(
            update.get("pdf_paths") or ([update["pdf_path"]] if update.get("pdf_path") else [])
        ))
        merged_paths = list(dict.fromkeys([*existing_paths, *new_paths]))
        outcome_event = {
            "action": "handoff_outcome_recorded",
            "agent": "scout",
            "sender": request["sender"],
            "recipient": "scout",
            "signature": request["signature"],
            "decision_id": request.get("decision_id"),
            "task_id": request.get("task_id"),
            "outcome_status": "discovered_new_documents" if new_paths else "no_new_documents",
            "new_documents_count": len([p for p in new_paths if p not in existing_paths]),
        }
        trace = [
            receipt,
            *(update.get("reasoning_trace") or []),
            outcome_event,
        ]
        if new_paths and merged_paths:
            return {
                **update,
                "pdf_path": merged_paths[0],
                "pdf_paths": merged_paths,
                "no_new_evidence": False,
                "reasoning_trace": trace,
            }
        if not existing_paths and update.get("answer_status"):
            return {
                **update,
                "pdf_path": None,
                "pdf_paths": [],
                "no_new_evidence": True,
                "reasoning_trace": [receipt, *(update.get("reasoning_trace") or [])],
            }
        # If discovery found no new PDF, preserve existing paths and let retrieval/escalation close cleanly
        return {
            "pdf_path": existing_paths[0] if existing_paths else None,
            "pdf_paths": existing_paths,
            "no_new_evidence": True,
            "reasoning_trace": trace,
        }

    def run_specialist(role: str, state: AgentState) -> dict[str, Any]:
        request = _handoff_for(state, role, "recipient")
        inputs = prepare_handoff_input(state, request) if request else state
        existing_schema = dict(inputs.get("document_schema") or state.get("document_schema") or {})
        dag_manifest = existing_schema.get("research_dag") if isinstance(existing_schema.get("research_dag"), dict) else None
        gap_queries = (
            pending_dag_investigation_queries(dag_manifest)
            if role == "retrieval" and state.get("retry_count", 0) > 0
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
            update = draft_visual_answer(inputs)
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

                current_round = int(state.get("retry_count") or 0)
                prev_seen = inputs.get("seen_chunks") or state.get("seen_chunks") or []
                try:
                    dag_update, dag_trace = execute_retrieval_dag(
                        sub_queries,
                        _run_branch,
                        top_k=int(inputs.get("top_k") or 10),
                        round_index=current_round,
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

                current_round = int(state.get("retry_count") or 0)
                prev_seen = inputs.get("seen_chunks") or state.get("seen_chunks") or []
                try:
                    fanout_update, fanout_trace = execute_document_fanout_dag(
                        selected_pdfs,
                        _run_doc_branch,
                        top_k=int(inputs.get("top_k") or 10),
                        round_index=current_round,
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
                round_index=int(state.get("retry_count") or 0),
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
                "signature": request["signature"],
                "retry_count": request["retry_count"],
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
            schema = dict(update.get("document_schema") or state.get("document_schema") or {})
            schema["evidence_store"] = EvidenceStore.from_state({**state, **update}).to_dict()
            src_lang = detect_corpus_language(
                update.get("source_units") or state.get("source_units") or state.get("retrieved_docs"),
                default=state.get("source_language") or "en",
            )
            update = {**update, "document_schema": schema, "source_language": src_lang}

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

        if role == "summarizer":
            schema = dict(update.get("document_schema") or state.get("document_schema") or {})
            schema["evidence_store"] = EvidenceStore.from_state({**state, **update}).to_dict()
            if isinstance(schema.get("research_dag"), dict):
                has_answer = bool((update.get("answer") or state.get("answer") or "").strip())
                schema["research_dag"] = mark_dag_stage_status(
                    schema.get("research_dag"),
                    "synthesize",
                    TaskStatus.COMPLETED if has_answer else TaskStatus.FAILED,
                    round_index=int(state.get("retry_count") or 0),
                    metadata_update={"claims_count": len(update.get("claims") or [])},
                )
            update = {**update, "document_schema": schema}

        should_evaluate_decision = not visual and (
            role in {"summarizer", "fact_checker"}
            or (role in {"reader", "retrieval"} and update.get("agent_decision") is not None)
        )
        if should_evaluate_decision:
            reviewed = {
                **state,
                **update,
                "allow_scout_handoff": scout_handoff_allowed,
                "reasoning_trace": [*(state.get("reasoning_trace") or []), *trace],
            }
            decision, audit_event = evaluate_agent_handoff(role, reviewed)
            if audit_event is not None:
                trace.append(audit_event)
            if role == "fact_checker":
                schema = dict(update.get("document_schema") or state.get("document_schema") or {})
                store = EvidenceStore.from_state(reviewed)
                schema["evidence_store"] = store.to_dict()
                if isinstance(schema.get("research_dag"), dict):
                    if fact_check_accepted(reviewed):
                        verify_outcome = "PASSED"
                    else:
                        label = str(reviewed.get("fact_check_label") or "").lower()
                        rejected = reviewed.get("rejected_claims") or []
                        has_contradiction = label in {"contradict", "contradicted", "refuted"} or any(
                            str(item.get("status", "")).lower() in {"contradict", "contradicted", "refuted"}
                            for item in rejected if isinstance(item, dict)
                        )
                        verify_outcome = "CONTRADICTED" if has_contradiction else "NEEDS_MORE_EVIDENCE"
                    schema["research_dag"] = mark_dag_stage_status(
                        schema.get("research_dag"),
                        "verify",
                        TaskStatus.COMPLETED,
                        round_index=int(state.get("retry_count") or 0),
                        review_outcome=verify_outcome,
                        metadata_update={
                            "fact_check_label": reviewed.get("fact_check_label"),
                            "verified_count": len(reviewed.get("verified_claims") or []),
                            "rejected_count": len(reviewed.get("rejected_claims") or []),
                        },
                    )
                gaps = reviewed.get("evidence_gaps") or []
                if gaps and decision is not None and decision.recipient in {"retrieval", "scout", "reader"}:
                    replanned = replan_research_dag_for_gaps(schema.get("research_dag"), gaps)
                    schema["research_dag"] = replanned
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
                handoff_event = {**decision.as_trace(), "agent": role, "retry_count": attempt}
                if audit_event and audit_event.get("decision_id"):
                    handoff_event["decision_id"] = audit_event["decision_id"]
                trace.append(handoff_event)
                update["retry_count"] = attempt
        update["reasoning_trace"] = trace
        return update

    def _mark_write_stage(state: AgentState, result: dict[str, Any]) -> dict[str, Any]:
        schema = dict(result.get("document_schema") or state.get("document_schema") or {})
        if isinstance(schema.get("research_dag"), dict):
            schema["research_dag"] = mark_dag_stage_status(
                schema.get("research_dag"),
                "write",
                TaskStatus.COMPLETED,
                round_index=int(state.get("retry_count") or 0),
                metadata_update={"confidence": result.get("confidence")},
            )
            result = {**result, "document_schema": schema}
        return result

    def _writer_node(state: AgentState) -> dict[str, Any]:
        writer = write_visual_answer if figure_context(state).get("explicit") else run_agents["writer"]
        result = write_with_citation_audit(state, writer)
        comparison = comparison_schedule(state.get("reasoning_trace") or []) if comparison_enabled else None
        if comparison:
            references = get_citation_audit(result["reasoning_trace"]).get("references", [])
            cited = {location[0] for ref in references if (location := citation_location(ref))}
            if not cited.issuperset(comparison["required_sources"]):
                return incomplete_comparison("missing_audited_source")
        if state.get("critic_action") == "lower_confidence":
            result["confidence"] = "low"
        else:
            result.setdefault("confidence", "high" if state.get("verified_claims") else "low")
        if result.get("agent_decision") is not None:
            w_trace = list(result.get("reasoning_trace") or [])
            reviewed_w = {
                **state,
                **result,
                "allow_scout_handoff": scout_handoff_allowed,
                "reasoning_trace": [*(state.get("reasoning_trace") or []), *w_trace],
            }
            w_decision, w_audit = evaluate_agent_handoff("writer", reviewed_w)
            if w_audit is not None:
                w_trace.append(w_audit)
            if w_decision is not None:
                attempt = state.get("retry_count", 0) + 1
                w_handoff = {**w_decision.as_trace(), "agent": "writer", "retry_count": attempt}
                if w_audit and w_audit.get("decision_id"):
                    w_handoff["decision_id"] = w_audit["decision_id"]
                w_trace.append(w_handoff)
                result = {**result, "retry_count": attempt, "reasoning_trace": w_trace}
            else:
                result = {**result, "reasoning_trace": w_trace}
        return _mark_write_stage(state, attach_answer_figures(state, result))

    def _escalation(state: AgentState) -> dict[str, Any]:
        return _mark_write_stage(
            state,
            attach_answer_figures(state, _write_escalated_report(state, run_agents["writer"])),
        )

    def _comparison_review(state: AgentState) -> dict[str, Any]:
        evidence = comparison_schedule(state.get("reasoning_trace") or [])
        return run_coro_sync(review_comparison(state, evidence["required_sources"], run_specialist, _writer_node))

    def _after_comparison_review(state: AgentState) -> str:
        return dispatch_next_node("comparison_review_node", state, comparison_plan=True)

    def _after_retrieval(state: AgentState) -> str:
        return dispatch_next_node("retrieval_node", state, comparison_plan=comparison_enabled)

    def _after_reader(state: AgentState) -> str:
        return dispatch_next_node("reader_node", state, comparison_plan=comparison_enabled)

    workflow = StateGraph(AgentState)

    # Bind the snapshot at the node boundary, including when LangGraph runs in an executor.
    router_fn = (
        partial(orchestrator_router_node, planner_llm_fn=planner_llm_fn, execution_mode=mode)
        if (planner_llm_fn or mode)
        else orchestrator_router_node
    )
    workflow.add_node("orchestrator_router", use_run_config(run_config)(router_fn))
    workflow.add_node("scout_node", use_run_config(run_config)(agent_inference_mode(scout_mode)(_run_scout)))
    for role in ("retrieval", "reader", "summarizer", "fact_checker"):
        workflow.add_node(f"{role}_node", use_run_config(run_config)(partial(run_specialist, role)))
    workflow.add_node("escalation_node", use_run_config(run_config)(_escalation))
    workflow.add_node("writer_node", use_run_config(run_config)(_writer_node))

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

    if start_at_node and start_at_node in valid_entry_nodes and start_at_node != "orchestrator_router":
        workflow.add_conditional_edges(
            START,
            lambda _state, _target=start_at_node: _target,
            valid_entry_nodes,
        )
    else:
        workflow.add_edge(START, "orchestrator_router")
    workflow.add_edge("orchestrator_router", "scout_node")
    workflow.add_conditional_edges("scout_node", route_after_scout, ["retrieval_node", END])
    if comparison_enabled:
        workflow.add_node("comparison_review_node", use_run_config(run_config)(_comparison_review))
        workflow.add_conditional_edges(
            "retrieval_node",
            _after_retrieval,
            ["reader_node", "scout_node", "comparison_review_node", END],
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
            ["reader_node", "scout_node"],
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

