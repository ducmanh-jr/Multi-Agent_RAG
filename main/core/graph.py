import asyncio
from functools import partial
from typing import Any, Callable, Optional
from langgraph.graph import END, START, StateGraph

from agents.agent0_scout.node import ScoutAgent
from core.citation_audit import citation_location, get_citation_audit, write_with_citation_audit
from core.comparison_plan import (
    compare_documents, comparison_schedule, get_comparison_budget, incomplete_comparison, review_comparison,
    run_coro_sync,
)
from core.config import RunConfig, agent_inference_mode, get_run_settings, use_run_config
from core.coordination import decide_handoff, fact_check_accepted
from core.evidence_store import EvidenceStore
from core.figure_qa import (
    add_figure_evidence, attach_answer_figures, check_visual_answer,
    draft_visual_answer, figure_context, write_visual_answer,
)
from core.handoff_inputs import prepare_handoff_input
from core.provider_router import is_gemini_permitted
from core.research_dag import (
    build_research_dag_plan,
    execute_retrieval_dag,
    replan_research_dag_for_gaps,
)
from core.state import AgentState
from core.supervisor import supervisor_planner
from modes.registry import get_active_agents, get_agent_runners, normalize_mode, resolve_agent_modes
from utils.logger import logger


# ===================== NODE FUNCTIONS =====================

def orchestrator_router_node(
    state: AgentState,
    *,
    planner_llm_fn: Optional[Callable[[str], Any]] = None,
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

    base_plan = supervisor_planner.plan_query(
        query=query,
        evidence_gaps=evidence_gaps,
        retry_count=retry_count,
    )
    plan = base_plan
    planning_source = "deterministic"

    custom_llm_fn = planner_llm_fn or state.get("planner_llm_fn")
    if callable(custom_llm_fn):
        refined = supervisor_planner.plan_query_with_llm(
            query=query,
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
            run_settings.get_inference_mode("retrieval")
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
                query=query,
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
    if trace and trace[-1].get("action") in {
        "discovery_clarify", "discovery_rejected", "discovery_failed", "download_failed", "waiting_for_document",
    }:
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


def build_research_rag_graph(
    mode: Optional[str] = None, *, run_config: RunConfig | None = None, auto_scout: bool = True,
    parallel_retrieval: bool = False, comparison_plan: bool = False,
    planner_llm_fn: Optional[Callable[[str], Any]] = None,
):
    """Build a run-local team. Only the two reviewing roles can initiate corrective work."""
    run_config = run_config or RunConfig.capture()
    with use_run_config(run_config):
        run_agents = get_agent_runners(mode)
        reader_mode = normalize_mode(mode or run_config.get_agent_mode("reader") or run_config.EXECUTION_MODE)
        scout = ScoutAgent(auto_scout=auto_scout)
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

    def run_comparison_step(role, inputs, plan, previous):
        reader = (
            partial(read_comparison_document, run_agents["reader"], inputs.get("query", ""))
            if is_cpu_min_team
            else partial(_read_comparison_generic, inputs.get("query", ""))
        )
        top_k = max(len(plan.paths), min(int(inputs.get("top_k") or 8), 10))
        if role == "retrieval":
            if is_cpu_min_team:
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
        update = scout.run(discovery_state)
        trace = [
            {
                "action": "agent_handoff_received",
                "agent": "scout",
                "sender": request["sender"],
                "recipient": "scout",
                "reason": request["reason"],
                "signature": request["signature"],
                "retry_count": request["retry_count"],
            },
            *(update.get("reasoning_trace") or []),
        ]
        new_paths = list(dict.fromkeys(
            update.get("pdf_paths") or ([update["pdf_path"]] if update.get("pdf_path") else [])
        ))
        merged_paths = list(dict.fromkeys([*existing_paths, *new_paths]))
        if new_paths and merged_paths:
            return {
                **update,
                "pdf_path": merged_paths[0],
                "pdf_paths": merged_paths,
                "no_new_evidence": False,
                "reasoning_trace": trace,
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
        visual = figure_context(inputs).get("explicit", False)
        previous = comparison_schedule(state.get("reasoning_trace") or []) if comparison_enabled and role == "reader" else None
        comparison = (
            supervisor_planner.plan_document_comparison(inputs)
            if comparison_enabled and (role == "retrieval" or (previous and request)) else None
        )
        if visual and role == "summarizer":
            update = draft_visual_answer(inputs)
        elif visual and role == "fact_checker":
            update = check_visual_answer(inputs)
        elif comparison is not None:
            update = run_comparison_step(role, inputs, comparison, previous)
        elif pilot_enabled and role == "retrieval":
            update = run_agents[role](inputs, parallel_search=True)
        elif parallel_retrieval and role == "retrieval":
            sub_queries = list(dict.fromkeys(
                q.strip() for q in (inputs.get("sub_queries") or [])
                if isinstance(q, str) and q.strip()
            ))
            if len(sub_queries) >= 2:
                def _run_branch(branch_query: str) -> dict[str, Any]:
                    branch_state = {
                        **inputs,
                        "query": branch_query,
                        "sub_queries": [branch_query],
                    }
                    return run_agents["retrieval"](branch_state)

                dag_update, dag_trace = execute_retrieval_dag(
                    sub_queries,
                    _run_branch,
                    top_k=int(inputs.get("top_k") or 10),
                    round_index=int(state.get("retry_count") or 0),
                )
                if dag_update.get("retrieved_docs"):
                    update = {
                        **dag_update,
                        "reasoning_trace": [dag_trace],
                    }
                else:
                    update = run_agents[role](inputs)
            else:
                update = run_agents[role](inputs)
        else:
            update = run_agents[role](inputs)
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
            reviewed = {
                **state,
                **update,
                "allow_scout_handoff": bool(auto_scout),
                "reasoning_trace": [*(state.get("reasoning_trace") or []), *trace],
            }
            if role == "fact_checker":
                schema = dict(update.get("document_schema") or state.get("document_schema") or {})
                store = EvidenceStore.from_state(reviewed)
                schema["evidence_store"] = store.to_dict()
                gaps = reviewed.get("evidence_gaps") or []
                if gaps:
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
        return attach_answer_figures(state, result)

    def _escalation(state: AgentState) -> dict[str, Any]:
        return attach_answer_figures(state, _write_escalated_report(state, run_agents["writer"]))

    def _comparison_review(state: AgentState) -> dict[str, Any]:
        evidence = comparison_schedule(state.get("reasoning_trace") or [])
        return run_coro_sync(review_comparison(state, evidence["required_sources"], run_specialist, _writer_node))

    def _after_comparison_review(state: AgentState) -> str:
        last = state["reasoning_trace"][-1]
        if last.get("action") != "agent_handoff":
            return END
        return "comparison_review_node" if last["recipient"] == "summarizer" else f"{last['recipient']}_node"

    def _after_retrieval(state: AgentState) -> str:
        schedule = comparison_schedule(state.get("reasoning_trace") or [])
        if schedule is None:
            return "reader_node"
        return "comparison_review_node" if schedule["status"] == "COMPLETED" else END

    def _after_reader(state: AgentState) -> str:
        schedule = comparison_schedule(state.get("reasoning_trace") or [])
        if schedule is None:
            return "summarizer_node"
        return "comparison_review_node" if schedule["status"] == "COMPLETED" else END

    workflow = StateGraph(AgentState)

    # Bind the snapshot at the node boundary, including when LangGraph runs in an executor.
    router_fn = partial(orchestrator_router_node, planner_llm_fn=planner_llm_fn) if planner_llm_fn else orchestrator_router_node
    workflow.add_node("orchestrator_router", use_run_config(run_config)(router_fn))
    workflow.add_node("scout_node", use_run_config(run_config)(agent_inference_mode(scout_mode)(_run_scout)))
    for role in ("retrieval", "reader", "summarizer", "fact_checker"):
        workflow.add_node(f"{role}_node", use_run_config(run_config)(partial(run_specialist, role)))
    workflow.add_node("escalation_node", use_run_config(run_config)(_escalation))
    workflow.add_node("writer_node", use_run_config(run_config)(_writer_node))

    workflow.add_edge(START, "orchestrator_router")
    workflow.add_edge("orchestrator_router", "scout_node")
    workflow.add_conditional_edges("scout_node", route_after_scout, ["retrieval_node", END])
    if comparison_enabled:
        workflow.add_node("comparison_review_node", use_run_config(run_config)(_comparison_review))
        workflow.add_conditional_edges("retrieval_node", _after_retrieval, ["reader_node", "comparison_review_node", END])
        workflow.add_conditional_edges("reader_node", _after_reader, ["summarizer_node", "comparison_review_node", END])
        workflow.add_conditional_edges(
            "comparison_review_node", _after_comparison_review,
            ["retrieval_node", "reader_node", "comparison_review_node", "scout_node", END],
        )
    else:
        workflow.add_edge("retrieval_node", "reader_node")
        workflow.add_edge("reader_node", "summarizer_node")

    correction_targets = ["retrieval_node", "reader_node", "summarizer_node"]
    workflow.add_conditional_edges(
        "summarizer_node", route_after_summarizer, [*correction_targets, "fact_checker_node"]
    )
    workflow.add_conditional_edges(
        "fact_checker_node", route_after_fact_check, [*correction_targets, "scout_node", "writer_node", "escalation_node"]
    )
    workflow.add_edge("writer_node", END)
    workflow.add_edge("escalation_node", END)

    return workflow.compile()


build_graph = build_research_rag_graph

