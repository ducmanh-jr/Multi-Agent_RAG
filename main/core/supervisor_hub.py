"""
Supervisor Hub Orchestrator (Hub-and-Spoke Multi-Agent Policy & Gatekeeper).

In `supervisor_mode="hub"`, every specialist node executes its assigned `AgentTask` and returns
an `AgentResult` + mandatory `AgentDecision` to `supervisor_hub_node`. The Supervisor:
  1. Evaluates specialist proposals strictly through `supervisor_evaluate_decision` (policy matrix,
     per-category budgets, verifier starvation guard, DAG budget, canonical anti-loop signatures).
  2. Applies a zero-extra-LLM fast-path deterministic policy for clear transitions and `comparison_plan`.
  3. Optionally invokes a bounded, structured-output LLM arbiter ONLY when ambiguous conflict
     conditions exist and Gemini is permitted for `planner` (disabled in local-only / CPU modes).
  4. Issues the next typed `AgentTask` into `state["pending_tasks"]` and logs compact blackboard
     messages into `state["agent_messages"]`.
"""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional

from langgraph.graph import END

from core.agent_protocol import AgentTask, make_agent_message
from core.comparison_plan import comparison_schedule
from core.config import agent_inference_mode, get_run_settings
from core.coordination import (
    AgentDecision,
    fact_check_accepted,
    next_handoff_budgets,
    supervisor_evaluate_decision,
)
from core.figure_qa import figure_context
from core.provider_router import is_gemini_permitted
from core.research_dag import replan_research_dag_for_gaps
from core.state import AgentState
from utils.logger import logger


MAX_HUB_STEPS = 24

_ROLE_OBJECTIVES: dict[str, str] = {
    "scout": "Validate attached documents or discover relevant academic papers.",
    "retrieval": "Retrieve grounded passages, tables, and equations matching sub-queries.",
    "reader": "Extract structured verbatim source units and verify table/figure completeness.",
    "summarizer": "Synthesize grounded atomic claims and draft answer with inner critic review.",
    "fact_checker": "Verify every claim cross-family against source units and surface specific gaps.",
    "writer": "Compile audited academic report with verified citation anchors.",
    "comparison_review": "Review multi-document comparison evidence and audit required sources.",
    "escalation": "Publish low-confidence escalated report with explicit verification warning.",
}

_ROLE_TO_NODE: dict[str, str] = {
    "scout": "scout_node",
    "retrieval": "retrieval_node",
    "reader": "reader_node",
    "summarizer": "summarizer_node",
    "fact_checker": "fact_checker_node",
    "writer": "writer_node",
    "comparison_review": "comparison_review_node",
    "escalation": "escalation_node",
}


def _hub_steps_so_far(state: Mapping[str, Any]) -> int:
    return sum(
        1
        for event in (state.get("reasoning_trace") or [])
        if isinstance(event, dict) and event.get("action") == "supervisor_hub_dispatched"
    )


def _is_ambiguous_conflict(
    last_role: str,
    state: Mapping[str, Any],
    candidates: list[str],
) -> bool:
    """
    Detects genuine orchestration ambiguity where an LLM arbiter may add value:
    mixed verified + rejected claims alongside unresolved evidence gaps and unread chunks.
    Simple/clear states ALWAYS take the deterministic fast-path with zero extra LLM calls.
    """
    if len(candidates) < 2:
        return False
    if last_role != "fact_checker":
        return False
    has_verified = bool(state.get("verified_claims"))
    has_rejected = bool(state.get("rejected_claims"))
    has_gaps = bool(state.get("evidence_gaps"))
    return has_verified and has_rejected and has_gaps


def _arbitrate_with_llm(
    *,
    state: Mapping[str, Any],
    candidates: list[str],
    last_result: Mapping[str, Any],
    arbiter_llm_fn: Optional[Callable[[str], Any]] = None,
    execution_mode: Optional[str] = None,
) -> tuple[str | None, str]:
    """Optional bounded structured-output arbiter choosing ONLY among Gatekeeper-valid candidates."""
    if not candidates:
        return None, ""

    llm_fn = arbiter_llm_fn
    if llm_fn is None:
        run_settings = get_run_settings()
        active_mode = execution_mode or run_settings.EXECUTION_MODE or "cpu_min"
        if not (
            is_gemini_permitted(active_mode, role="planner")
            and bool(getattr(run_settings, "GEMINI_API_KEY", ""))
        ):
            return None, ""
        from utils.llm_client import llm_client

        def _default_arbiter(prompt: str) -> Any:
            with agent_inference_mode("gemini"):
                return llm_client.generate_json(
                    prompt,
                    system_instruction=(
                        "You are the Research Supervisor Arbiter. Pick the single best next specialist "
                        "strictly from `allowed_candidates`. Return JSON: {\"next_role\": \"...\", \"rationale\": \"...\"}."
                    ),
                    agent="retrieval",
                    max_tokens=256,
                )

        llm_fn = _default_arbiter

    prompt = json.dumps(
        {
            "query": str(state.get("query") or "")[:300],
            "allowed_candidates": candidates,
            "last_result_summary": last_result.get("summary") or {},
            "retry_count": int(state.get("retry_count") or 0),
            "handoff_budgets": state.get("handoff_budgets") or {},
            "gaps": [
                str(g.get("missing_aspect") or g.get("suggested_query") or "")[:120]
                for g in (state.get("evidence_gaps") or [])[:3]
                if isinstance(g, dict)
            ],
        },
        ensure_ascii=False,
    )
    try:
        raw = llm_fn(prompt)
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        if isinstance(parsed, Mapping):
            chosen = str(parsed.get("next_role") or "").strip().lower()
            rationale = str(parsed.get("rationale") or "").strip()[:200]
            if chosen in candidates:
                return chosen, rationale
    except Exception as exc:
        logger.warning("[SupervisorHub] LLM arbitration failed (%s); using deterministic policy.", exc)
    return None, ""


def _next_default_role(
    last_role: str,
    state: Mapping[str, Any],
    *,
    comparison_plan: bool = False,
) -> str:
    """Fast-path deterministic transition when the specialist proceeded or a corrective handoff was rejected."""
    if not last_role or last_role == "supervisor":
        # Honor explicit supervisor -> retrieval handoff from Planner if emitted
        trace = state.get("reasoning_trace") or []
        if trace and isinstance(trace[-1], dict):
            last_ev = trace[-1]
            if last_ev.get("action") == "agent_handoff" and last_ev.get("sender") == "supervisor":
                rec = str(last_ev.get("recipient") or "").strip()
                if rec in _ROLE_TO_NODE:
                    return rec
        return "scout"

    if last_role == "scout":
        trace = state.get("reasoning_trace") or []
        if trace and isinstance(trace[-1], dict) and trace[-1].get("action") in {
            "discovery_clarify", "discovery_rejected", "discovery_failed", "download_failed", "waiting_for_document",
        }:
            return END
        return "retrieval"

    if last_role == "retrieval":
        if comparison_plan:
            schedule = comparison_schedule(state.get("reasoning_trace") or [])
            if schedule is not None:
                return "comparison_review" if schedule["status"] == "COMPLETED" else END
        return "reader"

    if last_role == "reader":
        if comparison_plan:
            schedule = comparison_schedule(state.get("reasoning_trace") or [])
            if schedule is not None:
                return "comparison_review" if schedule["status"] == "COMPLETED" else END
        return "summarizer"

    if last_role == "summarizer":
        return "fact_checker"

    if last_role == "fact_checker":
        if figure_context(state).get("explicit") and not state.get("rejected_claims"):
            return "writer"
        return "writer" if fact_check_accepted(state) else "escalation"

    if last_role == "comparison_review":
        trace = state.get("reasoning_trace") or []
        last_ev = trace[-1] if trace and isinstance(trace[-1], dict) else {}
        if last_ev.get("action") == "agent_handoff":
            rec = str(last_ev.get("recipient") or "").strip()
            return "comparison_review" if rec == "summarizer" else (rec if rec in _ROLE_TO_NODE else END)
        return END

    if last_role in {"writer", "escalation"}:
        return END

    return END


def supervisor_hub_step(
    state: AgentState,
    *,
    comparison_plan: bool = False,
    allow_scout_handoff: bool = False,
    parallel_retrieval: bool = False,
    arbiter_llm_fn: Optional[Callable[[str], Any]] = None,
    execution_mode: Optional[str] = None,
) -> dict[str, Any]:
    """
    Core Supervisor Hub node executed between every specialist turn in `supervisor_mode="hub"`.
    Consumes `state["last_result"]`, enforces Gatekeeper invariants, updates budgets & DAG,
    and enqueues the next typed `AgentTask` in `state["pending_tasks"]`.
    """
    round_idx = int(state.get("retry_count") or 0)
    last_result = state.get("last_result") if isinstance(state.get("last_result"), Mapping) else {}
    last_role = str(last_result.get("role") or "").strip()
    raw_decision = last_result.get("decision") if isinstance(last_result.get("decision"), Mapping) else None

    trace: list[dict[str, Any]] = []
    messages: list[dict[str, Any]] = []
    updates: dict[str, Any] = {}

    # Hard cycle safety stop so a misbehaving custom agent can never spin infinitely
    if _hub_steps_so_far(state) >= MAX_HUB_STEPS:
        logger.warning("[SupervisorHub] Reached MAX_HUB_STEPS=%d; escalating safely.", MAX_HUB_STEPS)
        esc_task = AgentTask(
            role="escalation",
            objective=_ROLE_OBJECTIVES["escalation"],
            sender="supervisor",
            reason="hub_step_limit_reached",
            round_index=round_idx,
        )
        trace.append({
            "node": "SupervisorHub",
            "action": "supervisor_hub_dispatched",
            "next_role": "escalation",
            "next_node": "escalation_node",
            "reason": "hub_step_limit_reached",
            "task_id": esc_task.task_id,
            "round_index": round_idx,
        })
        messages.append(make_agent_message(
            sender="supervisor",
            recipient="escalation",
            kind="task_assigned",
            action="escalate_or_finalize",
            reason="hub_step_limit_reached",
            task_id=esc_task.task_id,
            round_index=round_idx,
        ))
        return {
            "pending_tasks": [esc_task.to_dict()],
            "agent_messages": messages,
            "reasoning_trace": trace,
        }

    # Case 1: Evaluate specialist's explicit decision when present
    approved_handoff = None
    audit_event: dict[str, Any] | None = None
    evaluated_terminal = False

    if last_role and raw_decision:
        action = str(raw_decision.get("action") or "proceed").strip()
        if action == "escalate_or_finalize":
            evaluated_terminal = True
            _, audit_event = supervisor_evaluate_decision(
                raw_decision,
                {**state, "allow_scout_handoff": allow_scout_handoff},
                expected_sender=last_role,  # type: ignore[arg-type]
            )
            if audit_event is not None:
                trace.append(audit_event)
        elif action != "proceed":
            reviewed_state = {
                **state,
                "allow_scout_handoff": allow_scout_handoff,
            }
            approved_handoff, audit_event = supervisor_evaluate_decision(
                raw_decision,
                reviewed_state,  # type: ignore[arg-type]
                expected_sender=last_role,  # type: ignore[arg-type]
            )
            if audit_event is not None:
                trace.append(audit_event)
                messages.append(make_agent_message(
                    sender="supervisor",
                    recipient=str(audit_event.get("target_agent") or last_role),
                    kind="decision_approved" if audit_event.get("approved") else "decision_rejected",
                    action=str(audit_event.get("proposed_action") or action),
                    reason=str(audit_event.get("rejection_reason") or audit_event.get("reason") or ""),
                    task_id=audit_event.get("task_id"),
                    focus=raw_decision.get("parameters", {}).get("focus") if isinstance(raw_decision.get("parameters"), Mapping) else (),
                    summary={"approved": bool(audit_event.get("approved")), "budget_category": audit_event.get("budget_category")},
                    round_index=round_idx,
                ))

    # Case 2: Specialist proposed a corrective handoff AND Gatekeeper approved it!
    if approved_handoff is not None:
        next_role = approved_handoff.recipient
        attempt = round_idx + 1
        budget_cat = str((audit_event or {}).get("budget_category") or "revise")
        updated_budgets = next_handoff_budgets(
            state.get("handoff_budgets"),
            budget_cat,
            fallback_total=round_idx,
        )
        raw_params = raw_decision.get("parameters") if isinstance(raw_decision, Mapping) else {}
        task_meta: dict[str, Any] = {
            "signature": approved_handoff.as_trace()["signature"],
            "decision_id": (audit_event or {}).get("decision_id"),
        }
        if isinstance(raw_params, Mapping):
            if raw_params.get("strategy_hint"):
                task_meta["strategy_hint"] = str(raw_params["strategy_hint"])
            if raw_params.get("strategy_switched"):
                task_meta["strategy_switched"] = True

        next_task = AgentTask(
            role=next_role,
            objective=f"[{last_role} -> {next_role}] {approved_handoff.reason}",
            sender=last_role,
            reason=approved_handoff.reason,
            focus=approved_handoff.focus,
            source_ids=approved_handoff.source_ids,
            budget_category=budget_cat,
            round_index=attempt,
            task_id=approved_handoff.task_id or "",
            metadata=task_meta,
        )

        # Replan Research DAG if Fact-Checker requested new evidence for gaps
        if last_role == "fact_checker" and next_role in {"retrieval", "scout", "reader"}:
            gaps = state.get("evidence_gaps") or []
            schema = dict(state.get("document_schema") or {})
            if gaps and isinstance(schema.get("research_dag"), dict):
                replanned = replan_research_dag_for_gaps(schema.get("research_dag"), gaps)
                schema["research_dag"] = replanned
                updates["document_schema"] = schema
                updates["dag_round_index"] = int(replanned.get("round_index", state.get("dag_round_index") or 0))
                if replanned.get("added_gap_tasks", 0) > 0:
                    trace.append({
                        "agent": "task_scheduler",
                        "action": "research_dag_replanned",
                        "round_index": replanned["round_index"],
                        "added_gap_tasks": replanned["added_gap_tasks"],
                        "total_tasks": replanned["total_tasks"],
                    })

        handoff_trace = {
            **approved_handoff.as_trace(),
            "agent": last_role,
            "retry_count": attempt,
            "budget_category": budget_cat,
            "task_id": next_task.task_id,
        }
        if audit_event and audit_event.get("decision_id"):
            handoff_trace["decision_id"] = audit_event["decision_id"]
        trace.append(handoff_trace)
        trace.append({
            "node": "SupervisorHub",
            "action": "supervisor_hub_dispatched",
            "next_role": next_role,
            "next_node": _ROLE_TO_NODE[next_role],
            "reason": approved_handoff.reason,
            "policy_source": "agent_decision",
            "task_id": next_task.task_id,
            "round_index": attempt,
        })
        messages.append(make_agent_message(
            sender="supervisor",
            recipient=next_role,
            kind="task_assigned",
            action=str((raw_decision or {}).get("action") or approved_handoff.reason),
            reason=approved_handoff.reason,
            task_id=next_task.task_id,
            focus=approved_handoff.focus,
            summary={"retry_count": attempt, "budget_category": budget_cat},
            round_index=attempt,
        ))
        updates.update({
            "retry_count": attempt,
            "handoff_budgets": updated_budgets,
            "pending_tasks": [next_task.to_dict()],
            "agent_messages": messages,
            "reasoning_trace": trace,
        })
        if last_role in {"summarizer", "fact_checker", "writer"}:
            updates["draft_revision_needed"] = True
        return updates

    # Case 3: Specialist requested terminal stop/escalation
    if evaluated_terminal:
        if last_role == "scout":
            trace.append({
                "node": "SupervisorHub",
                "action": "supervisor_hub_dispatched",
                "next_role": END,
                "next_node": END,
                "reason": str((raw_decision or {}).get("reason") or "scout_terminal"),
                "policy_source": "agent_terminal",
                "round_index": round_idx,
            })
            return {"pending_tasks": [], "agent_messages": messages, "reasoning_trace": trace}
        esc_task = AgentTask(
            role="escalation",
            objective=_ROLE_OBJECTIVES["escalation"],
            sender=last_role or "supervisor",
            reason=str((raw_decision or {}).get("reason") or "escalate_or_finalize"),
            round_index=round_idx,
        )
        trace.append({
            "node": "SupervisorHub",
            "action": "supervisor_hub_dispatched",
            "next_role": "escalation",
            "next_node": "escalation_node",
            "reason": esc_task.reason,
            "policy_source": "agent_terminal",
            "task_id": esc_task.task_id,
            "round_index": round_idx,
        })
        messages.append(make_agent_message(
            sender="supervisor",
            recipient="escalation",
            kind="task_assigned",
            action="escalate_or_finalize",
            reason=esc_task.reason,
            task_id=esc_task.task_id,
            round_index=round_idx,
        ))
        return {
            "pending_tasks": [esc_task.to_dict()],
            "agent_messages": messages,
            "reasoning_trace": trace,
        }

    # Case 4: Deterministic fast-path (+ optional selective LLM arbitration on genuine ambiguity)
    next_role = _next_default_role(last_role, state, comparison_plan=comparison_plan)
    policy_source = "deterministic_fast_path"

    # Optional LLM arbitration ONLY when Fact-Checker has ambiguous conflict + remaining budget
    if (
        next_role == "escalation"
        and last_role == "fact_checker"
        and (arbiter_llm_fn is not None or is_gemini_permitted(execution_mode or get_run_settings().EXECUTION_MODE, role="planner"))
    ):
        alt_candidates = ["escalation", "writer"] if state.get("verified_claims") else ["escalation"]
        if _is_ambiguous_conflict(last_role, state, alt_candidates):
            chosen, rationale = _arbitrate_with_llm(
                state=state,
                candidates=alt_candidates,
                last_result=last_result,
                arbiter_llm_fn=arbiter_llm_fn,
                execution_mode=execution_mode,
            )
            if chosen in alt_candidates:
                next_role = chosen
                policy_source = "llm_arbiter"
                messages.append(make_agent_message(
                    sender="supervisor",
                    recipient=chosen,
                    kind="arbitration",
                    action="arbitrate_conflict",
                    reason=rationale or "resolved_ambiguous_conflict",
                    round_index=round_idx,
                ))

    if next_role == END or next_role not in _ROLE_TO_NODE:
        if last_role == "comparison_review" and state.get("answer_status") == "INSUFFICIENT_EVIDENCE":
            messages.append(make_agent_message(
                sender="supervisor",
                recipient="comparison_review",
                kind="workflow_terminated",
                action="incomplete_comparison",
                reason="insufficient_evidence",
                round_index=round_idx,
            ))
            return {
                "pending_tasks": [],
                "agent_messages": messages,
                "reasoning_trace": [],
            }
        trace.append({
            "node": "SupervisorHub",
            "action": "supervisor_hub_dispatched",
            "next_role": END,
            "next_node": END,
            "reason": "workflow_completed",
            "policy_source": policy_source,
            "round_index": round_idx,
        })
        return {
            "pending_tasks": [],
            "agent_messages": messages,
            "reasoning_trace": trace,
        }

    # Preserve comparison_review handoff request details if transitioning out of comparison_review
    handoff_focus: tuple[str, ...] = ()
    handoff_source_ids: tuple[str, ...] = ()
    handoff_reason = "normal_progression"
    handoff_meta: dict[str, Any] = {}
    handoff_budget_cat: str | None = None
    if last_role == "comparison_review":
        rtrace = state.get("reasoning_trace") or []
        last_ev = rtrace[-1] if rtrace and isinstance(rtrace[-1], dict) else {}
        if last_ev.get("action") == "agent_handoff":
            handoff_focus = tuple(last_ev.get("focus") or ())
            handoff_source_ids = tuple(last_ev.get("source_ids") or ())
            handoff_reason = str(last_ev.get("reason") or "comparison_handoff")
            handoff_meta = {
                "signature": last_ev.get("signature"),
                "decision_id": last_ev.get("decision_id"),
            }
            handoff_budget_cat = "revise"

    # Controlled parallel multi-task dispatch from Supervisor Hub when `parallel_retrieval=True`
    # and initial retrieval fans out across ≥2 independent documents or ≥2 independent sub-queries.
    if (
        parallel_retrieval
        and not comparison_plan
        and next_role == "retrieval"
        and round_idx == 0
        and handoff_reason == "normal_progression"
    ):
        selected_pdfs = list(dict.fromkeys(
            p for p in (state.get("pdf_paths") or ([state["pdf_path"]] if state.get("pdf_path") else []))
            if isinstance(p, str) and p.strip()
        ))
        sub_queries = list(dict.fromkeys(
            q.strip() for q in (state.get("sub_queries") or [])
            if isinstance(q, str) and q.strip()
        ))
        parallel_tasks: list[AgentTask] = []
        batch_kind = ""
        if len(sub_queries) >= 2:
            batch_kind = "sub_query"
            for idx_q, sub_q in enumerate(sub_queries[:3], start=1):
                parallel_tasks.append(AgentTask(
                    role="retrieval",
                    objective=f"Parallel sub-query retrieval #{idx_q}: {sub_q[:160]}",
                    sender="supervisor",
                    reason="parallel_subquery_batch",
                    focus=(sub_q,),
                    round_index=round_idx,
                    task_id=f"investigate_{idx_q}",
                    metadata={"parallel_batch": True, "batch_kind": "sub_query", "branch_index": idx_q, "target_query": sub_q},
                ))
        elif len(selected_pdfs) >= 2:
            batch_kind = "document"
            for idx_d, pdf_p in enumerate(selected_pdfs[:6], start=1):
                parallel_tasks.append(AgentTask(
                    role="retrieval",
                    objective=f"Parallel document retrieval #{idx_d}: {pdf_p[-80:]}",
                    sender="supervisor",
                    reason="parallel_document_batch",
                    focus=(str(state.get("query") or ""),),
                    source_ids=(pdf_p,),
                    round_index=round_idx,
                    task_id=f"doc_fanout_{idx_d}",
                    metadata={"parallel_batch": True, "batch_kind": "document", "branch_index": idx_d, "target_pdf": pdf_p},
                ))

        if len(parallel_tasks) >= 2:
            task_ids = [t.task_id for t in parallel_tasks]
            trace.append({
                "node": "SupervisorHub",
                "action": "supervisor_hub_dispatched",
                "next_role": "retrieval",
                "next_node": "parallel_dispatch_node",
                "reason": f"parallel_{batch_kind}_batch",
                "policy_source": "parallel_hub_scheduler",
                "batch_size": len(parallel_tasks),
                "task_ids": task_ids,
                "round_index": round_idx,
            })
            for ptask in parallel_tasks:
                messages.append(make_agent_message(
                    sender="supervisor",
                    recipient=ptask.role,
                    kind="task_assigned",
                    action=f"parallel_{batch_kind}_batch",
                    reason=ptask.reason,
                    task_id=ptask.task_id,
                    focus=ptask.focus,
                    summary={"parallel_batch": True, "batch_kind": batch_kind, "batch_size": len(parallel_tasks)},
                    round_index=round_idx,
                ))
            return {
                "pending_tasks": [t.to_dict() for t in parallel_tasks],
                "agent_messages": messages,
                "reasoning_trace": trace,
            }

    next_task = AgentTask(
        role=next_role,
        objective=_ROLE_OBJECTIVES.get(next_role, f"Execute {next_role}"),
        sender="supervisor",
        reason=handoff_reason,
        focus=handoff_focus,
        source_ids=handoff_source_ids,
        budget_category=handoff_budget_cat,
        round_index=round_idx,
        metadata=handoff_meta,
    )
    trace.append({
        "node": "SupervisorHub",
        "action": "supervisor_hub_dispatched",
        "next_role": next_role,
        "next_node": _ROLE_TO_NODE[next_role],
        "reason": handoff_reason,
        "policy_source": policy_source,
        "task_id": next_task.task_id,
        "round_index": round_idx,
    })
    messages.append(make_agent_message(
        sender="supervisor",
        recipient=next_role,
        kind="task_assigned",
        action="execute_task",
        reason=handoff_reason,
        task_id=next_task.task_id,
        focus=handoff_focus,
        round_index=round_idx,
    ))
    return {
        "pending_tasks": [next_task.to_dict()],
        "agent_messages": messages,
        "reasoning_trace": trace,
    }


def route_from_supervisor_hub(state: AgentState) -> str:
    """Conditional edge selector reading the next dispatched `AgentTask`(s) from `state['pending_tasks']`."""
    pending = state.get("pending_tasks") or []
    if not pending:
        return END
    if (
        len(pending) >= 2
        and all(
            isinstance(item, Mapping)
            and isinstance(item.get("metadata"), Mapping)
            and bool(item["metadata"].get("parallel_batch"))
            for item in pending
        )
    ):
        return "parallel_dispatch_node"
    first = pending[0]
    if not isinstance(first, Mapping):
        return END
    role = str(first.get("role") or "").strip()
    return _ROLE_TO_NODE.get(role, END)
