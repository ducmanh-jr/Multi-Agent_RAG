"""
Bounded handoffs and Supervised Agent Autonomy (AgentDecision contract - M3).
Agents propose structured next actions via `AgentDecision`; the Research Supervisor
validates policy permissions, reflection/retry budget, DAG task budget, and anti-loop
signatures before dispatching any handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Any, Iterable, Literal, Mapping

from core.config import get_run_settings
from core.state import AgentState


Sender = Literal["supervisor", "scout", "retrieval", "reader", "summarizer", "fact_checker", "writer"]
Recipient = Literal["scout", "retrieval", "reader", "summarizer", "fact_checker", "writer"]
DecisionAction = Literal[
    "request_more_evidence",
    "request_unread_chunks",
    "revise_draft",
    "revise_rejected_claims",
    "request_verification",
    "discover_additional_literature",
    "escalate_or_finalize",
    "proceed",
]

VALID_SENDERS: frozenset[str] = frozenset(
    {"supervisor", "scout", "retrieval", "reader", "summarizer", "fact_checker", "writer"}
)
VALID_DECISION_ACTIONS: frozenset[str] = frozenset(
    {
        "request_more_evidence",
        "request_unread_chunks",
        "revise_draft",
        "revise_rejected_claims",
        "request_verification",
        "discover_additional_literature",
        "escalate_or_finalize",
        "proceed",
    }
)
VALID_DECISION_STATUSES: frozenset[str] = frozenset(
    {"completed", "proposed", "needs_followup", "partial", "insufficient_evidence", "blocked"}
)

# Policy matrix: which target agents each of the 6 specialist roles (+ supervisor) may propose
ALLOWED_HANDOFF_TARGETS: dict[str, set[str]] = {
    "supervisor": {"scout", "retrieval"},
    "scout": {"retrieval"},
    "retrieval": {"scout", "reader", "retrieval"},
    "reader": {"retrieval", "reader", "scout"},
    "summarizer": {"retrieval", "reader", "summarizer"},
    "fact_checker": {"retrieval", "reader", "summarizer", "scout"},
    "writer": {"summarizer", "fact_checker"},
}

ACTION_TO_HANDOFF_REASON: dict[str, str] = {
    "request_more_evidence": "missing_evidence",
    "request_unread_chunks": "unread_chunks",
    "revise_draft": "revise_draft",
    "revise_rejected_claims": "revise_rejected_claims",
    "request_verification": "request_verification",
    "discover_additional_literature": "discover_additional_literature",
}

HANDOFF_BUDGET_CATEGORIES: dict[str, str] = {
    "request_more_evidence": "evidence",
    "request_unread_chunks": "read",
    "revise_draft": "revise",
    "revise_rejected_claims": "verify",
    "request_verification": "verify",
    "discover_additional_literature": "scout",
}

PRE_VERIFICATION_SENDERS: frozenset[str] = frozenset({"scout", "retrieval", "reader", "summarizer"})


def handoff_budget_category(decision: "AgentDecision") -> str:
    """Map an AgentDecision to its isolated handoff budget bucket."""
    if decision.action in HANDOFF_BUDGET_CATEGORIES:
        cat = HANDOFF_BUDGET_CATEGORIES[decision.action]
        if decision.source_agent in {"fact_checker", "writer"} and cat == "revise":
            return "verify"
        return cat
    if decision.target_agent == "scout":
        return "scout"
    if decision.target_agent == "reader":
        return "read"
    if decision.target_agent == "retrieval":
        return "evidence"
    if decision.source_agent in {"fact_checker", "writer"}:
        return "verify"
    return "revise"


def next_handoff_budgets(
    current: Mapping[str, Any] | None,
    category: str,
    *,
    fallback_total: int = 0,
) -> dict[str, int]:
    """Return an updated copy of per-category handoff usage counters plus total."""
    budgets: dict[str, int] = {
        "evidence": 0,
        "read": 0,
        "revise": 0,
        "verify": 0,
        "scout": 0,
        "total": max(0, int(fallback_total or 0)),
    }
    if isinstance(current, Mapping):
        for key, val in current.items():
            try:
                budgets[str(key)] = max(0, int(val))
            except (TypeError, ValueError):
                continue
    budgets[category] = budgets.get(category, 0) + 1
    cat_sum = sum(v for k, v in budgets.items() if k != "total")
    budgets["total"] = max(budgets.get("total", 0) + 1, cat_sum)
    return budgets


def _compute_handoff_signature(
    sender: str,
    recipient: str,
    reason: str,
    focus: Iterable[str],
    source_ids: Iterable[str] = (),
) -> str:
    identity = json.dumps(
        [
            sender,
            recipient,
            reason,
            sorted(str(item).casefold() for item in focus),
            sorted(str(sid) for sid in source_ids),
        ],
        ensure_ascii=False,
    )
    return sha256(identity.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Handoff:
    sender: Sender
    recipient: Recipient
    reason: str
    focus: tuple[str, ...]
    source_ids: tuple[str, ...] = ()
    task_id: str | None = None

    def as_trace(self) -> dict[str, Any]:
        # Ordering of independent gaps or chunks is immaterial when detecting a repeated request.
        sig = _compute_handoff_signature(
            self.sender, self.recipient, self.reason, self.focus, self.source_ids
        )
        payload: dict[str, Any] = {
            "action": "agent_handoff",
            "node": self.sender,
            "sender": self.sender,
            "recipient": self.recipient,
            "reason": self.reason,
            "focus": list(self.focus),
            "source_ids": list(self.source_ids),
            "signature": sig,
        }
        if self.task_id:
            payload["task_id"] = self.task_id
        return payload


class AgentDecision:
    """
    Structured proposal emitted by a specialist agent under Supervised Agent Autonomy.
    Supports both canonical M3/M4 fields (`decision_id`, `task_id`, `source_agent`, `action`, `reason`,
    `evidence_refs`, `parameters`, `signature`, `round_index`) and directional aliases
    (`status`, `next_action`, `target_agent`, `evidence_ids`, `suggested_queries`).
    """

    def __init__(
        self,
        *,
        source_agent: Sender = "fact_checker",
        action: DecisionAction | str | None = None,
        next_action: DecisionAction | str | None = None,
        target_agent: Recipient | str | None = None,
        reason: str = "",
        evidence_refs: Iterable[str] | None = None,
        evidence_ids: Iterable[str] | None = None,
        parameters: Mapping[str, Any] | None = None,
        suggested_queries: Iterable[str] | None = None,
        focus: Iterable[str] | None = None,
        round_index: int = 0,
        status: str = "completed",
        decision_id: str | None = None,
        task_id: str | None = None,
        signature: str | None = None,
    ):
        self.source_agent: Sender = source_agent
        self.status = str(status or "completed")
        resolved_action = str(action or next_action or "proceed").strip()
        self.action = resolved_action
        self.next_action = resolved_action
        self.reason = str(reason or "").strip()
        self.round_index = max(0, int(round_index or 0))

        refs = list(evidence_refs if evidence_refs is not None else (evidence_ids or []))
        self.evidence_refs: tuple[str, ...] = tuple(str(r) for r in refs if str(r).strip())
        self.evidence_ids: list[str] = list(self.evidence_refs)

        params = dict(parameters or {})
        if suggested_queries is not None:
            params["suggested_queries"] = [str(q).strip() for q in suggested_queries if str(q).strip()]
        if focus is not None:
            params["focus"] = [str(f).strip() for f in focus if str(f).strip()]
        if target_agent is not None:
            params["target_agent"] = str(target_agent).strip()

        resolved_target = str(target_agent or params.get("target_agent") or "").strip() or None
        if resolved_target is None and resolved_action not in {"proceed", "escalate_or_finalize"}:
            if resolved_action == "request_unread_chunks":
                resolved_target = "reader"
            elif resolved_action in {"revise_draft", "revise_rejected_claims"}:
                resolved_target = "summarizer"
            elif resolved_action == "request_verification":
                resolved_target = "fact_checker"
            elif resolved_action == "discover_additional_literature":
                resolved_target = "scout"
            elif resolved_action == "request_more_evidence":
                resolved_target = "retrieval"
        self.target_agent: str | None = resolved_target
        if resolved_target:
            params["target_agent"] = resolved_target

        focus_items = params.get("focus") or params.get("suggested_queries") or ([self.reason] if self.reason else [])
        self.focus: tuple[str, ...] = _focus(focus_items, self.reason or "follow_up")
        params["focus"] = list(self.focus)
        self.parameters: dict[str, Any] = params

        handoff_reason = ACTION_TO_HANDOFF_REASON.get(resolved_action, self.reason or resolved_action)
        source_ids = self.evidence_refs if resolved_target == "reader" else ()
        self.handoff_reason = handoff_reason
        self.canonical_signature = _compute_handoff_signature(
            self.source_agent,
            self.target_agent or "none",
            handoff_reason,
            self.focus,
            source_ids,
        )
        self.signature = signature or self.canonical_signature
        self.decision_id = decision_id or f"dec_{self.canonical_signature[:12]}"
        self.task_id: str | None = str(task_id or params.get("task_id") or "").strip() or None

    def compute_canonical_signature(self) -> str:
        """Recompute the canonical Supervisor anti-loop signature from immutable decision fields."""
        source_ids = self.evidence_refs if self.target_agent == "reader" else ()
        return _compute_handoff_signature(
            self.source_agent,
            self.target_agent or "none",
            self.handoff_reason,
            self.focus,
            source_ids,
        )

    def to_handoff(self) -> Handoff | None:
        if self.action == "proceed" or not self.target_agent:
            return None
        source_ids = self.evidence_refs if self.target_agent == "reader" else ()
        return Handoff(
            sender=self.source_agent,
            recipient=self.target_agent,  # type: ignore[arg-type]
            reason=self.handoff_reason,
            focus=self.focus,
            source_ids=source_ids,
            task_id=self.task_id,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "decision_id": self.decision_id,
            "source_agent": self.source_agent,
            "status": self.status,
            "action": self.action,
            "next_action": self.next_action,
            "target_agent": self.target_agent,
            "reason": self.reason,
            "evidence_refs": list(self.evidence_refs),
            "parameters": dict(self.parameters),
            "signature": self.signature,
            "round_index": self.round_index,
        }
        if self.task_id:
            payload["task_id"] = self.task_id
        return payload


def reflection_limit(state: AgentState) -> int:
    cap = max(0, get_run_settings().MAX_ORCHESTRATOR_RETRY)
    requested = state.get("max_reflection_loops")
    if requested is None:
        requested = state.get("max_iterations")
    return cap if requested is None else min(cap, max(0, int(requested)))


def is_terminal_answer(state: AgentState) -> bool:
    """The checker must still assess a refusal, including one that contradicts the evidence."""
    return state.get("answer_status") in {"NOT_FOUND_IN_DOCUMENT", "CLARIFICATION_NEEDED"}


def fact_check_accepted(state: AgentState) -> bool:
    label = str(state.get("fact_check_label", "")).lower()
    return label in {"support", "supported"} and not state.get("rejected_claims")


def _focus(items: Iterable[Any], fallback: str) -> tuple[str, ...]:
    result: list[str] = []
    for value in items:
        if isinstance(value, str):
            text = " ".join(value.split())[:300]
            if text and text not in result:
                result.append(text)
    return tuple(result[:4]) or (fallback,)


def _unread_chunks(state: AgentState) -> tuple[str, ...]:
    read_ids = {unit.get("id") for unit in state.get("source_units") or [] if isinstance(unit, dict)}
    unread = []
    for chunk in state.get("retrieved_docs") or []:
        if not isinstance(chunk, dict) or not (chunk.get("text") or chunk.get("content")):
            continue
        chunk_id = chunk.get("id") or chunk.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id and chunk_id not in read_ids:
            unread.append(chunk_id)
    return tuple(dict.fromkeys(unread))[:8]


def _context_request(sender: Sender, state: AgentState, focus: tuple[str, ...]) -> Handoff | None:
    unread = _unread_chunks(state)
    if unread:
        return Handoff(sender, "reader", "unread_chunks", focus, unread)
    if state.get("no_new_evidence"):
        # When local documents are exhausted and auto_scout is enabled, Fact Checker (or Reader/Retrieval)
        # can propose asking Scout to discover additional literature on the specific evidence gap.
        if sender in {"fact_checker", "reader", "retrieval"} and state.get("allow_scout_handoff"):
            return Handoff(sender, "scout", "discover_additional_literature", focus)
        return None
    return Handoff(sender, "retrieval", "missing_evidence", focus)


def _summarizer_request(state: AgentState) -> Handoff | None:
    action = state.get("critic_action")
    if is_terminal_answer(state) or action not in {"request_more_context", "rewrite"}:
        return None
    feedback = state.get("critic_feedback")
    feedback = feedback if isinstance(feedback, dict) else {}
    issues = []
    for key in ("missing_key_points", "unsupported_claims", "issues_found", "critique_rationale", "critique"):
        values = feedback.get(key) or []
        issues.extend(values if isinstance(values, list) else [values])
    focus = _focus(issues, state.get("query", ""))
    if action == "rewrite" and state.get("source_units"):
        return Handoff("summarizer", "summarizer", "revise_draft", focus)
    return _context_request("summarizer", state, focus)


def _checker_request(state: AgentState) -> Handoff | None:
    if fact_check_accepted(state):
        return None
    label = str(state.get("fact_check_label", "")).lower()
    rejected = state.get("rejected_claims") or []
    contradicted = label in {"contradict", "contradicted", "refuted"} or any(
        str(item.get("status", "")).lower() in {"contradict", "contradicted", "refuted"}
        for item in rejected if isinstance(item, dict)
    )
    if state.get("source_units") and (contradicted or (rejected and state.get("verified_claims"))):
        problems = []
        for item in rejected:
            if isinstance(item, dict):
                claim = item.get("claim_text") or item.get("claim") or item.get("text")
                reason = item.get("reason")
                problems.append(f"{claim} ({reason})" if claim and reason else claim or reason)
            else:
                problems.append(item)
        return Handoff("fact_checker", "summarizer", "revise_rejected_claims", _focus(problems, state.get("query", "")))
    gaps = [
        gap.get("suggested_query") or gap.get("missing_aspect")
        for gap in state.get("evidence_gaps") or [] if isinstance(gap, dict)
    ]
    return _context_request("fact_checker", state, _focus(gaps, state.get("query", "")))


def _coerce_explicit_decision(sender: Sender, raw: Any, state: AgentState) -> AgentDecision | None:
    if isinstance(raw, AgentDecision):
        return raw
    if isinstance(raw, Mapping):
        raw_source = raw.get("source_agent")
        declared_source = str(raw_source).strip() if raw_source is not None and str(raw_source).strip() else sender
        raw_round = raw.get("round_index")
        round_idx = int(state.get("retry_count") or 0) if raw_round is None else int(raw_round)
        return AgentDecision(
            source_agent=declared_source,  # type: ignore[arg-type]
            action=raw.get("action") or raw.get("next_action") or "proceed",
            target_agent=raw.get("target_agent") or raw.get("recipient"),
            reason=str(raw.get("reason") or ""),
            evidence_refs=raw.get("evidence_refs") or raw.get("evidence_ids") or (),
            parameters=raw.get("parameters") if isinstance(raw.get("parameters"), Mapping) else {},
            suggested_queries=raw.get("suggested_queries"),
            focus=raw.get("focus"),
            round_index=round_idx,
            status=str(raw.get("status") or "completed"),
            decision_id=raw.get("decision_id"),
            task_id=raw.get("task_id"),
            signature=raw.get("signature"),
        )
    return None


def _handoff_to_decision(handoff: Handoff, state: AgentState) -> AgentDecision:
    reason_to_action: dict[str, DecisionAction] = {
        "missing_evidence": "request_more_evidence",
        "unread_chunks": "request_unread_chunks",
        "revise_draft": "revise_draft",
        "revise_rejected_claims": "revise_rejected_claims",
        "discover_additional_literature": "discover_additional_literature",
    }
    action = reason_to_action.get(handoff.reason, "request_more_evidence")
    return AgentDecision(
        source_agent=handoff.sender,
        action=action,
        target_agent=handoff.recipient,
        reason=handoff.reason,
        evidence_refs=handoff.source_ids,
        focus=handoff.focus,
        round_index=int(state.get("retry_count") or 0),
        signature=handoff.as_trace()["signature"],
    )


def propose_agent_decision(sender: Sender, state: AgentState) -> AgentDecision | None:
    """
    Constructs a structured `AgentDecision` proposal from the agent's output state.
    Delegates to role-owned autonomous decision engines in `core.agent_decisions` so
    `coordination.py` acts purely as the Supervisor Gatekeeper without duplicated rules.
    """
    explicit = _coerce_explicit_decision(sender, state.get("agent_decision"), state)
    if explicit is not None:
        return None if explicit.action == "proceed" else explicit

    if sender in {"summarizer", "fact_checker"}:
        from core.agent_decisions import decide_for_role

        proposed = decide_for_role(sender, state, {})
        return None if proposed.action in {"proceed", "escalate_or_finalize"} else proposed
    return None


def supervisor_evaluate_decision(
    decision: AgentDecision | Mapping[str, Any] | None,
    state: AgentState,
    *,
    expected_sender: Sender | None = None,
) -> tuple[Handoff | None, dict[str, Any] | None]:
    """
    Research Supervisor gatekeeper for Supervised Agent Autonomy:
    Validates sender identity, decision schema (action, status, round_index), policy permissions,
    reflection budget, DAG task budget, and Supervisor-computed canonical anti-loop signatures
    before approving an `AgentDecision` into an executable `Handoff`.
    """
    if decision is None:
        return None, None
    if not isinstance(decision, AgentDecision):
        if isinstance(decision, Mapping):
            fallback_sender: Sender = expected_sender or str(decision.get("source_agent") or "fact_checker")  # type: ignore[assignment]
            decision = _coerce_explicit_decision(fallback_sender, decision, state)
            if decision is None:
                return None, None
        else:
            return None, None
    if decision.action == "proceed":
        return None, None

    current_round = int(state.get("retry_count") or 0)
    limit = reflection_limit(state)
    allowed_targets = ALLOWED_HANDOFF_TARGETS.get(decision.source_agent, set())
    target = decision.target_agent or ""
    canonical_signature = decision.compute_canonical_signature()
    budget_cat = handoff_budget_category(decision)
    raw_budgets = state.get("handoff_budgets") if isinstance(state.get("handoff_budgets"), Mapping) else {}
    raw_limits = state.get("handoff_budget_limits") if isinstance(state.get("handoff_budget_limits"), Mapping) else {}
    total_used = max(current_round, int(raw_budgets.get("total", 0) or 0))
    cat_used = int(raw_budgets.get(budget_cat, 0) or 0)
    verify_used = int(raw_budgets.get("verify", 0) or 0)
    explicit_total_limit = raw_limits.get("total")
    effective_total_limit = (
        min(limit, max(0, int(explicit_total_limit)))
        if explicit_total_limit is not None
        else limit
    )
    explicit_cat_limit = raw_limits.get(budget_cat)
    cat_limit = (
        max(0, int(explicit_cat_limit))
        if explicit_cat_limit is not None
        else (
            max(1, effective_total_limit - 1)
            if effective_total_limit >= 2
            and decision.source_agent in PRE_VERIFICATION_SENDERS
            and budget_cat in {"evidence", "read", "revise", "scout"}
            else effective_total_limit
        )
    )
    # Reserve at least 1 slot for post-verification repair (fact_checker / writer) whenever total limit >= 2
    # so pre-verification loops (reader/summarizer) cannot starve contradiction repair.
    verifier_reserved = (
        effective_total_limit >= 2
        and decision.source_agent in PRE_VERIFICATION_SENDERS
        and verify_used == 0
        and explicit_cat_limit is None
        and total_used >= (effective_total_limit - 1)
    )

    disabled_senders = {
        str(item).strip().lower()
        for item in (state.get("disabled_agent_handoffs") or [])
        if isinstance(item, str) and item.strip()
    }

    rejection_reason: str | None = None
    if expected_sender is not None and decision.source_agent != expected_sender:
        rejection_reason = f"source_agent_mismatch:{decision.source_agent}!={expected_sender}"
    elif decision.source_agent not in VALID_SENDERS:
        rejection_reason = f"invalid_source_agent:{decision.source_agent}"
    elif decision.action not in VALID_DECISION_ACTIONS:
        rejection_reason = f"invalid_decision_action:{decision.action}"
    elif decision.status not in VALID_DECISION_STATUSES:
        rejection_reason = f"invalid_decision_status:{decision.status}"
    elif decision.action == "escalate_or_finalize":
        rejection_reason = None
    elif decision.source_agent in disabled_senders:
        rejection_reason = f"agent_handoff_disabled:{decision.source_agent}"
    elif total_used >= effective_total_limit:
        rejection_reason = "reflection_budget_exhausted"
    elif decision.round_index != current_round:
        rejection_reason = f"round_index_mismatch:{decision.round_index}!={current_round}"
    elif target not in allowed_targets:
        rejection_reason = f"policy_target_not_permitted:{decision.source_agent}->{target}"
    elif target == "scout" and not state.get("allow_scout_handoff"):
        rejection_reason = "scout_handoff_disabled"
    elif target == "retrieval" and state.get("no_new_evidence"):
        rejection_reason = "no_new_evidence_in_corpus"
    elif cat_used >= cat_limit:
        rejection_reason = f"category_budget_exhausted:{budget_cat}"
    elif verifier_reserved:
        rejection_reason = "verifier_budget_reserved"
    else:
        schema = state.get("document_schema")
        dag_info = schema.get("research_dag") if isinstance(schema, Mapping) else None
        if isinstance(dag_info, Mapping) and dag_info.get("budget_exhausted") and target in {"retrieval", "scout"}:
            rejection_reason = "dag_task_budget_exhausted"

    if rejection_reason is None and decision.action != "escalate_or_finalize":
        repeated = any(
            isinstance(event, dict)
            and event.get("action") == "agent_handoff"
            and event.get("signature") == canonical_signature
            for event in state.get("reasoning_trace") or []
        )
        if repeated:
            rejection_reason = "duplicate_handoff_signature"

    approved = rejection_reason is None
    audit_event = {
        "action": "supervisor_decision_evaluated",
        "node": "SupervisorGatekeeper",
        "decision_id": decision.decision_id,
        "task_id": decision.task_id,
        "source_agent": decision.source_agent,
        "proposed_action": decision.action,
        "proposed_status": decision.status,
        "proposed_round_index": decision.round_index,
        "target_agent": decision.target_agent,
        "budget_category": budget_cat,
        "reason": decision.reason or decision.handoff_reason,
        "round_index": current_round,
        "approved": approved,
        "decision_state": "APPROVED" if approved else "REJECTED",
        "rejection_reason": rejection_reason,
        "signature": canonical_signature,
    }
    if not approved or decision.action == "escalate_or_finalize":
        return None, audit_event
    return decision.to_handoff(), audit_event


def evaluate_agent_handoff(sender: Sender, state: AgentState) -> tuple[Handoff | None, dict[str, Any] | None]:
    """Propose and evaluate a handoff for `sender`, returning both approved Handoff and Supervisor audit trace."""
    decision = propose_agent_decision(sender, state)
    return supervisor_evaluate_decision(decision, state, expected_sender=sender)


def decide_handoff(sender: Sender, state: AgentState) -> Handoff | None:
    """Evaluates whether `sender`'s proposal is approved by the Supervisor for handoff."""
    approved_handoff, _ = evaluate_agent_handoff(sender, state)
    return approved_handoff
