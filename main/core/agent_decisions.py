"""
Autonomous Per-Role Decision Engines for all 6 Specialist Agents.

Replaces centralized heuristic rules in `coordination.py` with role-owned `decide()` functions
so every specialist (`scout`, `retrieval`, `reader`, `summarizer`, `fact_checker`, `writer`)
inspects its own output and emits a mandatory, structured `AgentDecision` to the Supervisor Hub.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Mapping

from core.citation_audit import get_citation_audit
from core.coordination import (
    AgentDecision,
    _coerce_explicit_decision,
    _focus,
    _unread_chunks,
    fact_check_accepted,
    is_terminal_answer,
)
from core.figure_qa import figure_context
from core.state import AgentState


_SCOUT_TERMINAL_DECISIONS = frozenset(
    {"reject", "clarify", "no_relevant_papers", "download_failed", "waiting_for_document"}
)
_SCOUT_TERMINAL_TRACE_ACTIONS = frozenset(
    {"discovery_clarify", "discovery_rejected", "discovery_failed", "download_failed", "waiting_for_document"}
)
_WRITER_TERMINAL_STATUSES = frozenset(
    {
        "NOT_FOUND_IN_DOCUMENT",
        "CLARIFICATION_NEEDED",
        "INSUFFICIENT_EVIDENCE",
        "INCOMPLETE_COMPARISON",
        "OUT_OF_SCOPE",
    }
)
_GENERIC_ENTITY_SKIP = frozenset(
    {
        "paper", "study", "model", "method", "result", "results", "table", "figure",
        "system", "approach", "data", "dataset", "experiment", "value", "score",
        "tài liệu", "bài báo", "mô hình", "phương pháp", "kết quả", "bảng", "hình",
    }
)

# Exact capability map of non-proceed handoff targets each role's `decide()` can emit.
# Kept in lockstep with `ALLOWED_HANDOFF_TARGETS` (verified by regression tests so zero dead edges remain).
ROLE_DECISION_TARGETS: dict[str, set[str]] = {
    "supervisor": {"scout", "retrieval"},
    "scout": {"retrieval"},
    "retrieval": {"scout", "reader", "retrieval"},
    "reader": {"retrieval", "reader", "scout"},
    "summarizer": {"retrieval", "reader", "summarizer"},
    "fact_checker": {"retrieval", "reader", "summarizer", "scout"},
    "writer": {"summarizer", "fact_checker"},
}


def _current_round(state: Mapping[str, Any], update: Mapping[str, Any]) -> int:
    if update.get("retry_count") is not None:
        return max(0, int(update["retry_count"]))
    return max(0, int(state.get("retry_count") or 0))


def _task_id(task: Mapping[str, Any] | None) -> str | None:
    if isinstance(task, Mapping):
        tid = str(task.get("task_id") or "").strip()
        return tid or None
    return None


def _explicit_or_none(
    role: str,
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision | None:
    raw = update.get("agent_decision")
    if raw is None:
        return None
    merged = {**state, **update}
    dec = _coerce_explicit_decision(role, raw, merged)  # type: ignore[arg-type]
    if dec is not None and not dec.task_id:
        tid = _task_id(task)
        if tid:
            dec.task_id = tid
    return dec


def decide_scout(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision:
    """Scout autonomously decides whether to hand documents off to Retrieval or terminate cleanly."""
    explicit = _explicit_or_none("scout", state, update, task)
    if explicit is not None:
        return explicit

    round_idx = _current_round(state, update)
    tid = _task_id(task)
    scout_decision = str(update.get("scout_decision") or "").strip().lower()
    trace = update.get("reasoning_trace") or []
    last_action = str(trace[-1].get("action") or "") if trace and isinstance(trace[-1], dict) else ""

    if scout_decision in _SCOUT_TERMINAL_DECISIONS or last_action in _SCOUT_TERMINAL_TRACE_ACTIONS:
        return AgentDecision(
            source_agent="scout",
            action="escalate_or_finalize",
            target_agent=None,
            reason=scout_decision or last_action or "scout_terminal",
            round_index=round_idx,
            status="blocked",
            task_id=tid,
        )

    return AgentDecision(
        source_agent="scout",
        action="proceed",
        target_agent="retrieval",
        reason="documents_ready",
        round_index=round_idx,
        status="completed",
        task_id=tid,
    )


def _missing_required_entities(state: Mapping[str, Any], docs: list[dict[str, Any]]) -> list[str]:
    entities = [
        str(e).strip()
        for e in (state.get("required_entities") or [])
        if isinstance(e, str) and len(str(e).strip()) >= 4 and str(e).strip().lower() not in _GENERIC_ENTITY_SKIP
    ]
    if not entities:
        return []
    blob = "\n".join(
        str(d.get("text") or d.get("content") or "")
        for d in docs
        if isinstance(d, dict)
    ).casefold()
    missing: list[str] = []
    for ent in entities:
        ent_low = ent.casefold()
        if ent_low in blob:
            continue
        tokens = [t for t in re.findall(r"[a-z0-9_]+", ent_low) if len(t) >= 3]
        if tokens and all(tok in blob for tok in tokens):
            continue
        missing.append(ent)
    return missing


def decide_retrieval(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision:
    """
    Retrieval evaluates entity/corpus coverage on its own retrieved chunks:
      - If corpus yielded nothing / no new evidence and Scout is allowed -> propose Scout.
      - If strong required entities have zero coverage on initial pass and strategy switch
        has not run yet -> propose 1 bounded strategy-switch self-handoff (`retrieval -> retrieval`).
      - Otherwise -> proceed to Reader.
    """
    explicit = _explicit_or_none("retrieval", state, update, task)
    if explicit is not None:
        return explicit

    round_idx = _current_round(state, update)
    tid = _task_id(task)

    # Never override a terminal comparison refusal or explicit terminal answer_status
    if str(update.get("answer_status") or "").strip().upper() in _WRITER_TERMINAL_STATUSES:
        return AgentDecision(
            source_agent="retrieval",
            action="proceed",
            target_agent="reader",
            reason="retrieval_terminal_passthrough",
            round_index=round_idx,
            status="completed",
            task_id=tid,
        )
    if any(
        isinstance(e, dict) and e.get("action") == "comparison_task_schedule"
        for e in (update.get("reasoning_trace") or [])
    ):
        return AgentDecision(
            source_agent="retrieval",
            action="proceed",
            target_agent="reader",
            reason="comparison_schedule_completed",
            round_index=round_idx,
            status="completed",
            task_id=tid,
        )

    docs = [
        d for d in (update.get("retrieved_docs") if "retrieved_docs" in update else (state.get("retrieved_docs") or []))
        if isinstance(d, dict)
    ]
    no_new = bool(update.get("no_new_evidence", state.get("no_new_evidence", False)))
    allow_scout = bool(state.get("allow_scout_handoff"))
    query = str(state.get("resolved_query") or state.get("query") or "").strip()

    task_meta = dict(task.get("metadata") or {}) if isinstance(task, Mapping) else {}
    task_sender = str(task.get("sender") or "") if isinstance(task, Mapping) else ""
    already_switched = bool(task_meta.get("strategy_switched")) or any(
        isinstance(e, dict)
        and e.get("action") == "agent_handoff"
        and e.get("sender") == "retrieval"
        and e.get("recipient") == "retrieval"
        for e in (state.get("reasoning_trace") or [])
    )

    # 1. Corpus genuinely exhausted (empty docs + `no_new_evidence=True`) -> autonomously request Scout if enabled
    #    When `docs` is non-empty, always pass them to Reader/Summarizer/Fact-Checker first so Fact-Checker
    #    can formulate a gap-specific Scout query if needed.
    if len(docs) == 0 and no_new and allow_scout and task_sender != "scout":
        focus = _focus([query, *(state.get("required_entities") or [])], query or "additional_literature")
        return AgentDecision(
            source_agent="retrieval",
            action="discover_additional_literature",
            target_agent="scout",
            reason="corpus_exhausted_need_literature",
            focus=focus,
            round_index=round_idx,
            status="insufficient_evidence",
            task_id=tid,
        )

    # 2. Zero entity coverage on initial pass -> single bounded strategy switch (broad lexical + bilingual)
    has_scope = bool(state.get("pdf_paths") or state.get("pdf_path"))
    if (
        docs
        and has_scope
        and round_idx == 0
        and not no_new
        and not already_switched
        and state.get("enable_retrieval_strategy_switch", False)
    ):
        missing_ents = _missing_required_entities(state, docs)
        req_ents = [
            e for e in (state.get("required_entities") or [])
            if isinstance(e, str) and len(e.strip()) >= 4 and e.strip().lower() not in _GENERIC_ENTITY_SKIP
        ]
        if req_ents and len(missing_ents) == len(req_ents):
            return AgentDecision(
                source_agent="retrieval",
                action="request_more_evidence",
                target_agent="retrieval",
                reason="low_entity_coverage_switch_strategy",
                focus=_focus(missing_ents, query or "broad_search"),
                parameters={"strategy_hint": "broad_lexical_bilingual", "strategy_switched": True},
                round_index=round_idx,
                status="partial",
                task_id=tid,
            )

    return AgentDecision(
        source_agent="retrieval",
        action="proceed",
        target_agent="reader",
        reason="evidence_retrieved",
        round_index=round_idx,
        status="completed",
        task_id=tid,
    )


def decide_reader(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision:
    """
    Reader preserves its explicit missing-table / row-column-unit `AgentDecision` (from `reader_pipeline`),
    handles empty-extraction recovery (unread chunks -> retrieval -> scout), or proceeds to Summarizer.
    """
    if state.get("enable_reader_handoff") is not False:
        explicit = _explicit_or_none("reader", state, update, task)
        if explicit is not None:
            return explicit

    round_idx = _current_round(state, update)
    tid = _task_id(task)
    merged = {**state, **update}
    units = merged.get("source_units") or []
    docs = merged.get("retrieved_docs") or []
    no_new = bool(merged.get("no_new_evidence"))
    query = str(merged.get("resolved_query") or merged.get("query") or "").strip()

    if ("source_units" in update) and not units and state.get("enable_reader_handoff") is not False:
        unread = _unread_chunks(merged)
        if unread:
            return AgentDecision(
                source_agent="reader",
                action="request_unread_chunks",
                target_agent="reader",
                reason="unread_chunks",
                evidence_refs=unread,
                focus=_focus([query], "unread_chunks"),
                round_index=round_idx,
                status="needs_followup",
                task_id=tid,
            )
        if no_new and merged.get("allow_scout_handoff"):
            return AgentDecision(
                source_agent="reader",
                action="discover_additional_literature",
                target_agent="scout",
                reason="discover_additional_literature",
                focus=_focus([query], "missing_document"),
                round_index=round_idx,
                status="insufficient_evidence",
                task_id=tid,
            )
        if not docs and not no_new and (merged.get("pdf_paths") or merged.get("pdf_path")):
            return AgentDecision(
                source_agent="reader",
                action="request_more_evidence",
                target_agent="retrieval",
                reason="missing_evidence",
                focus=_focus([query], "empty_retrieval"),
                round_index=round_idx,
                status="needs_followup",
                task_id=tid,
            )

    return AgentDecision(
        source_agent="reader",
        action="proceed",
        target_agent="summarizer",
        reason="units_extracted",
        round_index=round_idx,
        status="completed",
        task_id=tid,
    )


def decide_summarizer(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision:
    """
    Summarizer translates its own Inner Critic output into an explicit `AgentDecision`
    (`revise_draft`, `request_unread_chunks`, `request_more_evidence`, or `proceed -> fact_checker`).
    """
    explicit = _explicit_or_none("summarizer", state, update, task)
    if explicit is not None:
        return explicit

    round_idx = _current_round(state, update)
    tid = _task_id(task)
    merged: AgentState = {**state, **update}  # type: ignore[assignment]
    action = merged.get("critic_action")

    if is_terminal_answer(merged) or action not in {"request_more_context", "rewrite"}:
        return AgentDecision(
            source_agent="summarizer",
            action="proceed",
            target_agent="fact_checker",
            reason="draft_ready_for_verification",
            round_index=round_idx,
            status="completed",
            task_id=tid,
        )

    feedback = merged.get("critic_feedback")
    feedback = feedback if isinstance(feedback, dict) else {}
    issues: list[Any] = []
    for key in ("missing_key_points", "unsupported_claims", "issues_found", "critique_rationale", "critique"):
        values = feedback.get(key) or []
        issues.extend(values if isinstance(values, list) else [values])
    focus = _focus(issues, str(merged.get("query") or ""))

    if action == "rewrite" and merged.get("source_units"):
        return AgentDecision(
            source_agent="summarizer",
            action="revise_draft",
            target_agent="summarizer",
            reason="revise_draft",
            focus=focus,
            round_index=round_idx,
            status="needs_followup",
            task_id=tid,
        )

    unread = _unread_chunks(merged)
    if unread:
        return AgentDecision(
            source_agent="summarizer",
            action="request_unread_chunks",
            target_agent="reader",
            reason="unread_chunks",
            evidence_refs=unread,
            focus=focus,
            round_index=round_idx,
            status="needs_followup",
            task_id=tid,
        )

    if merged.get("no_new_evidence"):
        return AgentDecision(
            source_agent="summarizer",
            action="proceed",
            target_agent="fact_checker",
            reason="no_new_evidence_proceed_to_verify",
            round_index=round_idx,
            status="partial",
            task_id=tid,
        )

    return AgentDecision(
        source_agent="summarizer",
        action="request_more_evidence",
        target_agent="retrieval",
        reason="missing_evidence",
        focus=focus,
        round_index=round_idx,
        status="needs_followup",
        task_id=tid,
    )


def _extract_specific_claim_issues(rejected_claims: list[Any], fallback_query: str) -> tuple[str, ...]:
    problems: list[str] = []
    for item in rejected_claims or []:
        if isinstance(item, dict):
            claim = str(item.get("claim_text") or item.get("claim") or item.get("text") or "").strip()
            reason = str(item.get("reason") or item.get("rationale") or "").strip()
            if claim and reason:
                problems.append(f"{claim} ({reason})")
            elif claim or reason:
                problems.append(claim or reason)
        elif isinstance(item, str) and item.strip():
            problems.append(item.strip())
    return _focus(problems, fallback_query)


def _extract_specific_gap_queries(
    evidence_gaps: list[Any],
    rejected_claims: list[Any],
    fallback_query: str,
) -> tuple[str, ...]:
    queries: list[str] = []
    for gap in evidence_gaps or []:
        if isinstance(gap, dict):
            sq = str(gap.get("suggested_query") or "").strip()
            aspect = str(gap.get("missing_aspect") or "").strip()
            if sq:
                queries.append(sq)
            elif aspect:
                queries.append(aspect)
    # Enrich with concrete rejected claim metrics/terms when evidence_gaps is sparse
    if not queries:
        for item in rejected_claims or []:
            if isinstance(item, dict):
                claim = str(item.get("claim_text") or item.get("claim") or item.get("text") or "").strip()
                if claim:
                    queries.append(claim)
    return _focus(queries, fallback_query)


def decide_fact_checker(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision:
    """
    Fact-Checker autonomously decides next action with claim-specific / gap-specific targets:
      - Verified -> proceed to Writer.
      - Contradiction or partial rejection with source_units -> revise_rejected_claims (Summarizer).
      - Missing evidence -> unread chunks (Reader) -> targeted search (Retrieval) -> Scout -> Escalate.
    """
    explicit = _explicit_or_none("fact_checker", state, update, task)
    if explicit is not None:
        return explicit

    round_idx = _current_round(state, update)
    tid = _task_id(task)
    merged: AgentState = {**state, **update}  # type: ignore[assignment]
    query = str(merged.get("query") or "")

    if fact_check_accepted(merged):
        return AgentDecision(
            source_agent="fact_checker",
            action="proceed",
            target_agent="writer",
            reason="claims_verified",
            round_index=round_idx,
            status="completed",
            task_id=tid,
        )

    if figure_context(merged).get("explicit") and not merged.get("rejected_claims"):
        return AgentDecision(
            source_agent="fact_checker",
            action="proceed",
            target_agent="writer",
            reason="visual_claims_verified",
            round_index=round_idx,
            status="completed",
            task_id=tid,
        )

    label = str(merged.get("fact_check_label") or "").lower()
    rejected = merged.get("rejected_claims") or []
    contradicted = label in {"contradict", "contradicted", "refuted"} or any(
        str(item.get("status", "")).lower() in {"contradict", "contradicted", "refuted"}
        for item in rejected
        if isinstance(item, dict)
    )

    if merged.get("source_units") and (contradicted or (rejected and merged.get("verified_claims"))):
        focus = _extract_specific_claim_issues(rejected, query)
        return AgentDecision(
            source_agent="fact_checker",
            action="revise_rejected_claims",
            target_agent="summarizer",
            reason="revise_rejected_claims",
            focus=focus,
            round_index=round_idx,
            status="needs_followup",
            task_id=tid,
        )

    gap_focus = _extract_specific_gap_queries(merged.get("evidence_gaps") or [], rejected, query)
    unread = _unread_chunks(merged)
    if unread:
        return AgentDecision(
            source_agent="fact_checker",
            action="request_unread_chunks",
            target_agent="reader",
            reason="unread_chunks",
            evidence_refs=unread,
            focus=gap_focus,
            round_index=round_idx,
            status="needs_followup",
            task_id=tid,
        )

    if merged.get("no_new_evidence"):
        if merged.get("allow_scout_handoff"):
            return AgentDecision(
                source_agent="fact_checker",
                action="discover_additional_literature",
                target_agent="scout",
                reason="discover_additional_literature",
                focus=gap_focus,
                round_index=round_idx,
                status="insufficient_evidence",
                task_id=tid,
            )
        return AgentDecision(
            source_agent="fact_checker",
            action="escalate_or_finalize",
            target_agent=None,
            reason="no_new_evidence_in_corpus",
            focus=gap_focus,
            round_index=round_idx,
            status="insufficient_evidence",
            task_id=tid,
        )

    return AgentDecision(
        source_agent="fact_checker",
        action="request_more_evidence",
        target_agent="retrieval",
        reason="missing_evidence",
        focus=gap_focus,
        round_index=round_idx,
        status="needs_followup",
        task_id=tid,
    )


def decide_writer(
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision:
    """
    Writer inspects the `citation_audit` produced alongside its draft report:
      - If claims failed citation audit while source_units exist (and run is not terminal/low-confidence),
        proposes `request_verification -> fact_checker` (when verification was missing) or
        `revise_draft -> summarizer` (to repair ungrounded claims).
      - Otherwise emits `proceed` to finalize publication.
    """
    explicit = _explicit_or_none("writer", state, update, task)
    if explicit is not None:
        return explicit

    round_idx = _current_round(state, update)
    tid = _task_id(task)
    merged = {**state, **update}
    answer_status = str(merged.get("answer_status") or "").strip().upper()
    confidence = str(merged.get("confidence") or "").strip().lower()

    if answer_status in _WRITER_TERMINAL_STATUSES or confidence == "low":
        return AgentDecision(
            source_agent="writer",
            action="proceed",
            target_agent=None,
            reason="report_published",
            round_index=round_idx,
            status="completed",
            task_id=tid,
        )

    audit = get_citation_audit(list(update.get("reasoning_trace") or []))
    claims_audit = audit.get("claims") if isinstance(audit, dict) else None
    source_units = merged.get("source_units") or []
    evidence_spans = merged.get("evidence_spans") or []
    has_audit_evidence_context = bool(
        evidence_spans
        or int(audit.get("unverified_evidence_count") or 0) > 0
        or (isinstance(merged.get("written_report_data"), dict) and merged["written_report_data"].get("citation_renderer"))
    )

    if isinstance(claims_audit, dict) and claims_audit and source_units:
        hard_failed_ids = [
            cid
            for cid, check in claims_audit.items()
            if isinstance(check, dict)
            and check.get("status") in {"semantic_or_math_conflict", "missing_numeric_evidence"}
        ]
        unresolved_ids = [
            cid
            for cid, check in claims_audit.items()
            if isinstance(check, dict)
            and check.get("status") == "unresolved_evidence"
            and not check.get("evidence_ids")
        ]
        failed_ids = hard_failed_ids or (unresolved_ids if has_audit_evidence_context else [])

        if failed_ids and (hard_failed_ids or len(failed_ids) * 2 >= len(claims_audit)):
            verified_claims = merged.get("verified_claims") or []
            ungrounded_texts: list[str] = []
            for pos, c in enumerate(verified_claims, start=1):
                if isinstance(c, dict):
                    cid = str(c.get("claim_id") or c.get("id") or f"claim_{pos}")
                    if cid in failed_ids:
                        txt = str(c.get("claim_text") or c.get("claim") or c.get("text") or "").strip()
                        if txt:
                            ungrounded_texts.append(txt)
            focus = _focus(ungrounded_texts, str(merged.get("query") or "citation_audit_failed"))

            fc_label = str(merged.get("fact_check_label") or "").strip().lower()
            if fc_label not in {"support", "supported"}:
                return AgentDecision(
                    source_agent="writer",
                    action="request_verification",
                    target_agent="fact_checker",
                    reason="citation_audit_requires_verification",
                    focus=focus,
                    round_index=round_idx,
                    status="needs_followup",
                    task_id=tid,
                )
            return AgentDecision(
                source_agent="writer",
                action="revise_draft",
                target_agent="summarizer",
                reason="citation_audit_failed",
                focus=focus,
                round_index=round_idx,
                status="needs_followup",
                task_id=tid,
            )

    return AgentDecision(
        source_agent="writer",
        action="proceed",
        target_agent=None,
        reason="report_published",
        round_index=round_idx,
        status="completed",
        task_id=tid,
    )


ROLE_DECIDERS: dict[str, Callable[[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any] | None], AgentDecision]] = {
    "scout": decide_scout,
    "retrieval": decide_retrieval,
    "reader": decide_reader,
    "summarizer": decide_summarizer,
    "fact_checker": decide_fact_checker,
    "writer": decide_writer,
}


def decide_for_role(
    role: str,
    state: Mapping[str, Any],
    update: Mapping[str, Any],
    task: Mapping[str, Any] | None = None,
) -> AgentDecision:
    """Dispatch to the role-owned autonomous decision engine for any of the 6 specialist roles."""
    fn = ROLE_DECIDERS.get(role)
    if fn is None:
        return AgentDecision(
            source_agent=role,  # type: ignore[arg-type]
            action="proceed",
            reason="completed",
            round_index=_current_round(state, update),
            status="completed",
            task_id=_task_id(task),
        )
    return fn(state, update, task)
