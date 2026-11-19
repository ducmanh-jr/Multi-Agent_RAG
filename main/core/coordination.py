"""Bounded handoffs requested by the Summarizer's critic and the Fact Checker."""

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Any, Iterable, Literal

from core.config import settings
from core.state import AgentState


Sender = Literal["summarizer", "fact_checker"]
Recipient = Literal["retrieval", "reader", "summarizer"]


@dataclass(frozen=True)
class Handoff:
    sender: Sender
    recipient: Recipient
    reason: str
    focus: tuple[str, ...]
    source_ids: tuple[str, ...] = ()

    def as_trace(self) -> dict[str, Any]:
        # Ordering of independent gaps or chunks is immaterial when detecting a repeated request.
        identity = json.dumps([
            self.sender, self.recipient, self.reason,
            sorted(item.casefold() for item in self.focus), sorted(self.source_ids),
        ], ensure_ascii=False)
        return {
            "action": "agent_handoff",
            "node": self.sender,
            "sender": self.sender,
            "recipient": self.recipient,
            "reason": self.reason,
            "focus": list(self.focus),
            "source_ids": list(self.source_ids),
            "signature": sha256(identity.encode("utf-8")).hexdigest(),
        }


def reflection_limit(state: AgentState) -> int:
    cap = max(0, settings.MAX_ORCHESTRATOR_RETRY)
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


def decide_handoff(sender: Sender, state: AgentState) -> Handoff | None:
    if (state.get("retry_count") or 0) >= reflection_limit(state):
        return None
    request = _summarizer_request(state) if sender == "summarizer" else _checker_request(state)
    if request is None:
        return None
    signature = request.as_trace()["signature"]
    repeated = any(
        isinstance(event, dict) and event.get("action") == "agent_handoff" and event.get("signature") == signature
        for event in state.get("reasoning_trace") or []
    )
    return None if repeated else request
