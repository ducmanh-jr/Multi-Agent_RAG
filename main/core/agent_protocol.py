"""
Typed Multi-Agent Communication Protocol (`AgentTask`, `AgentResult`, `AgentMessage`).

Provides structured, checkpoint-safe contracts between the Supervisor Hub and all 6 specialist
agents (`scout`, `retrieval`, `reader`, `summarizer`, `fact_checker`, `writer`) plus
`comparison_review`. Keeps blackboard messages compact (metadata & short focus strings only,
never raw chunk or PDF blobs) so SQLite checkpoints stay small and fast.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import time
from typing import Any, Iterable, Mapping

from core.coordination import AgentDecision


MAX_FOCUS_ITEMS = 4
MAX_FOCUS_LEN = 220
MAX_SOURCE_IDS = 12


def _clean_focus(items: Iterable[Any] | None) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in items or ():
        if isinstance(item, str):
            text = " ".join(item.split())[:MAX_FOCUS_LEN]
            if text and text not in cleaned:
                cleaned.append(text)
        if len(cleaned) >= MAX_FOCUS_ITEMS:
            break
    return tuple(cleaned)


def _clean_ids(items: Iterable[Any] | None) -> tuple[str, ...]:
    cleaned: list[str] = []
    for item in items or ():
        val = str(item).strip()
        if val and val not in cleaned:
            cleaned.append(val)
        if len(cleaned) >= MAX_SOURCE_IDS:
            break
    return tuple(cleaned)


def _compact_metadata(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only JSON-safe scalar/short-list metadata in protocol payloads."""
    if not isinstance(raw, Mapping):
        return {}
    out: dict[str, Any] = {}
    for k, v in raw.items():
        key = str(k)
        if v is None or isinstance(v, (bool, int, float)):
            out[key] = v
        elif isinstance(v, str):
            out[key] = v[:240]
        elif isinstance(v, (list, tuple)):
            compact_list: list[Any] = []
            for elem in list(v)[:8]:
                if elem is None or isinstance(elem, (bool, int, float)):
                    compact_list.append(elem)
                elif isinstance(elem, str):
                    compact_list.append(elem[:160])
            out[key] = compact_list
    return out


@dataclass(frozen=True)
class AgentTask:
    """Typed work order dispatched by the Supervisor Hub to a specialist agent."""

    role: str
    objective: str
    sender: str = "supervisor"
    reason: str = "scheduled"
    focus: tuple[str, ...] = ()
    source_ids: tuple[str, ...] = ()
    budget_category: str | None = None
    round_index: int = 0
    task_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "role", str(self.role or "").strip())
        object.__setattr__(self, "objective", " ".join(str(self.objective or "").split())[:260])
        object.__setattr__(self, "sender", str(self.sender or "supervisor").strip())
        object.__setattr__(self, "reason", str(self.reason or "scheduled").strip())
        object.__setattr__(self, "focus", _clean_focus(self.focus))
        object.__setattr__(self, "source_ids", _clean_ids(self.source_ids))
        object.__setattr__(self, "round_index", max(0, int(self.round_index or 0)))
        object.__setattr__(self, "metadata", _compact_metadata(self.metadata))
        if not self.task_id:
            basis = json.dumps(
                [
                    self.sender,
                    self.role,
                    self.reason,
                    self.round_index,
                    list(self.focus),
                    list(self.source_ids),
                    self.metadata.get("branch_index"),
                    self.metadata.get("target_pdf"),
                    self.metadata.get("target_query"),
                ],
                ensure_ascii=False,
            )
            tid = f"task_{self.role}_{sha256(basis.encode('utf-8')).hexdigest()[:10]}"
            object.__setattr__(self, "task_id", tid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "objective": self.objective,
            "sender": self.sender,
            "reason": self.reason,
            "focus": list(self.focus),
            "source_ids": list(self.source_ids),
            "budget_category": self.budget_category,
            "round_index": self.round_index,
            "metadata": dict(self.metadata),
        }

    def to_handoff_request(self) -> dict[str, Any] | None:
        """Convert a corrective task into the legacy handoff request dict consumed by `prepare_handoff_input`."""
        if not self.budget_category and self.reason in {"scheduled", "initial_plan", "normal_progression"}:
            return None
        payload: dict[str, Any] = {
            "action": "agent_handoff",
            "sender": self.sender,
            "recipient": self.role,
            "reason": self.reason,
            "focus": list(self.focus),
            "source_ids": list(self.source_ids),
            "task_id": self.task_id,
            "retry_count": self.round_index,
        }
        if self.metadata.get("signature"):
            payload["signature"] = self.metadata["signature"]
        if self.metadata.get("decision_id"):
            payload["decision_id"] = self.metadata["decision_id"]
        if self.metadata.get("strategy_hint"):
            payload["strategy_hint"] = self.metadata["strategy_hint"]
        return payload

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentTask":
        return cls(
            task_id=str(raw.get("task_id") or ""),
            role=str(raw.get("role") or ""),
            objective=str(raw.get("objective") or ""),
            sender=str(raw.get("sender") or "supervisor"),
            reason=str(raw.get("reason") or "scheduled"),
            focus=tuple(raw.get("focus") or ()),
            source_ids=tuple(raw.get("source_ids") or ()),
            budget_category=raw.get("budget_category"),
            round_index=int(raw.get("round_index") or 0),
            metadata=dict(raw.get("metadata") or {}),
        )


@dataclass(frozen=True)
class AgentResult:
    """Compact structured outcome returned by a specialist agent to the Supervisor Hub."""

    role: str
    status: str
    decision: dict[str, Any]
    task_id: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    tools_used: tuple[str, ...] = ()
    round_index: int = 0

    def __post_init__(self):
        object.__setattr__(self, "role", str(self.role or "").strip())
        object.__setattr__(self, "status", str(self.status or "completed").strip())
        object.__setattr__(self, "summary", _compact_metadata(self.summary))
        object.__setattr__(self, "tools_used", tuple(str(t) for t in (self.tools_used or ())[:8] if str(t).strip()))
        object.__setattr__(self, "round_index", max(0, int(self.round_index or 0)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "role": self.role,
            "status": self.status,
            "summary": dict(self.summary),
            "decision": dict(self.decision),
            "tools_used": list(self.tools_used),
            "round_index": self.round_index,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "AgentResult":
        return cls(
            task_id=raw.get("task_id"),
            role=str(raw.get("role") or ""),
            status=str(raw.get("status") or "completed"),
            summary=dict(raw.get("summary") or {}),
            decision=dict(raw.get("decision") or {}),
            tools_used=tuple(raw.get("tools_used") or ()),
            round_index=int(raw.get("round_index") or 0),
        )


def make_agent_message(
    *,
    sender: str,
    recipient: str,
    kind: str,
    action: str,
    reason: str = "",
    task_id: str | None = None,
    focus: Iterable[str] | None = None,
    summary: Mapping[str, Any] | None = None,
    round_index: int = 0,
) -> dict[str, Any]:
    """Create a compact, checkpoint-safe blackboard message for `state['agent_messages']`."""
    ts = round(time.time(), 3)
    cleaned_focus = list(_clean_focus(focus))
    basis = f"{sender}:{recipient}:{kind}:{action}:{task_id or ''}:{round_index}:{ts}"
    return {
        "message_id": f"msg_{sha256(basis.encode('utf-8')).hexdigest()[:10]}",
        "sender": str(sender),
        "recipient": str(recipient),
        "kind": str(kind),
        "task_id": task_id,
        "action": str(action),
        "reason": str(reason or "")[:200],
        "focus": cleaned_focus,
        "summary": _compact_metadata(summary),
        "round_index": max(0, int(round_index or 0)),
        "timestamp": ts,
    }


def compact_result_summary(
    role: str,
    state: Mapping[str, Any],
    update: Mapping[str, Any],
) -> dict[str, Any]:
    """Extract lightweight scalar telemetry for `AgentResult.summary` without storing raw chunk/PDF blobs."""
    merged_units = update.get("source_units") if "source_units" in update else state.get("source_units")
    merged_docs = update.get("retrieved_docs") if "retrieved_docs" in update else state.get("retrieved_docs")
    summary: dict[str, Any] = {}

    if role == "scout":
        pdf_paths = update.get("pdf_paths") or state.get("pdf_paths") or []
        summary["scout_decision"] = str(update.get("scout_decision") or ("retained" if pdf_paths else "none"))
        summary["pdf_count"] = len(pdf_paths)
        if update.get("answer_status"):
            summary["answer_status"] = str(update["answer_status"])
    elif role == "retrieval":
        summary["retrieved_count"] = len(merged_docs or [])
        summary["no_new_evidence"] = bool(update.get("no_new_evidence", state.get("no_new_evidence", False)))
        if update.get("retrieval_strategy"):
            summary["retrieval_strategy"] = str(update["retrieval_strategy"])
    elif role == "reader":
        summary["source_units_count"] = len(merged_units or [])
        summary["retrieved_count"] = len(merged_docs or [])
    elif role == "summarizer":
        summary["has_answer"] = bool(str(update.get("answer") or "").strip())
        summary["claims_count"] = len(update.get("claims") or [])
        summary["critic_action"] = str(update.get("critic_action") or "approve")
        if update.get("answer_status"):
            summary["answer_status"] = str(update["answer_status"])
    elif role == "fact_checker":
        summary["fact_check_label"] = str(update.get("fact_check_label") or "not_enough_info")
        summary["verified_count"] = len(update.get("verified_claims") or [])
        summary["rejected_count"] = len(update.get("rejected_claims") or [])
        summary["gaps_count"] = len(update.get("evidence_gaps") or [])
    elif role == "writer":
        report = str(update.get("final_report") or "")
        summary["report_length"] = len(report)
        summary["confidence"] = str(update.get("confidence") or state.get("confidence") or "")
        if update.get("answer_status"):
            summary["answer_status"] = str(update["answer_status"])
    elif role == "comparison_review":
        summary["answer_status"] = str(update.get("answer_status") or "")
        summary["report_length"] = len(str(update.get("final_report") or ""))

    return summary


def normalize_decision_dict(
    role: str,
    decision: AgentDecision | Mapping[str, Any] | None,
    *,
    round_index: int = 0,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Ensure every specialist emits a valid, serialized `AgentDecision` dict (defaulting to `proceed`)."""
    if isinstance(decision, AgentDecision):
        data = decision.to_dict()
        if task_id and not data.get("task_id"):
            data["task_id"] = task_id
        return data
    if isinstance(decision, Mapping):
        coerced = AgentDecision(
            source_agent=str(decision.get("source_agent") or role),  # type: ignore[arg-type]
            action=decision.get("action") or decision.get("next_action") or "proceed",
            target_agent=decision.get("target_agent") or decision.get("recipient"),
            reason=str(decision.get("reason") or ""),
            evidence_refs=decision.get("evidence_refs") or decision.get("evidence_ids") or (),
            parameters=decision.get("parameters") if isinstance(decision.get("parameters"), Mapping) else {},
            suggested_queries=decision.get("suggested_queries"),
            focus=decision.get("focus"),
            round_index=int(decision.get("round_index") if decision.get("round_index") is not None else round_index),
            status=str(decision.get("status") or "completed"),
            decision_id=decision.get("decision_id"),
            task_id=decision.get("task_id") or task_id,
            signature=decision.get("signature"),
        )
        return coerced.to_dict()
    default_dec = AgentDecision(
        source_agent=role,  # type: ignore[arg-type]
        action="proceed",
        reason="completed",
        round_index=round_index,
        status="completed",
        task_id=task_id,
    )
    return default_dec.to_dict()
