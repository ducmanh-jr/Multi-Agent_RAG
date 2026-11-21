"""Run a document-scoped comparison plan with a barrier before synthesis."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from dataclasses import dataclass, field
from itertools import zip_longest
from threading import Event
from typing import Any, Callable, Coroutine, TypeVar

from core.citation_audit import citation_location, claim_text, published_claim_id
from core.coordination import fact_check_accepted, is_terminal_answer
from core.research_tasks import ResearchTask, TaskStatus
from core.state import AgentState
from core.task_scheduler import ScheduleTimeoutError, TaskOutcome, TaskScheduler


COMPARISON_SCHEDULE = "comparison_task_schedule"
COMPARISON_REVIEW = "review_and_audit"
DEFAULT_COMPARISON_TIMEOUT_SECONDS = 120.0
MAX_COMPARISON_DOCUMENTS = 6
ComparisonReader = Callable[[str, list[dict[str, Any]]], dict[str, Any]]
_T = TypeVar("_T")


def run_coro_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Runs an async coroutine safely even if called from a thread with a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    ctx = copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()


@dataclass(frozen=True)
class ComparisonBudget:
    deadline_at: float
    stopped: Event = field(default_factory=Event)

    def remaining(self) -> float:
        if self.stopped.is_set():
            raise asyncio.CancelledError
        remaining = self.deadline_at - time.monotonic()
        if remaining <= 0:
            raise ScheduleTimeoutError("Comparison run timed out.")
        return remaining


_budget: ContextVar[ComparisonBudget | None] = ContextVar("comparison_budget", default=None)


def get_comparison_budget() -> ComparisonBudget | None:
    return _budget.get()


@contextmanager
def use_comparison_budget(budget: ComparisonBudget | None):
    token = _budget.set(budget)
    try:
        yield
    finally:
        _budget.reset(token)


@dataclass(frozen=True)
class ComparisonPlan:
    paths: tuple[str, ...]

    def __post_init__(self):
        if len(self.paths) < 2 or len(self.paths) > MAX_COMPARISON_DOCUMENTS:
            raise ValueError(f"ComparisonPlan requires 2..{MAX_COMPARISON_DOCUMENTS} documents.")

    def tasks(self, read_only: bool) -> tuple[ResearchTask, ...]:
        indices = range(1, len(self.paths) + 1)
        retrievals = () if read_only else tuple(ResearchTask(f"retrieve-{index}") for index in indices)
        readers = tuple(
            ResearchTask(f"read-{index}", depends_on_success=() if read_only else (f"retrieve-{index}",))
            for index in indices
        )
        return (*retrievals, *readers, ResearchTask("collect_evidence", depends_on_success=tuple(task.id for task in readers)))


def _check_ranking(ranking: list[dict[str, Any]]) -> str:
    locations = [citation_location(chunk) for chunk in ranking]
    sources = {location[0] for location in locations if location}
    if not locations or None in locations or len(sources) != 1:
        raise ValueError("No unambiguous evidence from the selected document.")
    return sources.pop()


async def compare_documents(
    plan: ComparisonPlan,
    rank_document: Callable[[str], list[dict[str, Any]]],
    read_document: ComparisonReader,
    top_k: int,
    *,
    retry_count: int,
    read_only: bool = False,
) -> dict[str, Any]:
    tasks = plan.tasks(read_only)
    collect = tasks[-1]
    read_paths = dict(zip(collect.depends_on_success, plan.paths, strict=True))
    retrieve_paths = {f"retrieve-{index}": path for index, path in enumerate(plan.paths, start=1)}
    budget = get_comparison_budget()
    remaining = budget.remaining() if budget else None
    scheduler = TaskScheduler(tasks, max_concurrency=min(len(plan.paths), 4), run_timeout_seconds=remaining)

    async def execute(task, dependencies):
        if budget:
            budget.remaining()
        if task.id in retrieve_paths:
            ranking = await asyncio.to_thread(rank_document, retrieve_paths[task.id])
            _check_ranking(ranking)
            return ranking
        if task.id in read_paths:
            path = read_paths[task.id]
            ranking = rank_document(path) if read_only else dependencies[task.depends_on_success[0]].output
            source = _check_ranking(ranking)
            read = await asyncio.to_thread(read_document, path, ranking)
            units = {(unit["id"], citation_location(unit)): unit for unit in read["source_units"]}
            pairs = [(chunk, units.get((chunk["id"], citation_location(chunk)))) for chunk in ranking]
            # Keep unread chunks available for handoff without spending a required source's first slot on them.
            paired = sorted(pairs, key=lambda pair: pair[1] is None)
            return source, paired, read
        reads = [outcome.output for outcome in dependencies.values()]
        required_sources = [source for source, _, _ in reads]
        if len(set(required_sources)) != len(plan.paths):
            raise ValueError("Selected documents have conflicting source identities.")
        blended = [pair for row in zip_longest(*(pairs for _, pairs, _ in reads)) for pair in row if pair is not None]
        selected = blended[:top_k]
        units = [unit for _, unit in selected if unit is not None]
        schema = {
            **reads[0][2]["document_schema"],
            "pages_covered": sorted({unit["page_no"] for unit in units}),
            "total_units": len(units),
        }
        return {"retrieved_docs": [chunk for chunk, _ in selected], "source_units": units,
                "document_schema": schema}, required_sources

    outcomes = await scheduler.run(execute)
    if budget:
        budget.remaining()
    gathered = outcomes[collect.id]
    evidence, sources = gathered.output if gathered.status == TaskStatus.COMPLETED else ({}, [])
    trace = [
        entry for task_id in read_paths if outcomes[task_id].status == TaskStatus.COMPLETED
        for entry in outcomes[task_id].output[2].get("reasoning_trace") or []
    ]
    trace.append(_schedule_event(
        tasks, outcomes, node="reader_node" if read_only else "retrieval_node",
        stage="read_handoff" if read_only else "retrieve_and_read", sources=sources,
        remaining=remaining, retry_count=retry_count,
    ))
    if gathered.status != TaskStatus.COMPLETED:
        retrieval_failed = any(
            task.id in retrieve_paths and outcomes[task.id].status != TaskStatus.COMPLETED for task in tasks
        )
        refusal = incomplete_comparison("required_retrieval_incomplete" if retrieval_failed else "required_reading_incomplete")
        return {**refusal, "retrieved_docs": [], "source_units": [], "reasoning_trace": [*trace, *refusal["reasoning_trace"]]}
    return {**evidence, "reasoning_trace": trace}


def _schedule_event(
    tasks: tuple[ResearchTask, ...], outcomes: dict[str, TaskOutcome], *,
    node: str, stage: str, sources: list[str], remaining: float | None, retry_count: int,
) -> dict[str, Any]:
    return {"node": node, "action": COMPARISON_SCHEDULE, "details": {
        "status": outcomes[tasks[-1].id].status.value, "required_sources": sources,
        "remaining_seconds_at_start": remaining, "stage": stage, "retry_count": retry_count,
        "tasks": [
            {
                "id": task.id, "status": outcomes[task.id].status.value,
                "depends_on_success": list(task.depends_on_success),
                "blocked_by": list(outcomes[task.id].blocked_by), "attempts": outcomes[task.id].attempts,
            }
            for task in tasks
        ],
    }}


def _claim_keys(claims: list[dict[str, Any]]) -> set[tuple[str, str]]:
    return {(published_claim_id(claim, index), claim_text(claim).strip()) for index, claim in enumerate(claims, 1)}


class _ReviewStopped(ValueError):
    """The role completed, but its result cannot unlock the dependent task."""


async def review_comparison(
    state: AgentState, sources: list[str],
    run_specialist: Callable[[str, AgentState], dict[str, Any]],
    audit_report: Callable[[AgentState], dict[str, Any]],
) -> dict[str, Any]:
    tasks = (
        ResearchTask("collect_evidence", status=TaskStatus.COMPLETED),
        ResearchTask("summarize", depends_on_success=("collect_evidence",)),
        ResearchTask("fact_check", depends_on_success=("summarize",)),
        ResearchTask("audit_report", depends_on_success=("fact_check",)),
    )
    budget = get_comparison_budget()
    remaining = budget.remaining() if budget else None
    scheduler = TaskScheduler(tasks, max_concurrency=1, run_timeout_seconds=remaining)
    runners = {
        "summarize": lambda view: run_specialist("summarizer", view),
        "fact_check": lambda view: run_specialist("fact_checker", view),
        "audit_report": audit_report,
    }
    changes: dict[str, Any] = {}
    trace = []
    handoff = None

    async def execute(task, _dependencies):
        nonlocal handoff
        if budget:
            budget.remaining()
        view = {**state, **changes, "reasoning_trace": [*(state.get("reasoning_trace") or []), *trace]}
        update = await asyncio.to_thread(runners[task.id], view)
        if budget:
            budget.remaining()
        events = list(update.get("reasoning_trace") or [])
        if events and events[-1].get("action") == "agent_handoff":
            handoff = events.pop()
        changes.update({key: value for key, value in update.items() if key != "reasoning_trace"})
        trace.extend(events)
        if handoff:
            raise _ReviewStopped("handoff_requested")
        if task.id == "summarize":
            claims = update.get("claims") or []
            keys = _claim_keys(claims)
            if (not (update.get("answer") or "").strip() or not claims or not update.get("evidence_spans")
                    or len({claim_id for claim_id, _ in keys}) != len(claims) or not all(text for _, text in keys)
                    or is_terminal_answer(update) or update.get("answer_status") == "INSUFFICIENT_EVIDENCE"
                    or update.get("critic_action") in {"rewrite", "request_more_context"}):
                raise _ReviewStopped("synthesis_incomplete")
        elif task.id == "fact_check":
            verified = update.get("verified_claims") or []
            expected = _claim_keys(view["claims"])
            if (not fact_check_accepted(update) or len(verified) != len(expected) or _claim_keys(verified) != expected
                    or any(str(claim.get("status", "")).lower() not in {"support", "supported"} for claim in verified)):
                raise _ReviewStopped("fact_check_incomplete")
        elif not (update.get("final_report") or "").strip() or update.get("answer_status") == "INSUFFICIENT_EVIDENCE":
            reason = next((event["reason"] for event in events if event.get("action") == "comparison_evidence_incomplete"), None)
            raise _ReviewStopped(reason or "report_audit_incomplete")

    outcomes = await scheduler.run(execute)
    if budget:
        budget.remaining()
    trace.append(_schedule_event(
        tasks, outcomes, node="comparison_review_node", stage=COMPARISON_REVIEW, sources=sources,
        remaining=remaining, retry_count=state.get("retry_count", 0),
    ))
    if handoff:
        return {**changes, "reasoning_trace": [*trace, handoff]}
    for task_id, outcome in outcomes.items():
        if outcome.status in {TaskStatus.FAILED, TaskStatus.CANCELLED}:
            error = outcome.error
            reason = str(error) if isinstance(error, _ReviewStopped) else f"{task_id}_{outcome.status.value.lower()}"
            refusal = incomplete_comparison(reason)
            trace = [event for event in trace if event.get("action") not in {
                "citation_audit", "answer_figures", "comparison_evidence_incomplete",
            }]
            return {**refusal, "reasoning_trace": [*trace, *refusal["reasoning_trace"]]}
    return {**changes, "reasoning_trace": trace}


def comparison_schedule(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(trace):
        if event.get("action") == COMPARISON_SCHEDULE:
            return event["details"]
    return None


def incomplete_comparison(reason: str) -> AgentState:
    message = "Chưa đủ bằng chứng đã xác minh từ cả hai tài liệu để đưa ra kết luận so sánh."
    return {
        "answer_status": "INSUFFICIENT_EVIDENCE", "answer": message, "short_answer": message,
        "final_report": message, "confidence": "low", "verified_claims": [], "claims": [],
        "rejected_claims": [], "evidence_spans": [], "fact_check_label": "not_enough_info",
        "reasoning_trace": [{"action": "comparison_evidence_incomplete", "reason": reason}],
    }
