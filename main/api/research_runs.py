"""Shared execution and persistence for queued, synchronous and streaming research."""

import asyncio
import hashlib
import time
import uuid
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from api.citations import file_sha256, pin_citation_provenance
from api.figures import pin_answer_figures
from api.run_manifest import manifest_event
from core import db
from core.citation_audit import get_citation_audit
from core.comparison_plan import (
    COMPARISON_REVIEW, DEFAULT_COMPARISON_TIMEOUT_SECONDS, ComparisonBudget, comparison_schedule,
    incomplete_comparison, use_comparison_budget,
)
from core.config import RunConfig
from core.graph import build_research_rag_graph
from core.supervisor import supervisor_planner
from core.task_scheduler import ScheduleTimeoutError
from core.token_usage import RunTokenLedger, capture_token_usage
from core.worker_observations import capture_worker_observations, drain_worker_observations
from utils.logger import logger


WAITING_MESSAGE = "Vui lòng chọn một tài liệu trong kho hoặc bật Auto-Scout để bắt đầu nghiên cứu."
WAITING_REPORT = f"# Thông báo Hệ thống\n\n> **Trạng thái:** `WAITING_FOR_DOCUMENT`\n\n{WAITING_MESSAGE}"
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}


def _source_path(value: str | None) -> Path | None:
    if not value:
        return None
    try:
        path = Path(value).resolve(strict=True)
        return path if path.suffix.lower() in DOCUMENT_EXTENSIONS and path.is_file() else None
    except (OSError, ValueError, RuntimeError):
        return None


def _document_snapshot(path: Path, document: dict[str, Any] | None = None) -> dict[str, Any] | None:
    digest = file_sha256(path)
    if digest is None:
        return None
    document = document or {}
    document_id = document.get("id")
    source_id = document_id or "local_" + hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:24]
    return {
        "id": source_id,
        "name": document.get("name") or path.name,
        "path": str(path),
        "sha256": digest,
        "catalogued": bool(document_id),
    }


def _resolve_paths(paths: list[str]) -> list[dict[str, Any]]:
    catalog = {}
    for document in db.list_documents():
        path = _source_path(document.get("path"))
        if path is not None:
            catalog.setdefault(path, document)
    unique = dict.fromkeys(path for value in paths if (path := _source_path(value)) is not None)
    return [snapshot for path in unique if (snapshot := _document_snapshot(path, catalog.get(path))) is not None]


def _resolve_selection(ids: tuple[str, ...], path: str | None) -> list[dict[str, Any]]:
    if not ids:
        return _resolve_paths([path]) if path else []
    selected = []
    seen = set()
    for document_id in dict.fromkeys(ids):
        document = db.get_document(document_id)
        source = _source_path(document.get("path")) if document else None
        if source is not None and source not in seen:
            snapshot = _document_snapshot(source, document)
            if snapshot is not None:
                selected.append(snapshot)
                seen.add(source)
    return selected


def run_answer_status(record: dict[str, Any]) -> str | None:
    for event in reversed(record.get("reasoning_trace") or []):
        if event.get("action") == "run_outcome":
            return event.get("answer_status")
        if event.get("action") == "waiting_for_document":
            return "WAITING_FOR_DOCUMENT"
    return None


@dataclass(frozen=True)
class ResearchRequest:
    query: str
    config: RunConfig
    mode: str | None = None
    document_ids: tuple[str, ...] = ()
    pdf_path: str | None = None
    auto_scout: bool = False
    max_reflection_loops: int | None = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    session_id: str | None = None
    parallel_retrieval: bool = False
    comparison_plan: bool = False
    comparison_timeout_seconds: float = DEFAULT_COMPARISON_TIMEOUT_SECONDS
    token_budget: int | None = None


@dataclass(frozen=True)
class RunStep:
    node: str
    output: dict[str, Any]
    traces: list[dict[str, Any]]


@dataclass(frozen=True)
class ResearchResult:
    record: dict[str, Any]
    state: dict[str, Any]


@dataclass(frozen=True)
class ResearchRun:
    record: dict[str, Any]
    task: asyncio.Task[ResearchResult] | None
    steps: asyncio.Queue[RunStep | None] | None

    async def result(self) -> ResearchResult:
        if self.task is None:
            return ResearchResult(self.record, {})
        return await asyncio.shield(self.task)


@dataclass(frozen=True)
class _ComparisonExecution:
    budget: ComparisonBudget
    cancelled: asyncio.Event = field(default_factory=asyncio.Event)


class ResearchRunService:
    def __init__(self):
        self.tasks: dict[str, asyncio.Task[ResearchResult]] = {}
        self._comparisons: dict[str, _ComparisonExecution] = {}
        self._token_ledgers: dict[str, RunTokenLedger] = {}

    async def cancel(self, run_id: str) -> str | None:
        status = await asyncio.to_thread(db.cancel_run, run_id)
        execution = self._comparisons.get(run_id)
        if status == "cancelled" and execution:
            execution.cancelled.set()
            execution.budget.stopped.set()
        if status == "cancelled" and (ledger := self._token_ledgers.get(run_id)):
            ledger.close()
        return status

    async def start(self, request: ResearchRequest, *, stream: bool = False) -> ResearchRun:
        # Keep acceptance and scheduling together if the HTTP caller disconnects during admission.
        return await asyncio.shield(self._admit(request, stream))

    async def _admit(self, request: ResearchRequest, stream: bool) -> ResearchRun:
        sources = await asyncio.to_thread(_resolve_selection, request.document_ids, request.pdf_path)
        scout_enabled = request.auto_scout and not sources
        waiting = not sources and not scout_enabled
        admission = manifest_event(
            request.config, mode=request.mode, auto_scout=request.auto_scout, scout_enabled=scout_enabled,
            reflection_loops=request.max_reflection_loops, parallel_retrieval=request.parallel_retrieval,
            comparison_plan=request.comparison_plan,
            token_budget=request.token_budget,
        )
        traces = [admission]
        comparison_query = (
            admission["configuration"]["comparison_plan_enabled"]
            and supervisor_planner.plan_query(request.query).is_comparative_query
        )
        requested_doc_count = len(set(request.document_ids))
        missing_comparison_source = comparison_query and requested_doc_count >= 2 and len(sources) < requested_doc_count
        comparison = comparison_query and not missing_comparison_source and (
            scout_enabled or supervisor_planner.plan_document_comparison({
                "pdf_paths": [source["path"] for source in sources], "is_comparative_query": True,
            }) is not None
        )
        if comparison:
            admission["configuration"]["comparison_timeout_seconds"] = request.comparison_timeout_seconds
        report = None
        if missing_comparison_source:
            admission["configuration"]["auto_scout_enabled"] = False
            refusal = incomplete_comparison("selected_source_unavailable")
            report = refusal["final_report"]
            traces.extend(refusal["reasoning_trace"])
            traces.append({"action": "run_outcome", "answer_status": refusal["answer_status"]})
        elif waiting:
            report = WAITING_REPORT
            traces.append({"action": "waiting_for_document", "message": WAITING_MESSAGE})
        terminal = waiting or missing_comparison_source
        record = await asyncio.to_thread(
            db.save_run,
            run_id=f"run_{uuid.uuid4().hex[:10]}",
            question=request.query,
            status="completed" if terminal else "processing",
            final_report=report,
            reasoning_trace=traces,
            session_id=request.session_id,
            documents=sources,
        )
        steps = asyncio.Queue() if stream else None
        if terminal:
            if steps is not None:
                steps.put_nowait(None)
            return ResearchRun(record, None, steps)

        execution = None
        if comparison:
            execution = _ComparisonExecution(ComparisonBudget(time.monotonic() + request.comparison_timeout_seconds))
            self._comparisons[record["id"]] = execution
        with (
            capture_worker_observations(),
            use_comparison_budget(execution.budget if execution else None),
            capture_token_usage(request.token_budget) as ledger,
        ):
            self._token_ledgers[record["id"]] = ledger
            task = asyncio.create_task(self._execute(record["id"], request, sources, traces, steps, execution, ledger))
        self.tasks[record["id"]] = task
        return ResearchRun(record, task, steps)

    async def _execute(
        self,
        run_id: str,
        request: ResearchRequest,
        sources: list[dict[str, Any]],
        traces: list[dict[str, Any]],
        steps: asyncio.Queue[RunStep | None] | None,
        execution: _ComparisonExecution | None,
        ledger: RunTokenLedger,
    ) -> ResearchResult:
        paths = [source["path"] for source in sources]
        state: dict[str, Any] = {
            "query": request.query,
            "pdf_path": paths[0] if paths else None,
            "pdf_paths": paths,
            "retry_count": 0,
            "chat_history": request.chat_history,
        }
        if request.max_reflection_loops is not None:
            state["max_reflection_loops"] = request.max_reflection_loops
        deadline_at = execution.budget.deadline_at if execution else None
        remaining = max(0, deadline_at - time.monotonic()) if deadline_at is not None else None
        timeout = asyncio.timeout(remaining)
        watcher = None
        if execution:
            owner = asyncio.current_task()
            assert owner is not None

            async def interrupt():
                await execution.cancelled.wait()
                owner.cancel()

            watcher = asyncio.create_task(interrupt())
        try:
            async with timeout:
                if await asyncio.to_thread(db.run_is_processing, run_id):
                    graph = await asyncio.to_thread(
                        build_research_rag_graph, mode=request.mode, run_config=request.config,
                        auto_scout=request.auto_scout and not sources, parallel_retrieval=request.parallel_retrieval,
                        comparison_plan=request.comparison_plan,
                    )
                    start_time = time.monotonic()
                    async with aclosing(graph.astream(state.copy(), stream_mode="updates")) as updates:
                        while await asyncio.to_thread(db.run_is_processing, run_id):
                            chunk = await anext(updates, None)
                            traces.extend(ledger.drain())
                            ledger.check()
                            if chunk is None:
                                break
                            now = time.monotonic()
                            elapsed = round(max(0.1, now - start_time), 1)
                            start_time = now
                            traces.extend(drain_worker_observations())
                            for node, output in chunk.items():
                                if not isinstance(output, dict):
                                    continue
                                added = None
                                if node == "scout_node" and not sources and output.get("pdf_path"):
                                    found_paths = output.get("pdf_paths") or [output["pdf_path"]]
                                    added = await asyncio.to_thread(_resolve_paths, found_paths)
                                    if not added:
                                        raise RuntimeError("Tài liệu Auto-Scout vừa tìm không còn truy cập được.")
                                    sources = added
                                    paths = [source["path"] for source in sources]
                                node_traces = [
                                    deepcopy(trace) for trace in output.get("reasoning_trace") or []
                                    if isinstance(trace, dict)
                                ]
                                catalogued = [source for source in sources if source["catalogued"]]
                                pin_answer_figures(node_traces, catalogued, paths)
                                for trace in node_traces:
                                    trace.update(duration=f"{elapsed}s", duration_sec=elapsed, timestamp=time.time())
                                traces.extend(node_traces)
                                state.update(output)
                                progress = [
                                    event for event in traces
                                    if not (execution or ledger.limit is not None)
                                    or event.get("action") not in {"citation_audit", "answer_figures"}
                                ]
                                accepted = await asyncio.to_thread(
                                    db.update_run_result, run_id, status="processing",
                                    reasoning_trace=progress, documents=added, deadline_at=deadline_at,
                                )
                                if not accepted:
                                    break
                                if steps is not None:
                                    steps.put_nowait(RunStep(node, output, node_traces))

                traces.extend(drain_worker_observations())
                traces.extend(ledger.drain())
                ledger.check()
                spans = state.get("evidence_spans") or []
                # The stream may still be reading the node traces queued during execution.
                traces = deepcopy(traces)
                await asyncio.to_thread(pin_citation_provenance, traces, state, sources)
                comparison = comparison_schedule(traces) if request.comparison_plan else None
                if comparison and state.get("answer_status") != "INSUFFICIENT_EVIDENCE":
                    cited = {ref.get("catalog_id") for ref in get_citation_audit(traces).get("references", [])}
                    reason = None
                    if comparison.get("stage") != COMPARISON_REVIEW or comparison.get("status") != "COMPLETED":
                        reason = "comparison_review_incomplete"
                    elif not cited.issuperset(source["id"] for source in sources):
                        reason = "source_snapshot_unconfirmed"
                    if reason:
                        refusal = incomplete_comparison(reason)
                        state.update(refusal)
                        spans = []
                        traces = [event for event in traces if event.get("action") not in {"citation_audit", "answer_figures"}]
                        traces.extend(refusal["reasoning_trace"])
                if state.get("answer_status"):
                    traces.append({"action": "run_outcome", "answer_status": state["answer_status"]})
            accepted = await asyncio.to_thread(
                db.update_run_result, run_id, status="completed", final_report=state.get("final_report") or "",
                verified_claims=state.get("verified_claims"), rejected_claims=state.get("rejected_claims"),
                evidence_spans=spans, reasoning_trace=traces, deadline_at=deadline_at,
            )
            if not accepted and deadline_at is not None and time.monotonic() >= deadline_at:
                raise ScheduleTimeoutError("Comparison run timed out.")
        except asyncio.CancelledError:
            if execution is None or not execution.cancelled.is_set():
                await asyncio.to_thread(db.cancel_run, run_id)
                raise
        except Exception as exc:
            traces.extend(drain_worker_observations())
            traces.extend(ledger.drain())
            if budget_stop := ledger.stop_status:
                traces = [
                    event for event in traces
                    if event.get("action") not in {"citation_audit", "answer_figures", "run_outcome"}
                ]
                traces.append({"action": "run_outcome", "answer_status": budget_stop})
                error = budget_stop
            elif execution and (timeout.expired() or isinstance(exc, ScheduleTimeoutError)):
                execution.budget.stopped.set()
                traces = [
                    event for event in traces
                    if event.get("action") not in {"citation_audit", "answer_figures", "run_outcome"}
                ]
                traces.append({"action": "run_outcome", "answer_status": "TIMED_OUT"})
                error = "Comparison run timed out."
            else:
                logger.exception("Research run %s failed", run_id)
                error = str(exc) or type(exc).__name__
            await asyncio.to_thread(
                db.update_run_result, run_id, status="failed", reasoning_trace=traces, error=error,
            )
        finally:
            ledger.close()
            try:
                if execution:
                    execution.budget.stopped.set()
                if watcher:
                    watcher.cancel()
                    await asyncio.gather(watcher, return_exceptions=True)
            finally:
                self._comparisons.pop(run_id, None)
                self._token_ledgers.pop(run_id, None)
                self.tasks.pop(run_id, None)
                if steps is not None:
                    steps.put_nowait(None)

        record = await asyncio.to_thread(db.get_run, run_id)
        record = record or {"id": run_id, "status": "cancelled"}
        return ResearchResult(record, state if record["status"] == "completed" else {})
