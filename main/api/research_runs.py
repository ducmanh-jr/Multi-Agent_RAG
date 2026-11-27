"""Shared execution and persistence for queued, synchronous and streaming research."""

import asyncio
import hashlib
import time
import uuid
from contextlib import aclosing
from copy import deepcopy
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from langgraph.graph import END

from api.citations import file_sha256, pin_citation_provenance
from api.figures import pin_answer_figures
from api.run_manifest import RUN_MANIFEST_ACTION, manifest_event
from core import db
from core.citation_audit import get_citation_audit
from core.comparison_plan import (
    COMPARISON_REVIEW, DEFAULT_COMPARISON_TIMEOUT_SECONDS, ComparisonBudget, comparison_schedule,
    incomplete_comparison, use_comparison_budget,
)
from core.config import RunConfig
from core.graph import build_research_rag_graph, resolve_next_node_after_checkpoint
from core.supervisor import supervisor_planner
from core.task_scheduler import ScheduleTimeoutError
from core.token_usage import RunTokenLedger, capture_token_usage
from core.worker_observations import capture_worker_observations, drain_worker_observations
from utils.language import localized_status_message, resolve_language_bundle
from utils.logger import logger


WAITING_MESSAGE = "Vui lòng chọn một tài liệu trong kho hoặc bật Auto-Scout để bắt đầu nghiên cứu."
WAITING_REPORT = f"# Thông báo Hệ thống\n\n> **Trạng thái:** `WAITING_FOR_DOCUMENT`\n\n{WAITING_MESSAGE}"
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md"}
CHECKPOINT_STATE_KEYS = (
    "query",
    "query_with_context",
    "resolved_query",
    "inherited_entities",
    "chat_history",
    "document_scope",
    "query_language",
    "answer_language",
    "source_language",
    "pdf_path",
    "pdf_paths",
    "intent",
    "complexity",
    "sub_queries",
    "required_entities",
    "is_numeric_query",
    "is_comparative_query",
    "alpha",
    "top_k",
    "retry_count",
    "max_reflection_loops",
    "retrieved_docs",
    "seen_chunks",
    "no_new_evidence",
    "source_units",
    "answer",
    "short_answer",
    "claims",
    "evidence_spans",
    "critic_action",
    "critic_feedback",
    "verified_claims",
    "rejected_claims",
    "evidence_gaps",
    "fact_check_label",
    "answer_status",
    "confidence",
    "final_report",
    "statistical_audit",
    "code_verification",
    "document_schema",
    "reasoning_trace",
)


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
    answer_language: str | None = None
    resume_state: dict[str, Any] | None = None
    resumed_from_run_id: str | None = None
    resumed_from_step: int | None = None
    resumed_from_node: str | None = None
    resume_start_at_node: str | None = None


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

    async def resume(
        self,
        source_run_id: str,
        *,
        config: RunConfig | None = None,
        mode: str | None = None,
    ) -> ResearchRun:
        """Creates a recovered run initialized from the latest persisted SQLite checkpoint."""
        record = await asyncio.to_thread(db.get_run, source_run_id)
        if not record:
            raise KeyError(f"Run '{source_run_id}' not found.")
        checkpoint = await asyncio.to_thread(db.get_latest_run_checkpoint, source_run_id)
        if not checkpoint:
            raise ValueError(f"Run '{source_run_id}' has no persisted checkpoints to resume from.")

        manifest_cfg: dict[str, Any] = {}
        for event in record.get("reasoning_trace") or []:
            if isinstance(event, dict) and event.get("action") == RUN_MANIFEST_ACTION:
                if isinstance(event.get("configuration"), dict):
                    manifest_cfg = dict(event["configuration"])
                break

        saved_agent_modes = (
            dict(manifest_cfg["agent_modes"])
            if isinstance(manifest_cfg.get("agent_modes"), dict)
            else None
        )
        effective_mode = mode if mode is not None else manifest_cfg.get("mode_override")
        run_cfg = config or RunConfig.capture(agent_modes=saved_agent_modes)
        if config is not None and saved_agent_modes and effective_mode is None:
            run_cfg = config.with_overrides(
                AGENT_RETRIEVAL_MODE=saved_agent_modes.get("retrieval", config.AGENT_RETRIEVAL_MODE),
                AGENT_READER_MODE=saved_agent_modes.get("reader", config.AGENT_READER_MODE),
                AGENT_SUMMARIZER_MODE=saved_agent_modes.get("summarizer", config.AGENT_SUMMARIZER_MODE),
                AGENT_FACTCHECKER_MODE=saved_agent_modes.get(
                    "fact_checker",
                    saved_agent_modes.get("factchecker", config.AGENT_FACTCHECKER_MODE),
                ),
                AGENT_WRITER_MODE=saved_agent_modes.get("writer", config.AGENT_WRITER_MODE),
            )

        ckpt_state = dict(checkpoint.get("state") or {})
        doc_ids = tuple(d for d in (record.get("document_ids") or []) if isinstance(d, str) and d)
        ckpt_scope = [
            item for item in (ckpt_state.get("document_scope") or [])
            if isinstance(item, dict)
        ]
        if not doc_ids and ckpt_scope:
            doc_ids = tuple(
                str(item["id"])
                for item in ckpt_scope
                if isinstance(item.get("id"), str) and item.get("id") and item.get("catalogued", True)
            )
        ckpt_paths = list(ckpt_state.get("pdf_paths") or [])
        if not ckpt_paths and ckpt_scope:
            ckpt_paths = [str(item["path"]) for item in ckpt_scope if isinstance(item.get("path"), str) and item.get("path")]
            if ckpt_paths:
                ckpt_state["pdf_paths"] = ckpt_paths
                ckpt_state.setdefault("pdf_path", ckpt_paths[0])
        pdf_path = ckpt_state.get("pdf_path") or (ckpt_paths[0] if ckpt_paths else None)
        saved_chat_history = [
            dict(turn) for turn in (ckpt_state.get("chat_history") or [])
            if isinstance(turn, dict)
        ]
        saved_answer_lang = ckpt_state.get("answer_language")
        comparison_plan = bool(manifest_cfg.get("comparison_plan_requested", False))
        retrieval_cfg = manifest_cfg.get("retrieval") if isinstance(manifest_cfg.get("retrieval"), dict) else {}
        limits_cfg = manifest_cfg.get("limits") if isinstance(manifest_cfg.get("limits"), dict) else {}
        parallel_retrieval = bool(retrieval_cfg.get("parallel_requested", True))
        auto_scout = bool(manifest_cfg.get("auto_scout_requested", False))
        max_reflection_loops = limits_cfg.get("requested_reflection_loops")
        if max_reflection_loops is None and "max_reflection_loops" in ckpt_state:
            max_reflection_loops = ckpt_state.get("max_reflection_loops")
        token_budget = manifest_cfg.get("token_budget")
        comparison_timeout_seconds = float(
            manifest_cfg.get("comparison_timeout_seconds") or DEFAULT_COMPARISON_TIMEOUT_SECONDS
        )
        start_at_node = resolve_next_node_after_checkpoint(
            checkpoint.get("node_name"),
            ckpt_state,
            comparison_plan=comparison_plan,
        )

        request = ResearchRequest(
            query=str(ckpt_state.get("query") or record.get("question") or ""),
            config=run_cfg,
            mode=effective_mode,
            document_ids=doc_ids,
            pdf_path=pdf_path if not doc_ids else None,
            auto_scout=auto_scout,
            max_reflection_loops=max_reflection_loops,
            chat_history=saved_chat_history,
            parallel_retrieval=parallel_retrieval,
            comparison_plan=comparison_plan,
            comparison_timeout_seconds=comparison_timeout_seconds,
            token_budget=token_budget,
            answer_language=saved_answer_lang,
            session_id=record.get("session_id"),
            resume_state=ckpt_state,
            resumed_from_run_id=source_run_id,
            resumed_from_step=checkpoint.get("step_index"),
            resumed_from_node=checkpoint.get("node_name"),
            resume_start_at_node=start_at_node,
        )
        return await self.start(request)

    async def _admit(self, request: ResearchRequest, stream: bool) -> ResearchRun:
        resume_paths = (
            [p for p in (request.resume_state.get("pdf_paths") or []) if isinstance(p, str) and p]
            if isinstance(request.resume_state, dict)
            else []
        )
        if not request.document_ids and len(resume_paths) > 1:
            sources = await asyncio.to_thread(_resolve_paths, resume_paths)
        else:
            sources = await asyncio.to_thread(_resolve_selection, request.document_ids, request.pdf_path)
        has_checkpoint_evidence = bool(
            isinstance(request.resume_state, dict)
            and (request.resume_state.get("source_units") or request.resume_state.get("retrieved_docs"))
        )
        scout_enabled = request.auto_scout and not sources and not has_checkpoint_evidence
        waiting = not sources and not scout_enabled and not has_checkpoint_evidence
        admission = manifest_event(
            request.config, mode=request.mode, auto_scout=request.auto_scout, scout_enabled=scout_enabled,
            reflection_loops=request.max_reflection_loops, parallel_retrieval=request.parallel_retrieval,
            comparison_plan=request.comparison_plan,
            token_budget=request.token_budget,
        )
        traces = [admission]
        if request.resumed_from_run_id:
            traces.append({
                "action": "run_resumed_from_checkpoint",
                "source_run_id": request.resumed_from_run_id,
                "resumed_from_step": request.resumed_from_step,
                "resumed_from_node": request.resumed_from_node,
                "resume_start_at_node": request.resume_start_at_node,
            })
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
            lang_bundle = resolve_language_bundle(
                request.query,
                explicit_answer_language=request.answer_language,
            )
            w_lang = lang_bundle["answer_language"]
            w_msg = localized_status_message("WAITING_FOR_DOCUMENT", language=w_lang, query=request.query)
            if w_lang == "en":
                report = f"# System Notice\n\n> **Status:** `WAITING_FOR_DOCUMENT`\n\n{w_msg}"
            else:
                report = WAITING_REPORT
            traces.append({"action": "waiting_for_document", "message": w_msg})
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
        lang_bundle = resolve_language_bundle(
            request.query,
            explicit_answer_language=request.answer_language,
        )
        state: dict[str, Any] = {
            "query": request.query,
            "query_language": lang_bundle["query_language"],
            "answer_language": lang_bundle["answer_language"],
            "source_language": lang_bundle["source_language"],
            "pdf_path": paths[0] if paths else None,
            "pdf_paths": paths,
            "document_scope": [
                {
                    "id": s["id"],
                    "name": s["name"],
                    "path": s["path"],
                    "sha256": s["sha256"],
                    "catalogued": bool(s.get("catalogued", False)),
                }
                for s in sources
            ],
            "retry_count": 0,
            "chat_history": request.chat_history,
        }
        if isinstance(request.resume_state, dict):
            for key in CHECKPOINT_STATE_KEYS:
                if key in request.resume_state and request.resume_state[key] is not None:
                    state[key] = deepcopy(request.resume_state[key])
            if paths:
                state["pdf_path"] = paths[0]
                state["pdf_paths"] = paths
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
                if (
                    request.resume_start_at_node not in {END, "__end__"}
                    and await asyncio.to_thread(db.run_is_processing, run_id)
                ):
                    graph_kwargs: dict[str, Any] = {
                        "mode": request.mode,
                        "run_config": request.config,
                        "auto_scout": request.auto_scout and not sources,
                        "parallel_retrieval": request.parallel_retrieval,
                        "comparison_plan": request.comparison_plan,
                    }
                    if request.auto_scout and sources:
                        graph_kwargs["allow_scout_handoff"] = True
                    if request.resume_start_at_node:
                        graph_kwargs["start_at_node"] = request.resume_start_at_node
                    graph = await asyncio.to_thread(build_research_rag_graph, **graph_kwargs)
                    start_time = time.monotonic()
                    step_counter = int(request.resumed_from_step or 0)
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
                                if node == "scout_node" and (output.get("pdf_path") or output.get("pdf_paths")):
                                    found_paths = output.get("pdf_paths") or [output["pdf_path"]]
                                    if not sources:
                                        added = await asyncio.to_thread(_resolve_paths, found_paths)
                                        if not added:
                                            raise RuntimeError("Tài liệu Auto-Scout vừa tìm không còn truy cập được.")
                                        sources = added
                                        paths = [source["path"] for source in sources]
                                    else:
                                        existing_path_set = {source["path"] for source in sources}
                                        resolved_found = await asyncio.to_thread(_resolve_paths, found_paths)
                                        newly_added = [s for s in resolved_found if s["path"] not in existing_path_set]
                                        if newly_added:
                                            sources = [*sources, *newly_added]
                                            paths = [source["path"] for source in sources]
                                            added = sources
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
                                step_counter += 1
                                schema_snap = state.get("document_schema") or {}
                                checkpoint_state = {
                                    key: state.get(key)
                                    for key in CHECKPOINT_STATE_KEYS
                                    if key in state
                                }
                                await asyncio.to_thread(
                                    db.save_run_checkpoint,
                                    run_id,
                                    step_counter,
                                    node,
                                    checkpoint_state,
                                    evidence_store=schema_snap.get("evidence_store"),
                                    dag_state=schema_snap.get("research_dag"),
                                )
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
