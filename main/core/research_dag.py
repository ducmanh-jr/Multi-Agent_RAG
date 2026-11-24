"""
Dynamic Research DAG Planner & TaskScheduler Executor (PR-07).
Builds validated dependency-aware ResearchPlans (parallel investigation tasks ->
evidence merge -> synthesis -> verification -> report) and supports dynamic
replanning when Fact-Checker reports unresolved EvidenceGaps.
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from core.comparison_plan import run_coro_sync
from core.research_tasks import ResearchPlan, ResearchTask, TaskStatus
from core.schemas import RouterPlanOutput
from core.task_scheduler import TaskOutcome, TaskPolicy, TaskScheduler

TaskKind = Literal["investigate", "gap_investigate", "merge", "synthesize", "verify", "write"]


@dataclass(frozen=True)
class ResearchDAGBudget:
    """Bounded resource limits for one dynamic research DAG execution."""

    max_tasks: int = 12
    max_parallel_subqueries: int = 3
    max_concurrency: int = 2
    task_timeout_seconds: float = 75.0
    run_timeout_seconds: float = 180.0
    max_task_retries: int = 0


@dataclass(frozen=True)
class ResearchDAGNode:
    """Typed specification for one node inside a dynamic ResearchPlan DAG."""

    id: str
    kind: TaskKind
    query: str = ""
    depends_on_success: tuple[str, ...] = ()
    depends_on_terminal: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_task(self) -> ResearchTask:
        return ResearchTask(
            id=self.id,
            depends_on_success=self.depends_on_success,
            depends_on_terminal=self.depends_on_terminal,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "query": self.query,
            "depends_on_success": list(self.depends_on_success),
            "depends_on_terminal": list(self.depends_on_terminal),
            "metadata": dict(self.metadata),
        }


@dataclass
class DynamicResearchDAG:
    """Validated DAG plan combining ResearchPlan topology with typed node specs and budget."""

    nodes: dict[str, ResearchDAGNode]
    plan: ResearchPlan
    budget: ResearchDAGBudget
    round_index: int = 0
    seen_queries: set[str] = field(default_factory=set)

    def to_manifest(self) -> dict[str, Any]:
        tasks_status = self.plan.tasks
        return {
            "round_index": self.round_index,
            "total_tasks": len(self.nodes),
            "budget": {
                "max_tasks": self.budget.max_tasks,
                "max_concurrency": self.budget.max_concurrency,
                "task_timeout_seconds": self.budget.task_timeout_seconds,
                "run_timeout_seconds": self.budget.run_timeout_seconds,
            },
            "nodes": [
                {
                    **node.to_dict(),
                    "status": str(tasks_status[nid].status),
                }
                for nid, node in self.nodes.items()
            ],
        }


def _query_signature(text: str) -> str:
    normalized = " ".join((text or "").lower().split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()[:12]


def build_research_dag_plan(
    router_plan: RouterPlanOutput,
    *,
    evidence_gaps: Sequence[Mapping[str, Any]] | None = None,
    budget: ResearchDAGBudget | None = None,
    round_index: int = 0,
) -> DynamicResearchDAG:
    """
    Constructs a validated ResearchPlan DAG from a Supervisor RouterPlanOutput:
      investigate_1..k (parallel) -> merge_evidence -> synthesize -> verify -> write
    """
    cfg = budget or ResearchDAGBudget()
    raw_queries = [q.strip() for q in (router_plan.sub_queries or []) if isinstance(q, str) and q.strip()]
    if evidence_gaps:
        for gap in evidence_gaps:
            sq = str(gap.get("suggested_query") or gap.get("missing_aspect") or "").strip()
            if sq and sq not in raw_queries:
                raw_queries.append(sq)
    if not raw_queries:
        raw_queries = ["primary_research_query"]

    selected_queries = list(dict.fromkeys(raw_queries))[: max(1, cfg.max_parallel_subqueries)]
    nodes: dict[str, ResearchDAGNode] = {}
    seen_sigs: set[str] = set()
    investigate_ids: list[str] = []

    prefix = f"r{round_index}_" if round_index > 0 else ""
    for idx, sub_q in enumerate(selected_queries, start=1):
        tid = f"{prefix}investigate_{idx}"
        nodes[tid] = ResearchDAGNode(
            id=tid,
            kind="gap_investigate" if round_index > 0 else "investigate",
            query=sub_q,
            metadata={"branch_index": idx, "intent": router_plan.intent, "complexity": router_plan.complexity},
        )
        investigate_ids.append(tid)
        seen_sigs.add(_query_signature(sub_q))

    merge_id = f"{prefix}merge_evidence"
    synth_id = f"{prefix}synthesize"
    verify_id = f"{prefix}verify"
    write_id = f"{prefix}write"

    # Merge waits for all investigation branches to finish (terminal), tolerating partial branch failures.
    nodes[merge_id] = ResearchDAGNode(
        id=merge_id,
        kind="merge",
        depends_on_terminal=tuple(investigate_ids),
        metadata={"branch_count": len(investigate_ids)},
    )
    nodes[synth_id] = ResearchDAGNode(
        id=synth_id,
        kind="synthesize",
        depends_on_success=(merge_id,),
    )
    nodes[verify_id] = ResearchDAGNode(
        id=verify_id,
        kind="verify",
        depends_on_success=(synth_id,),
    )
    nodes[write_id] = ResearchDAGNode(
        id=write_id,
        kind="write",
        depends_on_success=(verify_id,),
    )

    plan = ResearchPlan([node.to_task() for node in nodes.values()])
    return DynamicResearchDAG(
        nodes=nodes,
        plan=plan,
        budget=cfg,
        round_index=round_index,
        seen_queries=seen_sigs,
    )


def replan_research_dag_for_gaps(
    existing_manifest: Mapping[str, Any] | None,
    evidence_gaps: Sequence[Mapping[str, Any]],
    *,
    budget: ResearchDAGBudget | None = None,
) -> dict[str, Any]:
    """
    Appends new gap-driven investigation tasks to an existing DAG manifest while
    enforcing total task budget and deduplicating previously attempted queries.
    """
    cfg = budget or ResearchDAGBudget()
    manifest = dict(existing_manifest or {})
    existing_nodes = list(manifest.get("nodes") or [])
    prior_round = int(manifest.get("round_index", 0))
    next_round = prior_round + 1
    seen_sigs = {_query_signature(str(n.get("query") or "")) for n in existing_nodes if n.get("query")}
    remaining_slots = max(0, cfg.max_tasks - len(existing_nodes))

    new_gap_nodes: list[dict[str, Any]] = []
    if remaining_slots >= 2:
        max_new_investigations = min(cfg.max_parallel_subqueries, remaining_slots - 1)
        prev_verify = next(
            (n["id"] for n in reversed(existing_nodes) if n.get("kind") == "verify"),
            None,
        )
        dep_terminal = (prev_verify,) if prev_verify else ()

        gap_ids: list[str] = []
        for idx, gap in enumerate(evidence_gaps, start=1):
            if len(gap_ids) >= max_new_investigations:
                break
            q = str(gap.get("suggested_query") or gap.get("missing_aspect") or "").strip()
            if not q:
                continue
            sig = _query_signature(q)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            gid = f"r{next_round}_gap_investigate_{idx}"
            gap_ids.append(gid)
            new_gap_nodes.append(
                {
                    "id": gid,
                    "kind": "gap_investigate",
                    "query": q,
                    "depends_on_success": [],
                    "depends_on_terminal": list(dep_terminal),
                    "metadata": {"missing_aspect": str(gap.get("missing_aspect") or "")[:180]},
                    "status": str(TaskStatus.PENDING),
                }
            )

        if gap_ids:
            new_gap_nodes.append(
                {
                    "id": f"r{next_round}_merge_gap_evidence",
                    "kind": "merge",
                    "query": "",
                    "depends_on_success": [],
                    "depends_on_terminal": gap_ids,
                    "metadata": {"branch_count": len(gap_ids)},
                    "status": str(TaskStatus.PENDING),
                }
            )

    all_nodes = [*existing_nodes, *new_gap_nodes]
    return {
        "round_index": next_round if new_gap_nodes else prior_round,
        "total_tasks": len(all_nodes),
        "added_gap_tasks": len(new_gap_nodes),
        "budget_exhausted": remaining_slots < 2,
        "budget": manifest.get("budget")
        or {
            "max_tasks": cfg.max_tasks,
            "max_concurrency": cfg.max_concurrency,
            "task_timeout_seconds": cfg.task_timeout_seconds,
            "run_timeout_seconds": cfg.run_timeout_seconds,
        },
        "nodes": all_nodes,
    }


def _chunk_dedup_key(chunk: Mapping[str, Any]) -> str:
    cid = str(chunk.get("id") or "").strip()
    if cid:
        return f"id:{cid}"
    doc_id = str(chunk.get("doc_id") or chunk.get("document_id") or "").strip().lower()
    page_no = str(chunk.get("page_no") or chunk.get("page") or 1)
    text = " ".join(str(chunk.get("text") or chunk.get("content") or "").split())[:200].lower()
    return f"loc:{doc_id}:{page_no}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def merge_branch_chunks(
    branch_results: Sequence[Sequence[dict[str, Any]]],
    *,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """
    Merges and deduplicates retrieved chunks across parallel DAG investigation branches
    using Reciprocal Rank Fusion (RRF) so evidence corroborated by multiple sub-queries ranks higher.
    """
    k_rrf = 60.0
    fused_scores: dict[str, float] = {}
    best_chunk: dict[str, dict[str, Any]] = {}
    hit_counts: dict[str, int] = {}

    for chunks in branch_results:
        for rank, chunk in enumerate(chunks or [], start=1):
            if not isinstance(chunk, dict):
                continue
            key = _chunk_dedup_key(chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (k_rrf + rank))
            hit_counts[key] = hit_counts.get(key, 0) + 1
            if key not in best_chunk:
                best_chunk[key] = dict(chunk)
            else:
                prev_score = float(best_chunk[key].get("rrf_score") or best_chunk[key].get("score") or 0.0)
                curr_score = float(chunk.get("rrf_score") or chunk.get("score") or 0.0)
                if curr_score > prev_score:
                    best_chunk[key] = dict(chunk)

    ordered_keys = sorted(
        best_chunk.keys(),
        key=lambda k: (fused_scores[k], float(best_chunk[k].get("rrf_score") or best_chunk[k].get("score") or 0.0)),
        reverse=True,
    )
    merged: list[dict[str, Any]] = []
    for key in ordered_keys[: max(1, top_k)]:
        item = dict(best_chunk[key])
        item["dag_rrf_score"] = round(fused_scores[key], 6)
        item["dag_branch_hits"] = hit_counts[key]
        merged.append(item)
    return merged


async def execute_retrieval_dag_async(
    sub_queries: Sequence[str],
    retrieve_branch: Callable[[str], dict[str, Any]],
    *,
    top_k: int = 10,
    budget: ResearchDAGBudget | None = None,
    round_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Executes parallel sub-query retrieval tasks followed by a dependent merge task
    using the validated TaskScheduler engine.
    Returns (merged_retrieval_update, trace_event).
    """
    cfg = budget or ResearchDAGBudget()
    queries = list(dict.fromkeys(q.strip() for q in sub_queries if isinstance(q, str) and q.strip()))
    queries = queries[: max(1, cfg.max_parallel_subqueries)]
    if not queries:
        return {"retrieved_docs": []}, {
            "agent": "task_scheduler",
            "action": "research_dag_executed",
            "status": "skipped",
            "reason": "no_sub_queries",
        }

    prefix = f"r{round_index}_" if round_index > 0 else ""
    branch_tasks: list[ResearchTask] = []
    query_by_task: dict[str, str] = {}
    policies: dict[str, TaskPolicy] = {}

    for idx, sq in enumerate(queries, start=1):
        tid = f"{prefix}investigate_{idx}"
        branch_tasks.append(ResearchTask(id=tid))
        query_by_task[tid] = sq
        policies[tid] = TaskPolicy(
            timeout_seconds=cfg.task_timeout_seconds,
            max_retries=cfg.max_task_retries,
            retry_delay_seconds=0.1,
            retry_on=(TimeoutError,),
        )

    merge_tid = f"{prefix}merge_evidence"
    merge_task = ResearchTask(
        id=merge_tid,
        depends_on_terminal=tuple(query_by_task.keys()),
    )
    all_tasks = [*branch_tasks, merge_task]

    scheduler = TaskScheduler(
        all_tasks,
        max_concurrency=min(cfg.max_concurrency, len(branch_tasks)),
        task_policies=policies,
        run_timeout_seconds=cfg.run_timeout_seconds,
    )

    started = time.perf_counter()

    async def _executor(task: ResearchTask, deps: Mapping[str, TaskOutcome]) -> Any:
        if task.id == merge_tid:
            branch_lists: list[list[dict[str, Any]]] = []
            extra_updates: dict[str, Any] = {}
            for dep_id, dep_out in deps.items():
                if dep_out.status == TaskStatus.COMPLETED and isinstance(dep_out.output, dict):
                    docs = dep_out.output.get("retrieved_docs") or []
                    if isinstance(docs, list):
                        branch_lists.append(docs)
                    for passthrough_key in ("seen_chunks", "alpha", "top_k"):
                        if passthrough_key in dep_out.output and passthrough_key not in extra_updates:
                            extra_updates[passthrough_key] = dep_out.output[passthrough_key]
            if not branch_lists:
                raise RuntimeError("All parallel investigation branches failed to produce evidence.")
            merged_docs = merge_branch_chunks(branch_lists, top_k=top_k)
            return {**extra_updates, "retrieved_docs": merged_docs}

        branch_query = query_by_task[task.id]
        return await asyncio.to_thread(retrieve_branch, branch_query)

    outcomes = await scheduler.run(_executor)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

    task_summaries = []
    for tid, out in outcomes.items():
        hits = 0
        if out.status == TaskStatus.COMPLETED and isinstance(out.output, dict):
            hits = len(out.output.get("retrieved_docs") or [])
        task_summaries.append({
            "id": tid,
            "query": query_by_task.get(tid, ""),
            "status": str(out.status),
            "attempts": out.attempts,
            "hits": hits,
            "error": type(out.error).__name__ if out.error else None,
        })

    merge_outcome = outcomes[merge_tid]
    merged_update = (
        dict(merge_outcome.output)
        if merge_outcome.status == TaskStatus.COMPLETED and isinstance(merge_outcome.output, dict)
        else {"retrieved_docs": []}
    )
    completed_branches = sum(
        1 for tid in query_by_task if outcomes[tid].status == TaskStatus.COMPLETED
    )
    trace_event = {
        "agent": "task_scheduler",
        "action": "research_dag_executed",
        "status": "completed" if merge_outcome.status == TaskStatus.COMPLETED else "degraded",
        "round_index": round_index,
        "branch_count": len(query_by_task),
        "completed_branches": completed_branches,
        "merged_chunks": len(merged_update.get("retrieved_docs") or []),
        "elapsed_ms": elapsed_ms,
        "tasks": task_summaries,
    }
    return merged_update, trace_event


def execute_retrieval_dag(
    sub_queries: Sequence[str],
    retrieve_branch: Callable[[str], dict[str, Any]],
    *,
    top_k: int = 10,
    budget: ResearchDAGBudget | None = None,
    round_index: int = 0,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synchronous wrapper for invoking execute_retrieval_dag_async inside LangGraph nodes."""
    return run_coro_sync(
        execute_retrieval_dag_async(
            sub_queries,
            retrieve_branch,
            top_k=top_k,
            budget=budget,
            round_index=round_index,
        )
    )


def pending_dag_investigation_queries(manifest: Mapping[str, Any] | None) -> list[str]:
    """Returns queries for any PENDING gap_investigate (or investigate) nodes in the DAG manifest."""
    if not isinstance(manifest, Mapping):
        return []
    nodes = manifest.get("nodes") or []
    gap_queries = [
        str(n.get("query") or "").strip()
        for n in nodes
        if isinstance(n, Mapping)
        and n.get("kind") == "gap_investigate"
        and str(n.get("status") or TaskStatus.PENDING) == str(TaskStatus.PENDING)
        and str(n.get("query") or "").strip()
    ]
    if gap_queries:
        return list(dict.fromkeys(gap_queries))
    inv_queries = [
        str(n.get("query") or "").strip()
        for n in nodes
        if isinstance(n, Mapping)
        and n.get("kind") == "investigate"
        and str(n.get("status") or TaskStatus.PENDING) == str(TaskStatus.PENDING)
        and str(n.get("query") or "").strip()
    ]
    return list(dict.fromkeys(inv_queries))


def sync_dag_manifest_from_retrieval(
    manifest: Mapping[str, Any] | None,
    trace_event: Mapping[str, Any] | None = None,
    *,
    retrieved_count: int = 0,
) -> dict[str, Any]:
    """
    Synchronizes PENDING investigation and merge nodes in the DAG manifest with actual
    TaskScheduler execution outcomes (or single-branch retrieval outcomes).
    """
    if not isinstance(manifest, Mapping) or not manifest.get("nodes"):
        return dict(manifest or {})
    updated = dict(manifest)
    nodes = [dict(n) for n in (manifest.get("nodes") or []) if isinstance(n, Mapping)]

    task_by_id: dict[str, Mapping[str, Any]] = {}
    task_by_query: dict[str, Mapping[str, Any]] = {}
    if isinstance(trace_event, Mapping):
        for t in trace_event.get("tasks") or []:
            if isinstance(t, Mapping) and t.get("id"):
                task_by_id[str(t["id"])] = t
                q = str(t.get("query") or "").strip()
                if q:
                    task_by_query[_query_signature(q)] = t

    any_branch_completed = retrieved_count > 0
    for node in nodes:
        kind = node.get("kind")
        curr_status = str(node.get("status") or TaskStatus.PENDING)
        if curr_status != str(TaskStatus.PENDING):
            continue
        if kind in {"investigate", "gap_investigate"}:
            nid = str(node.get("id") or "")
            q_sig = _query_signature(str(node.get("query") or ""))
            matched = task_by_id.get(nid) or task_by_query.get(q_sig)
            meta = dict(node.get("metadata") or {})
            if matched:
                st = str(matched.get("status") or TaskStatus.COMPLETED)
                node["status"] = st
                meta["attempts"] = int(matched.get("attempts") or 1)
                meta["hits"] = int(matched.get("hits") or 0)
                if matched.get("error"):
                    meta["error"] = matched["error"]
                if st == str(TaskStatus.COMPLETED):
                    any_branch_completed = True
            else:
                node["status"] = str(TaskStatus.COMPLETED if retrieved_count > 0 else TaskStatus.FAILED)
                meta["hits"] = retrieved_count
            node["metadata"] = meta
        elif kind == "merge":
            nid = str(node.get("id") or "")
            matched = task_by_id.get(nid)
            meta = dict(node.get("metadata") or {})
            if matched:
                node["status"] = str(matched.get("status") or TaskStatus.COMPLETED)
                meta["merged_chunks"] = int(matched.get("hits") or retrieved_count)
            else:
                node["status"] = str(TaskStatus.COMPLETED if any_branch_completed else TaskStatus.FAILED)
                meta["merged_chunks"] = retrieved_count
            node["metadata"] = meta

    updated["nodes"] = nodes
    updated["completed_tasks"] = sum(
        1 for n in nodes if str(n.get("status")) == str(TaskStatus.COMPLETED)
    )
    return updated


def mark_dag_stage_status(
    manifest: Mapping[str, Any] | None,
    kind: TaskKind,
    status: TaskStatus | str,
    *,
    metadata_update: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Marks the latest PENDING (or latest matching) node of `kind` with `status` and optional metadata."""
    if not isinstance(manifest, Mapping) or not manifest.get("nodes"):
        return dict(manifest or {})
    updated = dict(manifest)
    nodes = [dict(n) for n in (manifest.get("nodes") or []) if isinstance(n, Mapping)]
    target_idx = None
    for idx, node in enumerate(nodes):
        if node.get("kind") == kind and str(node.get("status") or TaskStatus.PENDING) == str(TaskStatus.PENDING):
            target_idx = idx
            break
    if target_idx is None:
        for idx in range(len(nodes) - 1, -1, -1):
            if nodes[idx].get("kind") == kind:
                target_idx = idx
                break
    if target_idx is not None:
        node = dict(nodes[target_idx])
        node["status"] = str(TaskStatus(status))
        if metadata_update:
            node["metadata"] = {**dict(node.get("metadata") or {}), **dict(metadata_update)}
        nodes[target_idx] = node
    updated["nodes"] = nodes
    updated["completed_tasks"] = sum(
        1 for n in nodes if str(n.get("status")) == str(TaskStatus.COMPLETED)
    )
    return updated


async def execute_full_research_dag_async(
    dag: DynamicResearchDAG,
    initial_state: Mapping[str, Any] | None = None,
    *,
    retrieve_branch: Callable[[str], dict[str, Any]] | None = None,
    retrieve_fn: Callable[[str], dict[str, Any]] | None = None,
    synthesize_fn: Callable[[dict[str, Any]], dict[str, Any]],
    verify_fn: Callable[[dict[str, Any]], dict[str, Any]],
    write_fn: Callable[[dict[str, Any]], dict[str, Any]],
    top_k: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Executes the entire end-to-end ResearchPlan DAG (investigate_* -> merge_evidence ->
    synthesize -> verify -> write) through TaskScheduler with strict dependency enforcement.
    Returns (final_state, updated_dag_manifest).
    """
    retriever = retrieve_branch or retrieve_fn
    if retriever is None:
        raise ValueError("Either retrieve_branch or retrieve_fn must be provided.")
    state: dict[str, Any] = dict(initial_state or {})
    top_k_val = int(state.get("top_k") or top_k or 10)
    tasks = [node.to_task() for node in dag.nodes.values()]
    policies = {
        nid: TaskPolicy(
            timeout_seconds=dag.budget.task_timeout_seconds,
            max_retries=dag.budget.max_task_retries if node.kind in {"investigate", "gap_investigate"} else 0,
            retry_delay_seconds=0.1,
            retry_on=(TimeoutError,),
        )
        for nid, node in dag.nodes.items()
    }
    scheduler = TaskScheduler(
        tasks,
        max_concurrency=dag.budget.max_concurrency,
        task_policies=policies,
        run_timeout_seconds=dag.budget.run_timeout_seconds,
    )

    async def _executor(task: ResearchTask, deps: Mapping[str, TaskOutcome]) -> Any:
        node = dag.nodes[task.id]
        if node.kind in {"investigate", "gap_investigate"}:
            return await asyncio.to_thread(retriever, node.query)
        if node.kind == "merge":
            branch_lists: list[list[dict[str, Any]]] = []
            for dep_out in deps.values():
                if dep_out.status == TaskStatus.COMPLETED and isinstance(dep_out.output, dict):
                    docs = dep_out.output.get("retrieved_docs") or []
                    if isinstance(docs, list):
                        branch_lists.append(docs)
            if not branch_lists:
                raise RuntimeError("No investigation branch produced evidence.")
            merged_docs = merge_branch_chunks(branch_lists, top_k=top_k_val)
            state["retrieved_docs"] = merged_docs
            state.setdefault("source_units", merged_docs)
            return {"retrieved_docs": merged_docs}
        if node.kind == "synthesize":
            out = await asyncio.to_thread(synthesize_fn, dict(state))
            if isinstance(out, dict):
                state.update(out)
            return out
        if node.kind == "verify":
            out = await asyncio.to_thread(verify_fn, dict(state))
            if isinstance(out, dict):
                state.update(out)
            return out
        if node.kind == "write":
            out = await asyncio.to_thread(write_fn, dict(state))
            if isinstance(out, dict):
                state.update(out)
            return out
        return {}

    outcomes = await scheduler.run(_executor)
    manifest = dag.to_manifest()
    for node_entry in manifest["nodes"]:
        nid = node_entry["id"]
        if nid in outcomes:
            out = outcomes[nid]
            node_entry["status"] = str(out.status)
            meta = dict(node_entry.get("metadata") or {})
            meta["attempts"] = out.attempts
            if out.error:
                meta["error"] = type(out.error).__name__
            node_entry["metadata"] = meta
    manifest["completed_tasks"] = sum(
        1 for n in manifest["nodes"] if str(n.get("status")) == str(TaskStatus.COMPLETED)
    )
    state.setdefault("document_schema", {})
    state["document_schema"] = {**dict(state["document_schema"]), "research_dag": manifest}
    return state, manifest


def execute_full_research_dag(
    dag: DynamicResearchDAG,
    initial_state: Mapping[str, Any] | None = None,
    *,
    retrieve_branch: Callable[[str], dict[str, Any]] | None = None,
    retrieve_fn: Callable[[str], dict[str, Any]] | None = None,
    synthesize_fn: Callable[[dict[str, Any]], dict[str, Any]],
    verify_fn: Callable[[dict[str, Any]], dict[str, Any]],
    write_fn: Callable[[dict[str, Any]], dict[str, Any]],
    top_k: int = 10,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synchronous wrapper for full end-to-end ResearchPlan DAG execution via TaskScheduler."""
    return run_coro_sync(
        execute_full_research_dag_async(
            dag,
            initial_state,
            retrieve_branch=retrieve_branch,
            retrieve_fn=retrieve_fn,
            synthesize_fn=synthesize_fn,
            verify_fn=verify_fn,
            write_fn=write_fn,
            top_k=top_k,
        )
    )
