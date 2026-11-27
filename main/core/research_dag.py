"""
Dynamic Research DAG Planner & TaskScheduler Executor (PR-07 / M1 Hardened).
Builds validated dependency-aware ResearchPlans (parallel investigation tasks ->
evidence merge -> synthesis -> verification -> report) and supports dynamic
round-based replanning when Fact-Checker reports unresolved EvidenceGaps.
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
ReviewOutcome = Literal["PASSED", "NEEDS_MORE_EVIDENCE", "CONTRADICTED", "UNVERIFIED"]


def make_dag_node_id(
    round_index: int,
    kind: TaskKind,
    branch_index: int | None = None,
) -> str:
    """
    Canonical deterministic node/task ID builder shared across Planner, Replanner,
    Executor, and Manifest Synchronizer.
    """
    r = max(0, int(round_index))
    prefix = f"r{r}_" if r > 0 else ""
    if kind in {"investigate", "gap_investigate"}:
        b_idx = 1 if branch_index is None else int(branch_index)
        return f"{prefix}{kind}_{b_idx}"
    if kind == "merge":
        return f"{prefix}merge_evidence"
    return f"{prefix}{kind}"


@dataclass(frozen=True)
class ResearchDAGBudget:
    """Bounded resource limits for one dynamic research DAG execution."""

    max_tasks: int = 16
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
    round_index: int = 0
    branch_index: int | None = None
    query: str = ""
    depends_on_success: tuple[str, ...] = ()
    depends_on_terminal: tuple[str, ...] = ()
    review_outcome: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_task(self) -> ResearchTask:
        return ResearchTask(
            id=self.id,
            depends_on_success=self.depends_on_success,
            depends_on_terminal=self.depends_on_terminal,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "round_index": self.round_index,
            "branch_index": self.branch_index,
            "query": self.query,
            "depends_on_success": list(self.depends_on_success),
            "depends_on_terminal": list(self.depends_on_terminal),
            "metadata": dict(self.metadata),
        }
        if self.review_outcome is not None:
            payload["review_outcome"] = self.review_outcome
        return payload


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
            "control_plane": "langgraph_supervisor",
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

    inv_kind: TaskKind = "gap_investigate" if round_index > 0 else "investigate"
    for idx, sub_q in enumerate(selected_queries, start=1):
        tid = make_dag_node_id(round_index, inv_kind, idx)
        nodes[tid] = ResearchDAGNode(
            id=tid,
            kind=inv_kind,
            round_index=round_index,
            branch_index=idx,
            query=sub_q,
            metadata={
                "round_index": round_index,
                "branch_index": idx,
                "intent": router_plan.intent,
                "complexity": router_plan.complexity,
            },
        )
        investigate_ids.append(tid)
        seen_sigs.add(_query_signature(sub_q))

    merge_id = make_dag_node_id(round_index, "merge")
    synth_id = make_dag_node_id(round_index, "synthesize")
    verify_id = make_dag_node_id(round_index, "verify")
    write_id = make_dag_node_id(round_index, "write")

    # Merge waits for all investigation branches to finish (terminal), tolerating partial branch failures.
    nodes[merge_id] = ResearchDAGNode(
        id=merge_id,
        kind="merge",
        round_index=round_index,
        depends_on_terminal=tuple(investigate_ids),
        metadata={"round_index": round_index, "branch_count": len(investigate_ids)},
    )
    nodes[synth_id] = ResearchDAGNode(
        id=synth_id,
        kind="synthesize",
        round_index=round_index,
        depends_on_success=(merge_id,),
        metadata={"round_index": round_index},
    )
    nodes[verify_id] = ResearchDAGNode(
        id=verify_id,
        kind="verify",
        round_index=round_index,
        depends_on_success=(synth_id,),
        metadata={"round_index": round_index},
    )
    nodes[write_id] = ResearchDAGNode(
        id=write_id,
        kind="write",
        round_index=round_index,
        depends_on_success=(verify_id,),
        metadata={"round_index": round_index},
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
    Appends a new gap-driven research round subgraph to an existing DAG manifest while
    enforcing total task budget, deduplicating previously attempted queries, and preserving
    prior round nodes intact (creating round-scoped merge -> synthesize -> verify nodes
    and re-pointing the pending write node to the new round's verify node).
    """
    cfg = budget or ResearchDAGBudget()
    manifest = dict(existing_manifest or {})
    existing_nodes = [dict(n) for n in (manifest.get("nodes") or []) if isinstance(n, Mapping)]
    for n in existing_nodes:
        n.setdefault("round_index", int(n.get("metadata", {}).get("round_index", 0)))

    prior_round = int(manifest.get("round_index", 0))
    next_round = prior_round + 1
    seen_sigs = {_query_signature(str(n.get("query") or "")) for n in existing_nodes if n.get("query")}
    remaining_slots = max(0, cfg.max_tasks - len(existing_nodes))

    new_round_nodes: list[dict[str, Any]] = []
    added_gap_tasks = 0
    if remaining_slots >= 2:
        # Reserve 2 downstream slots for r{next_round}_synthesize and r{next_round}_verify when budget allows,
        # while still honoring tight 2-3 slot budgets in explicit budget boundary tests.
        pipeline_reserve = 2 if remaining_slots >= 4 else 0
        max_new_investigations = min(
            cfg.max_parallel_subqueries,
            max(1, remaining_slots - 1 - pipeline_reserve),
        )
        prev_verify = next(
            (n["id"] for n in reversed(existing_nodes) if n.get("kind") == "verify"),
            None,
        )
        dep_terminal = (prev_verify,) if prev_verify else ()

        gap_ids: list[str] = []
        for gap in evidence_gaps:
            if len(gap_ids) >= max_new_investigations:
                break
            q = str(gap.get("suggested_query") or gap.get("missing_aspect") or "").strip()
            if not q:
                continue
            sig = _query_signature(q)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            branch_idx = len(gap_ids) + 1
            gid = make_dag_node_id(next_round, "gap_investigate", branch_idx)
            gap_ids.append(gid)
            new_round_nodes.append(
                {
                    "id": gid,
                    "kind": "gap_investigate",
                    "round_index": next_round,
                    "branch_index": branch_idx,
                    "query": q,
                    "depends_on_success": [],
                    "depends_on_terminal": list(dep_terminal),
                    "metadata": {
                        "round_index": next_round,
                        "branch_index": branch_idx,
                        "missing_aspect": str(gap.get("missing_aspect") or "")[:180],
                    },
                    "status": str(TaskStatus.PENDING),
                }
            )

        if gap_ids:
            merge_id = make_dag_node_id(next_round, "merge")
            new_round_nodes.append(
                {
                    "id": merge_id,
                    "kind": "merge",
                    "round_index": next_round,
                    "branch_index": None,
                    "query": "",
                    "depends_on_success": [],
                    "depends_on_terminal": list(gap_ids),
                    "metadata": {"round_index": next_round, "branch_count": len(gap_ids)},
                    "status": str(TaskStatus.PENDING),
                }
            )
            added_gap_tasks = len(gap_ids) + 1

            if pipeline_reserve >= 2:
                synth_id = make_dag_node_id(next_round, "synthesize")
                verify_id = make_dag_node_id(next_round, "verify")
                new_round_nodes.append(
                    {
                        "id": synth_id,
                        "kind": "synthesize",
                        "round_index": next_round,
                        "branch_index": None,
                        "query": "",
                        "depends_on_success": [merge_id],
                        "depends_on_terminal": [],
                        "metadata": {"round_index": next_round},
                        "status": str(TaskStatus.PENDING),
                    }
                )
                new_round_nodes.append(
                    {
                        "id": verify_id,
                        "kind": "verify",
                        "round_index": next_round,
                        "branch_index": None,
                        "query": "",
                        "depends_on_success": [synth_id],
                        "depends_on_terminal": [],
                        "metadata": {"round_index": next_round},
                        "status": str(TaskStatus.PENDING),
                    }
                )
                # Re-point any pending write node to wait on this new round's verify stage
                for existing in reversed(existing_nodes):
                    if existing.get("kind") == "write" and str(existing.get("status")) == str(TaskStatus.PENDING):
                        existing["depends_on_success"] = [verify_id]
                        existing["round_index"] = next_round
                        meta = dict(existing.get("metadata") or {})
                        meta["round_index"] = next_round
                        existing["metadata"] = meta
                        break

    all_nodes = [*existing_nodes, *new_round_nodes]
    return {
        "round_index": next_round if new_round_nodes else prior_round,
        "total_tasks": len(all_nodes),
        "added_gap_tasks": added_gap_tasks,
        "added_round_tasks": len(new_round_nodes),
        "budget_exhausted": remaining_slots < 2 or (bool(evidence_gaps) and added_gap_tasks == 0 and remaining_slots < 4),
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
    cid = str(chunk.get("id") or chunk.get("chunk_id") or "").strip()
    if cid:
        return f"id:{cid}"
    ctx_hash = str(chunk.get("context_hash") or "").strip()
    if ctx_hash:
        return f"ctx:{ctx_hash}"
    doc_id = str(chunk.get("doc_id") or chunk.get("document_id") or "").strip().lower()
    page_no = str(chunk.get("page_no") or chunk.get("page") or 1)
    text = " ".join(str(chunk.get("text") or chunk.get("content") or "").split())[:200].lower()
    return f"loc:{doc_id}:{page_no}:{hashlib.sha1(text.encode('utf-8')).hexdigest()[:10]}"


def _extract_chunk_provenance(chunk: Mapping[str, Any], *, branch_id: str | None = None) -> list[dict[str, Any]]:
    """Extracts normalized provenance records from a chunk, including its primary doc/page/unit identity."""
    entries: list[dict[str, Any]] = []
    seen_sigs: set[tuple[str, int, str]] = set()

    raw_list = list(chunk.get("provenance") or [])
    doc_id = str(chunk.get("doc_id") or chunk.get("document_id") or "").strip()
    try:
        page_no = int(chunk.get("page_no") or chunk.get("page") or 1)
    except Exception:
        page_no = 1
    unit_id = str(chunk.get("id") or chunk.get("chunk_id") or chunk.get("source_unit_id") or "").strip()

    if doc_id or unit_id:
        primary: dict[str, Any] = {
            "doc_id": doc_id or "unknown",
            "page_no": page_no,
            "unit_id": unit_id or "chunk",
        }
        if chunk.get("context_hash"):
            primary["context_hash"] = str(chunk["context_hash"])
        if chunk.get("source_path"):
            primary["source_path"] = str(chunk["source_path"])
        if chunk.get("document_sha256"):
            primary["document_sha256"] = str(chunk["document_sha256"])
        if chunk.get("origin_task_id"):
            primary["task_id"] = str(chunk["origin_task_id"])
        if branch_id:
            primary["branch_id"] = branch_id
            primary.setdefault("task_id", branch_id)
        raw_list.insert(0, primary)

    for item in raw_list:
        if not isinstance(item, Mapping):
            continue
        p_doc = str(item.get("doc_id") or item.get("document_id") or doc_id or "unknown").strip()
        try:
            p_page = int(item.get("page_no") or item.get("page") or page_no or 1)
        except Exception:
            p_page = 1
        p_uid = str(item.get("unit_id") or item.get("source_unit_id") or unit_id or "chunk").strip()
        sig = (p_doc, p_page, p_uid)
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            entry = dict(item)
            entry["doc_id"] = p_doc
            entry["page_no"] = p_page
            entry["unit_id"] = p_uid
            entries.append(entry)
    return entries


def _merge_chunk_records(
    existing: Mapping[str, Any],
    incoming: Mapping[str, Any],
    *,
    branch_id: str | None = None,
) -> dict[str, Any]:
    """
    Merges two chunk dicts representing the same deduplication key while preserving
    multi-source provenance and all discovering branches.
    """
    prev_score = float(existing.get("rrf_score") or existing.get("score") or 0.0)
    curr_score = float(incoming.get("rrf_score") or incoming.get("score") or 0.0)
    base = dict(incoming) if curr_score > prev_score else dict(existing)

    # Merge provenance entries across both records
    merged_prov: list[dict[str, Any]] = []
    seen_sigs: set[tuple[str, int, str]] = set()
    for p in [
        *_extract_chunk_provenance(existing),
        *_extract_chunk_provenance(incoming, branch_id=branch_id),
    ]:
        sig = (str(p.get("doc_id") or ""), int(p.get("page_no") or 1), str(p.get("unit_id") or ""))
        if sig not in seen_sigs:
            seen_sigs.add(sig)
            merged_prov.append(p)
    if merged_prov:
        base["provenance"] = merged_prov

    # Merge discovering branches
    branches: list[str] = []
    for b in [
        *(existing.get("discovered_by_branches") or []),
        *(incoming.get("discovered_by_branches") or []),
        *([branch_id] if branch_id else []),
    ]:
        if isinstance(b, str) and b and b not in branches:
            branches.append(b)
    if branches:
        base["discovered_by_branches"] = branches

    return base


def merge_branch_chunks(
    branch_results: Sequence[Sequence[dict[str, Any]]],
    *,
    top_k: int = 10,
    branch_ids: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Merges and deduplicates retrieved chunks across parallel DAG investigation branches
    using Reciprocal Rank Fusion (RRF) so evidence corroborated by multiple sub-queries ranks higher,
    while preserving multi-source provenance and discovering branch lineage.
    """
    k_rrf = 60.0
    fused_scores: dict[str, float] = {}
    best_chunk: dict[str, dict[str, Any]] = {}
    hit_counts: dict[str, int] = {}

    for b_idx, chunks in enumerate(branch_results):
        b_id = branch_ids[b_idx] if branch_ids and b_idx < len(branch_ids) else f"branch_{b_idx + 1}"
        for rank, chunk in enumerate(chunks or [], start=1):
            if not isinstance(chunk, dict):
                continue
            key = _chunk_dedup_key(chunk)
            fused_scores[key] = fused_scores.get(key, 0.0) + (1.0 / (k_rrf + rank))
            hit_counts[key] = hit_counts.get(key, 0) + 1
            if key not in best_chunk:
                initial = dict(chunk)
                initial["provenance"] = _extract_chunk_provenance(chunk, branch_id=b_id)
                branches = list(chunk.get("discovered_by_branches") or [])
                if b_id not in branches:
                    branches.append(b_id)
                initial["discovered_by_branches"] = branches
                best_chunk[key] = initial
            else:
                best_chunk[key] = _merge_chunk_records(best_chunk[key], chunk, branch_id=b_id)

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


def merge_seen_chunks_across_branches(
    branch_outputs: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    previous_seen_chunks: Sequence[Mapping[str, Any]] | None = None,
    round_index: int = 0,
) -> tuple[list[dict[str, Any]], bool]:
    """
    Computes the cumulative union of seen_chunks across all parallel branches and prior rounds:
      NewEvidence = Union(AllBranchChunks) - PreviousSeenChunks
      no_new_evidence = (|NewEvidence| == 0) when round_index > 0
    Preserves multi-source provenance across all merged chunk records.
    """
    seen_by_key: dict[str, dict[str, Any]] = {}
    previous_keys: set[str] = set()

    if previous_seen_chunks is not None:
        for raw in previous_seen_chunks:
            if not isinstance(raw, Mapping):
                continue
            key = _chunk_dedup_key(raw)
            previous_keys.add(key)
            if key not in seen_by_key:
                seen_by_key[key] = dict(raw)
            else:
                seen_by_key[key] = _merge_chunk_records(seen_by_key[key], raw)

    branch_doc_keys: set[str] = set()
    any_branch_reported_new = False
    branch_seen_key_sets: list[set[str]] = []

    for branch_id, output in branch_outputs:
        if not isinstance(output, Mapping):
            continue
        if output.get("no_new_evidence") is False:
            any_branch_reported_new = True
        b_seen = output.get("seen_chunks") or []
        b_docs = output.get("retrieved_docs") or []

        curr_branch_keys: set[str] = set()
        for item in [*b_seen, *b_docs]:
            if not isinstance(item, Mapping):
                continue
            key = _chunk_dedup_key(item)
            curr_branch_keys.add(key)
            if key not in seen_by_key:
                rec = dict(item)
                rec["provenance"] = _extract_chunk_provenance(item, branch_id=branch_id)
                seen_by_key[key] = rec
            else:
                seen_by_key[key] = _merge_chunk_records(seen_by_key[key], item, branch_id=branch_id)
        if curr_branch_keys:
            branch_seen_key_sets.append(curr_branch_keys)

        for doc in b_docs:
            if isinstance(doc, Mapping):
                branch_doc_keys.add(_chunk_dedup_key(doc))

    if previous_seen_chunks is not None:
        new_evidence_keys = (branch_doc_keys | set(seen_by_key.keys())) - previous_keys
        no_new_evidence = bool(round_index > 0 and len(new_evidence_keys) == 0)
    else:
        # Infer whether any branch discovered chunks not present in the intersection of all branches' seen sets
        common_baseline = set.intersection(*branch_seen_key_sets) if len(branch_seen_key_sets) >= 2 else set()
        union_keys = set(seen_by_key.keys())
        has_divergent_new = len(union_keys - common_baseline) > 0 if len(branch_seen_key_sets) >= 2 else False
        no_new_evidence = bool(
            round_index > 0
            and not any_branch_reported_new
            and not has_divergent_new
        )

    return list(seen_by_key.values()), no_new_evidence


def pending_dag_investigation_nodes(
    manifest: Mapping[str, Any] | None,
    *,
    round_index: int | None = None,
) -> list[dict[str, Any]]:
    """Returns pending gap_investigate or investigate node specs from the DAG manifest."""
    if not isinstance(manifest, Mapping):
        return []
    nodes = [dict(n) for n in (manifest.get("nodes") or []) if isinstance(n, Mapping)]
    for target_kind in ("gap_investigate", "investigate"):
        matched = [
            n
            for n in nodes
            if n.get("kind") == target_kind
            and str(n.get("status") or TaskStatus.PENDING) == str(TaskStatus.PENDING)
            and (round_index is None or int(n.get("round_index", 0)) == int(round_index))
            and str(n.get("query") or "").strip()
        ]
        if matched:
            return matched
    return []


def pending_dag_investigation_queries(manifest: Mapping[str, Any] | None) -> list[str]:
    """Returns queries for any PENDING gap_investigate (or investigate) nodes in the DAG manifest."""
    nodes = pending_dag_investigation_nodes(manifest)
    return list(dict.fromkeys(str(n.get("query") or "").strip() for n in nodes if str(n.get("query") or "").strip()))


async def execute_retrieval_dag_async(
    sub_queries: Sequence[str],
    retrieve_branch: Callable[[str], dict[str, Any]],
    *,
    top_k: int = 10,
    budget: ResearchDAGBudget | None = None,
    round_index: int = 0,
    task_kind: TaskKind | None = None,
    dag_manifest: Mapping[str, Any] | None = None,
    previous_seen_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Executes parallel sub-query retrieval tasks followed by a dependent merge task
    using the validated TaskScheduler engine.
    Reuses planned task IDs directly from `dag_manifest` when provided, or uses the
    canonical `make_dag_node_id` builder so IDs never drift between Replan and Execution.
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

    effective_kind: TaskKind = task_kind or ("gap_investigate" if round_index > 0 else "investigate")
    planned_nodes = pending_dag_investigation_nodes(dag_manifest, round_index=round_index)
    planned_by_sig = {_query_signature(str(n.get("query") or "")): str(n["id"]) for n in planned_nodes if n.get("id")}

    branch_tasks: list[ResearchTask] = []
    query_by_task: dict[str, str] = {}
    policies: dict[str, TaskPolicy] = {}
    used_ids: set[str] = set()

    for idx, sq in enumerate(queries, start=1):
        sig = _query_signature(sq)
        tid = planned_by_sig.get(sig)
        if not tid or tid in used_ids:
            tid = make_dag_node_id(round_index, effective_kind, idx)
        used_ids.add(tid)
        branch_tasks.append(ResearchTask(id=tid))
        query_by_task[tid] = sq
        policies[tid] = TaskPolicy(
            timeout_seconds=cfg.task_timeout_seconds,
            max_retries=cfg.max_task_retries,
            retry_delay_seconds=0.1,
            retry_on=(TimeoutError,),
        )

    merge_tid = make_dag_node_id(round_index, "merge")
    if isinstance(dag_manifest, Mapping):
        for n in dag_manifest.get("nodes") or []:
            if (
                isinstance(n, Mapping)
                and n.get("kind") == "merge"
                and int(n.get("round_index", 0)) == int(round_index)
                and str(n.get("status") or TaskStatus.PENDING) == str(TaskStatus.PENDING)
                and n.get("id")
            ):
                merge_tid = str(n["id"])
                break

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
            branch_ids: list[str] = []
            branch_outputs: list[tuple[str, Mapping[str, Any]]] = []
            extra_updates: dict[str, Any] = {}
            for dep_id, dep_out in deps.items():
                if dep_out.status == TaskStatus.COMPLETED and isinstance(dep_out.output, dict):
                    branch_outputs.append((dep_id, dep_out.output))
                    docs = dep_out.output.get("retrieved_docs") or []
                    if isinstance(docs, list):
                        branch_lists.append(docs)
                        branch_ids.append(dep_id)
                    for passthrough_key in ("alpha", "top_k"):
                        if passthrough_key in dep_out.output and passthrough_key not in extra_updates:
                            extra_updates[passthrough_key] = dep_out.output[passthrough_key]
            if not branch_lists:
                raise RuntimeError("All parallel investigation branches failed to produce evidence.")
            merged_docs = merge_branch_chunks(branch_lists, top_k=top_k, branch_ids=branch_ids)
            merged_seen, no_new_ev = merge_seen_chunks_across_branches(
                branch_outputs,
                previous_seen_chunks=previous_seen_chunks,
                round_index=round_index,
            )
            result_payload: dict[str, Any] = {
                **extra_updates,
                "retrieved_docs": merged_docs,
                "seen_chunks": merged_seen,
                "no_new_evidence": no_new_ev,
            }
            return result_payload

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
    task_kind: TaskKind | None = None,
    dag_manifest: Mapping[str, Any] | None = None,
    previous_seen_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synchronous wrapper for invoking execute_retrieval_dag_async inside LangGraph nodes."""
    return run_coro_sync(
        execute_retrieval_dag_async(
            sub_queries,
            retrieve_branch,
            top_k=top_k,
            budget=budget,
            round_index=round_index,
            task_kind=task_kind,
            dag_manifest=dag_manifest,
            previous_seen_chunks=previous_seen_chunks,
        )
    )


def _enrich_chunk_lineage(
    chunk: Mapping[str, Any],
    *,
    doc_path: str | None = None,
    task_id: str | None = None,
) -> dict[str, Any]:
    """Attach explicit per-document and per-task lineage to a retrieved chunk without dropping existing fields."""
    item = dict(chunk)
    if doc_path and not item.get("source_path"):
        item["source_path"] = doc_path
    if task_id:
        item["origin_task_id"] = task_id
        branches = list(item.get("discovered_by_branches") or [])
        if task_id not in branches:
            branches.append(task_id)
        item["discovered_by_branches"] = branches
    prov = _extract_chunk_provenance(item, branch_id=task_id)
    for entry in prov:
        if doc_path and not entry.get("source_path"):
            entry["source_path"] = doc_path
        if task_id and not entry.get("task_id"):
            entry["task_id"] = task_id
        if item.get("document_sha256") and not entry.get("document_sha256"):
            entry["document_sha256"] = str(item["document_sha256"])
    if prov:
        item["provenance"] = prov
    return item


async def execute_document_fanout_dag_async(
    pdf_paths: Sequence[str],
    retrieve_document: Callable[[str], dict[str, Any]],
    *,
    top_k: int = 10,
    budget: ResearchDAGBudget | None = None,
    round_index: int = 0,
    previous_seen_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Fans out retrieval per document via TaskScheduler (one task per PDF path, bounded concurrency),
    then merges chunks while guaranteeing balanced multi-document representation and preserving
    full lineage (`source_path`, `doc_id`, `document_sha256`, `origin_task_id`, `provenance`).
    """
    cfg = budget or ResearchDAGBudget()
    unique_paths = list(dict.fromkeys(str(p).strip() for p in pdf_paths if isinstance(p, str) and str(p).strip()))
    if not unique_paths:
        return {"retrieved_docs": []}, {
            "agent": "task_scheduler",
            "action": "document_fanout_dag_executed",
            "status": "skipped",
            "reason": "no_documents",
            "round_index": round_index,
            "tasks": [],
        }

    prefix = f"r{round_index}_" if round_index > 0 else ""
    doc_tasks: list[ResearchTask] = []
    path_by_task: dict[str, str] = {}
    policies: dict[str, TaskPolicy] = {}
    for idx, path in enumerate(unique_paths, start=1):
        tid = f"{prefix}doc_investigate_{idx}"
        doc_tasks.append(ResearchTask(id=tid))
        path_by_task[tid] = path
        policies[tid] = TaskPolicy(
            timeout_seconds=cfg.task_timeout_seconds,
            max_retries=cfg.max_task_retries,
            retry_delay_seconds=0.1,
            retry_on=(TimeoutError,),
        )

    merge_tid = f"{prefix}merge_documents"
    merge_task = ResearchTask(id=merge_tid, depends_on_terminal=tuple(path_by_task.keys()))
    scheduler = TaskScheduler(
        [*doc_tasks, merge_task],
        max_concurrency=min(max(cfg.max_concurrency, 2), len(doc_tasks), 4),
        task_policies=policies,
        run_timeout_seconds=cfg.run_timeout_seconds,
    )
    started = time.perf_counter()

    async def _executor(task: ResearchTask, deps: Mapping[str, TaskOutcome]) -> Any:
        if task.id == merge_tid:
            branch_lists: list[list[dict[str, Any]]] = []
            branch_ids: list[str] = []
            branch_outputs: list[tuple[str, Mapping[str, Any]]] = []
            top_per_doc: list[dict[str, Any]] = []
            for dep_id, dep_out in deps.items():
                if dep_out.status != TaskStatus.COMPLETED or not isinstance(dep_out.output, dict):
                    continue
                d_path = path_by_task.get(dep_id)
                enriched_docs = [
                    _enrich_chunk_lineage(c, doc_path=d_path, task_id=dep_id)
                    for c in (dep_out.output.get("retrieved_docs") or [])
                    if isinstance(c, Mapping)
                ]
                enriched_seen = [
                    _enrich_chunk_lineage(c, doc_path=d_path, task_id=dep_id)
                    for c in (dep_out.output.get("seen_chunks") or [])
                    if isinstance(c, Mapping)
                ]
                branch_outputs.append((dep_id, {**dep_out.output, "retrieved_docs": enriched_docs, "seen_chunks": enriched_seen}))
                if enriched_docs:
                    branch_lists.append(enriched_docs)
                    branch_ids.append(dep_id)
                    top_per_doc.append(enriched_docs[0])
            if not branch_lists:
                raise RuntimeError("All per-document retrieval fan-out branches failed to produce evidence.")
            effective_top_k = max(int(top_k or 8), len(branch_lists))
            fused = merge_branch_chunks(branch_lists, top_k=effective_top_k + len(branch_lists), branch_ids=branch_ids)
            # Guarantee every responsive document retains at least its #1 chunk before filling by RRF score
            balanced: list[dict[str, Any]] = []
            seen_keys: set[str] = set()
            fused_by_key = {_chunk_dedup_key(c): c for c in fused}
            for lead in top_per_doc:
                k = _chunk_dedup_key(lead)
                if k not in seen_keys:
                    seen_keys.add(k)
                    balanced.append(fused_by_key.get(k, lead))
            for cand in fused:
                k = _chunk_dedup_key(cand)
                if k not in seen_keys:
                    seen_keys.add(k)
                    balanced.append(cand)
                if len(balanced) >= effective_top_k:
                    break
            merged_seen, no_new_ev = merge_seen_chunks_across_branches(
                branch_outputs,
                previous_seen_chunks=previous_seen_chunks,
                round_index=round_index,
            )
            return {
                "retrieved_docs": balanced[:effective_top_k],
                "seen_chunks": merged_seen,
                "no_new_evidence": no_new_ev,
            }

        d_path = path_by_task[task.id]
        return await asyncio.to_thread(retrieve_document, d_path)

    outcomes = await scheduler.run(_executor)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    task_summaries = []
    for tid, out in outcomes.items():
        hits = len(out.output.get("retrieved_docs") or []) if out.status == TaskStatus.COMPLETED and isinstance(out.output, dict) else 0
        task_summaries.append({
            "id": tid,
            "doc_path": path_by_task.get(tid, ""),
            "status": str(out.status),
            "attempts": out.attempts,
            "hits": hits,
            "error": type(out.error).__name__ if out.error else None,
        })
    merge_out = outcomes[merge_tid]
    merged_update = (
        dict(merge_out.output)
        if merge_out.status == TaskStatus.COMPLETED and isinstance(merge_out.output, dict)
        else {"retrieved_docs": []}
    )
    completed_docs = sum(1 for tid in path_by_task if outcomes[tid].status == TaskStatus.COMPLETED)
    trace_event = {
        "agent": "task_scheduler",
        "action": "document_fanout_dag_executed",
        "status": "completed" if merge_out.status == TaskStatus.COMPLETED else "degraded",
        "round_index": round_index,
        "document_count": len(unique_paths),
        "completed_documents": completed_docs,
        "merged_chunks": len(merged_update.get("retrieved_docs") or []),
        "elapsed_ms": elapsed_ms,
        "tasks": task_summaries,
    }
    return merged_update, trace_event


def execute_document_fanout_dag(
    pdf_paths: Sequence[str],
    retrieve_document: Callable[[str], dict[str, Any]],
    *,
    top_k: int = 10,
    budget: ResearchDAGBudget | None = None,
    round_index: int = 0,
    previous_seen_chunks: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synchronous wrapper for per-document parallel fan-out retrieval with lineage preservation."""
    return run_coro_sync(
        execute_document_fanout_dag_async(
            pdf_paths,
            retrieve_document,
            top_k=top_k,
            budget=budget,
            round_index=round_index,
            previous_seen_chunks=previous_seen_chunks,
        )
    )


def sync_dag_manifest_from_retrieval(
    manifest: Mapping[str, Any] | None,
    trace_event: Mapping[str, Any] | None = None,
    *,
    retrieved_count: int = 0,
    round_index: int | None = None,
) -> dict[str, Any]:
    """
    Synchronizes PENDING investigation and merge nodes in the DAG manifest with actual
    TaskScheduler execution outcomes (or single-branch retrieval outcomes) for the active round.
    CRITICAL (M4): When `trace_event` carries task-level outcomes (`tasks`), ONLY nodes whose
    exact `task_id` appears in `trace_event["tasks"]` are transitioned; other pending branches
    remain `PENDING`, and `merge` stays `PENDING` until all its dependencies are terminal.
    """
    if not isinstance(manifest, Mapping) or not manifest.get("nodes"):
        return dict(manifest or {})
    updated = dict(manifest)
    nodes = [dict(n) for n in (manifest.get("nodes") or []) if isinstance(n, Mapping)]

    target_round = (
        int(round_index)
        if round_index is not None
        else int(
            (trace_event.get("round_index") if isinstance(trace_event, Mapping) else None)
            or manifest.get("round_index")
            or 0
        )
    )

    has_task_event_payload = isinstance(trace_event, Mapping) and ("tasks" in trace_event)
    task_by_id: dict[str, Mapping[str, Any]] = {}
    if isinstance(trace_event, Mapping):
        for t in trace_event.get("tasks") or []:
            if isinstance(t, Mapping) and t.get("id"):
                task_by_id[str(t["id"])] = t

    terminal_statuses = {
        str(TaskStatus.COMPLETED),
        str(TaskStatus.FAILED),
        str(TaskStatus.SKIPPED),
        str(TaskStatus.CANCELLED),
    }

    # Pass 1: Update investigation nodes strictly by exact task_id when a task event payload is present.
    for node in nodes:
        node_round = int(node.get("round_index", node.get("metadata", {}).get("round_index", 0)))
        node["round_index"] = node_round
        if node_round != target_round:
            continue
        kind = node.get("kind")
        curr_status = str(node.get("status") or TaskStatus.PENDING)
        if curr_status != str(TaskStatus.PENDING):
            continue
        if kind in {"investigate", "gap_investigate"}:
            nid = str(node.get("id") or "")
            matched = task_by_id.get(nid)
            meta = dict(node.get("metadata") or {})
            if matched is not None:
                st = str(matched.get("status") or TaskStatus.COMPLETED)
                node["status"] = st
                meta["attempts"] = int(matched.get("attempts") or 1)
                meta["hits"] = int(matched.get("hits") or 0)
                if matched.get("error"):
                    meta["error"] = matched["error"]
                node["metadata"] = meta
            elif not has_task_event_payload:
                # Single-pass non-scheduler retrieval completed all planned round queries in one call.
                node["status"] = str(TaskStatus.COMPLETED if retrieved_count > 0 else TaskStatus.FAILED)
                meta["hits"] = retrieved_count
                node["metadata"] = meta

    node_status_by_id = {str(n.get("id") or ""): str(n.get("status") or TaskStatus.PENDING) for n in nodes}

    # Pass 2: Update merge nodes strictly by exact task_id, or only when ALL dependencies are terminal.
    for node in nodes:
        node_round = int(node.get("round_index", node.get("metadata", {}).get("round_index", 0)))
        if node_round != target_round or node.get("kind") != "merge":
            continue
        if str(node.get("status") or TaskStatus.PENDING) != str(TaskStatus.PENDING):
            continue
        nid = str(node.get("id") or "")
        matched = task_by_id.get(nid)
        meta = dict(node.get("metadata") or {})
        if matched is not None:
            node["status"] = str(matched.get("status") or TaskStatus.COMPLETED)
            meta["merged_chunks"] = int(matched.get("hits") or retrieved_count)
            if matched.get("error"):
                meta["error"] = matched["error"]
            node["metadata"] = meta
        else:
            deps = [
                str(d) for d in (
                    list(node.get("depends_on_terminal") or [])
                    + list(node.get("depends_on_success") or [])
                )
                if d
            ]
            if deps:
                dep_statuses = [node_status_by_id.get(d, str(TaskStatus.PENDING)) for d in deps]
                if not all(st in terminal_statuses for st in dep_statuses):
                    # At least one upstream branch is still PENDING/RUNNING -> merge MUST stay PENDING!
                    continue
                any_dep_completed = any(st == str(TaskStatus.COMPLETED) for st in dep_statuses)
                node["status"] = str(TaskStatus.COMPLETED if any_dep_completed else TaskStatus.FAILED)
                meta["merged_chunks"] = retrieved_count
                node["metadata"] = meta
            elif not has_task_event_payload:
                node["status"] = str(TaskStatus.COMPLETED if retrieved_count > 0 else TaskStatus.FAILED)
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
    round_index: int | None = None,
    node_id: str | None = None,
    review_outcome: ReviewOutcome | str | None = None,
    metadata_update: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Marks the target node for `(round_index, kind)` (or explicit `node_id`) with execution `status`,
    optional `review_outcome` (separating task completion from verification verdict), and metadata.
    Never overwrites a completed node from a different research round.
    """
    if not isinstance(manifest, Mapping) or not manifest.get("nodes"):
        return dict(manifest or {})
    updated = dict(manifest)
    nodes = [dict(n) for n in (manifest.get("nodes") or []) if isinstance(n, Mapping)]
    for n in nodes:
        n.setdefault("round_index", int(n.get("metadata", {}).get("round_index", 0)))

    target_idx: int | None = None
    if node_id:
        for idx, node in enumerate(nodes):
            if str(node.get("id")) == str(node_id):
                target_idx = idx
                break
    elif round_index is not None:
        r_target = int(round_index)
        for idx, node in enumerate(nodes):
            if (
                node.get("kind") == kind
                and int(node.get("round_index", 0)) == r_target
                and str(node.get("status") or TaskStatus.PENDING) == str(TaskStatus.PENDING)
            ):
                target_idx = idx
                break
        if target_idx is None:
            for idx in range(len(nodes) - 1, -1, -1):
                if nodes[idx].get("kind") == kind and int(nodes[idx].get("round_index", 0)) == r_target:
                    target_idx = idx
                    break
    else:
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
        meta = dict(node.get("metadata") or {})
        if metadata_update:
            meta.update(dict(metadata_update))
        if review_outcome is not None:
            outcome_str = str(review_outcome)
            node["review_outcome"] = outcome_str
            meta["review_outcome"] = outcome_str
            updated["last_verify_outcome"] = outcome_str
        node["metadata"] = meta
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
    Compatibility/standalone DAG executor for running a single-round ResearchPlan DAG
    (investigate_* -> merge_evidence -> synthesize -> verify -> write) through TaskScheduler.
    Note: In the application runtime, LangGraph + Research Supervisor serves as the primary
    multi-round Control Plane while TaskScheduler executes parallel intra-round tasks.
    Returns (final_state, updated_dag_manifest).
    """
    retriever = retrieve_branch or retrieve_fn
    if retriever is None:
        raise ValueError("Either retrieve_branch or retrieve_fn must be provided.")
    state: dict[str, Any] = dict(initial_state or {})
    top_k_val = int(state.get("top_k") or top_k or 10)
    prev_seen = list(state.get("seen_chunks") or [])
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
            branch_ids: list[str] = []
            branch_outputs: list[tuple[str, Mapping[str, Any]]] = []
            for dep_id, dep_out in deps.items():
                if dep_out.status == TaskStatus.COMPLETED and isinstance(dep_out.output, dict):
                    branch_outputs.append((dep_id, dep_out.output))
                    docs = dep_out.output.get("retrieved_docs") or []
                    if isinstance(docs, list):
                        branch_lists.append(docs)
                        branch_ids.append(dep_id)
            if not branch_lists:
                raise RuntimeError("No investigation branch produced evidence.")
            merged_docs = merge_branch_chunks(branch_lists, top_k=top_k_val, branch_ids=branch_ids)
            merged_seen, no_new_ev = merge_seen_chunks_across_branches(
                branch_outputs,
                previous_seen_chunks=prev_seen,
                round_index=dag.round_index,
            )
            state["retrieved_docs"] = merged_docs
            state["seen_chunks"] = merged_seen
            state["no_new_evidence"] = no_new_ev
            state.setdefault("source_units", merged_docs)
            return {
                "retrieved_docs": merged_docs,
                "seen_chunks": merged_seen,
                "no_new_evidence": no_new_ev,
            }
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
            if node_entry.get("kind") == "verify" and out.status == TaskStatus.COMPLETED:
                label = str(state.get("fact_check_label") or "").lower()
                rejected = state.get("rejected_claims") or []
                if label in {"support", "supported"} and not rejected:
                    v_outcome = "PASSED"
                elif label in {"contradict", "contradicted", "refuted"}:
                    v_outcome = "CONTRADICTED"
                else:
                    v_outcome = "NEEDS_MORE_EVIDENCE"
                node_entry["review_outcome"] = v_outcome
                meta["review_outcome"] = v_outcome
                manifest["last_verify_outcome"] = v_outcome
            node_entry["metadata"] = meta
    manifest["completed_tasks"] = sum(
        1 for n in manifest["nodes"] if str(n.get("status")) == str(TaskStatus.COMPLETED)
    )
    from core.evidence_store import EvidenceStore

    state.setdefault("document_schema", {})
    state["document_schema"] = {
        **dict(state["document_schema"]),
        "research_dag": manifest,
        "evidence_store": EvidenceStore.from_state(state).to_dict(),
    }
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
