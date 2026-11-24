"""
M1 Fail-First Regression & Concurrency Test Suite:
  - BUG-01: Multi-branch seen_chunks union, multi-source provenance merge, and no_new_evidence recomputation.
  - BUG-02: EvidenceStore default provenance_verified=False when source_units are absent.
  - BUG-03: Separation of DAG task execution status (COMPLETED) and verification review_outcome
            (PASSED / NEEDS_MORE_EVIDENCE / CONTRADICTED).
  - BUG-04: Consistent task IDs across Planner, Replanner, Executor, and Manifest Sync; full round subgraph
            creation on gap replanning without overwriting prior round history.
  - RISK-05: SQLiteFTS5Searcher :memory: connection RLock, CPUMinRetrievalAgent _doc_index_cache RLock,
             and non-redundant fallback behavior when Retrieval DAG completes with 0 hits.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
import threading
from typing import Any

import pytest

from agents.agent1_retrieval.tools.fts5_search import SQLiteFTS5Searcher
from core.config import settings
from core.evidence_store import EvidenceStore
from core.research_dag import (
    ResearchDAGBudget,
    build_research_dag_plan,
    execute_full_research_dag,
    execute_retrieval_dag,
    mark_dag_stage_status,
    replan_research_dag_for_gaps,
    sync_dag_manifest_from_retrieval,
)
from core.research_tasks import TaskStatus
from core.supervisor import supervisor_planner
from modes.cpu_min.retrieval import CPUMinRetrievalAgent


def test_bug01_parallel_retrieval_merges_seen_chunks_provenance_and_no_new_evidence():
    """
    Branch A finds no new chunks (only old_chunk, no_new_evidence=True).
    Branch B finds chunk_b1 and shared_quote (from doc_01.pdf p.12).
    Branch C finds shared_quote (from doc_02.pdf p.8) and chunk_c1.
    Merged output must:
      1. Retain seen_chunks from ALL branches (old_chunk, chunk_b1, chunk_c1, shared_quote).
      2. Compute no_new_evidence=False because union of branches found new chunks beyond previous_seen.
      3. Merge provenance and discovered_by_branches when identical chunks/IDs are found across branches.
    """
    previous_seen = [
        {"id": "old_chunk", "doc_id": "doc_00.pdf", "page_no": 1, "text": "Previously seen baseline chunk."}
    ]

    def retrieve_branch(query: str) -> dict[str, Any]:
        if query == "branch_a":
            return {
                "retrieved_docs": list(previous_seen),
                "seen_chunks": list(previous_seen),
                "no_new_evidence": True,
            }
        if query == "branch_b":
            b_docs = [
                {
                    "id": "chunk_b1",
                    "doc_id": "doc_01.pdf",
                    "page_no": 4,
                    "text": "Unique experimental result from branch B.",
                    "rrf_score": 0.9,
                },
                {
                    "id": "chunk_shared",
                    "doc_id": "doc_01.pdf",
                    "page_no": 12,
                    "text": "Shared benchmark observation across papers.",
                    "rrf_score": 0.85,
                    "provenance": [{"doc_id": "doc_01.pdf", "page_no": 12, "unit_id": "chunk_shared"}],
                },
            ]
            return {
                "retrieved_docs": b_docs,
                "seen_chunks": [*previous_seen, *b_docs],
                "no_new_evidence": False,
            }
        c_docs = [
            {
                "id": "chunk_shared",
                "doc_id": "doc_02.pdf",
                "page_no": 8,
                "text": "Shared benchmark observation across papers.",
                "rrf_score": 0.88,
                "provenance": [{"doc_id": "doc_02.pdf", "page_no": 8, "unit_id": "chunk_shared"}],
            },
            {
                "id": "chunk_c1",
                "doc_id": "doc_02.pdf",
                "page_no": 9,
                "text": "Ablation table from branch C.",
                "rrf_score": 0.8,
            },
        ]
        return {
            "retrieved_docs": c_docs,
            "seen_chunks": [*previous_seen, *c_docs],
            "no_new_evidence": False,
        }

    merged, trace = execute_retrieval_dag(
        ["branch_a", "branch_b", "branch_c"],
        retrieve_branch,
        top_k=10,
        round_index=1,
        previous_seen_chunks=previous_seen,
    )
    assert trace["status"] == "completed"
    seen_ids = {c["id"] for c in merged.get("seen_chunks") or []}
    assert {"old_chunk", "chunk_b1", "chunk_c1", "chunk_shared"}.issubset(seen_ids)
    assert merged.get("no_new_evidence") is False

    # Check multi-source provenance merge on chunk_shared
    shared_in_docs = next(d for d in merged["retrieved_docs"] if d["id"] == "chunk_shared")
    prov_docs = {p.get("doc_id") for p in shared_in_docs.get("provenance") or []}
    assert {"doc_01.pdf", "doc_02.pdf"}.issubset(prov_docs)
    assert len(shared_in_docs.get("discovered_by_branches") or []) >= 2


def test_bug01_full_research_dag_also_merges_seen_chunks():
    base_plan = supervisor_planner.plan_query("Compare Transformer and RNN on parallelization and memory")
    dag = build_research_dag_plan(base_plan)

    def retrieve_fn(q: str) -> dict[str, Any]:
        cid = f"cid_{q}"
        chunk = {"id": cid, "doc_id": "p.pdf", "page_no": 1, "text": f"Evidence for {q}"}
        return {"retrieved_docs": [chunk], "seen_chunks": [chunk]}

    state_out, _ = execute_full_research_dag(
        dag,
        {"query": "Compare Transformer and RNN on parallelization and memory"},
        retrieve_fn=retrieve_fn,
        synthesize_fn=lambda st: {"answer": "Ans", "claims": []},
        verify_fn=lambda st: {"fact_check_label": "support", "verified_claims": [], "rejected_claims": []},
        write_fn=lambda st: {"final_report": "Report"},
    )
    assert len(state_out.get("seen_chunks") or []) >= 2


def test_bug02_evidence_store_marks_unverified_when_source_units_absent():
    store = EvidenceStore()
    added = store.add_evidence_spans(
        [
            {
                "id": "span_no_units",
                "source_unit_id": "u_missing",
                "doc_id": "paper.pdf",
                "page_no": 2,
                "quote": "Unverified quote without source units.",
            }
        ],
        source_units=[],
    )
    assert added == 1
    span = store.spans["span_no_units"]
    assert span["provenance_verified"] is False
    assert span["provenance_reason"] == "unverified_no_units"

    summary = store.summary()
    assert summary["span_count"] == 1
    assert summary["verified_span_count"] == 0
    assert summary["unverified_span_count"] == 1


def test_bug03_dag_verify_stage_separates_execution_status_and_review_outcome(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    graph_mod = importlib.import_module("core.graph")

    base_plan = supervisor_planner.plan_query("What is the F1 score?")
    dag = build_research_dag_plan(base_plan)
    manifest = dag.to_manifest()

    updated = mark_dag_stage_status(
        manifest,
        "verify",
        TaskStatus.COMPLETED,
        round_index=0,
        review_outcome="NEEDS_MORE_EVIDENCE",
        metadata_update={"fact_check_label": "not_enough_info"},
    )
    verify_node = next(n for n in updated["nodes"] if n["kind"] == "verify")
    assert verify_node["status"] == "COMPLETED"
    assert verify_node["review_outcome"] == "NEEDS_MORE_EVIDENCE"
    assert verify_node["metadata"]["review_outcome"] == "NEEDS_MORE_EVIDENCE"


def test_bug04_replan_creates_full_round_subgraph_and_preserves_round0_history():
    base_plan = supervisor_planner.plan_query("Compare AlphaGo and AlphaZero architectures")
    dag = build_research_dag_plan(base_plan, budget=ResearchDAGBudget(max_tasks=16))
    manifest = dag.to_manifest()

    # Complete round 0 tasks
    manifest = sync_dag_manifest_from_retrieval(manifest, retrieved_count=4, round_index=0)
    manifest = mark_dag_stage_status(
        manifest, "synthesize", TaskStatus.COMPLETED, round_index=0, metadata_update={"round0_marker": True}
    )
    manifest = mark_dag_stage_status(
        manifest,
        "verify",
        TaskStatus.COMPLETED,
        round_index=0,
        review_outcome="NEEDS_MORE_EVIDENCE",
        metadata_update={"fact_check_label": "not_enough_info"},
    )

    gaps = [
        {"missing_aspect": "Training TPU hours", "suggested_query": "AlphaZero TPU training hours"},
        {"missing_aspect": "Self-play games", "suggested_query": "AlphaZero total self-play games"},
    ]
    replanned = replan_research_dag_for_gaps(manifest, gaps, budget=ResearchDAGBudget(max_tasks=16))
    assert replanned["round_index"] == 1

    r1_nodes = [n for n in replanned["nodes"] if n.get("round_index") == 1]
    r1_kinds = [n["kind"] for n in r1_nodes]
    assert "gap_investigate" in r1_kinds
    assert "merge" in r1_kinds
    assert "synthesize" in r1_kinds
    assert "verify" in r1_kinds

    # Execute round 1 retrieval DAG using the planned task IDs and sync manifest
    _, trace_r1 = execute_retrieval_dag(
        ["AlphaZero TPU training hours", "AlphaZero total self-play games"],
        lambda q: {"retrieved_docs": [{"id": f"id_{q[:6]}", "doc_id": "az.pdf", "page_no": 1, "text": q}]},
        round_index=1,
        task_kind="gap_investigate",
    )
    synced_r1 = sync_dag_manifest_from_retrieval(replanned, trace_r1, retrieved_count=2, round_index=1)
    r1_merge = next(n for n in synced_r1["nodes"] if n.get("round_index") == 1 and n["kind"] == "merge")
    assert r1_merge["status"] == "COMPLETED"

    # Update round 1 synthesize & verify and confirm round 0 nodes were NOT overwritten
    synced_r1 = mark_dag_stage_status(
        synced_r1, "synthesize", TaskStatus.COMPLETED, round_index=1, metadata_update={"round1_marker": True}
    )
    synced_r1 = mark_dag_stage_status(
        synced_r1,
        "verify",
        TaskStatus.COMPLETED,
        round_index=1,
        review_outcome="PASSED",
        metadata_update={"fact_check_label": "support"},
    )

    r0_synth = next(n for n in synced_r1["nodes"] if n.get("round_index", 0) == 0 and n["kind"] == "synthesize")
    r1_synth = next(n for n in synced_r1["nodes"] if n.get("round_index") == 1 and n["kind"] == "synthesize")
    assert r0_synth["metadata"].get("round0_marker") is True
    assert "round1_marker" not in r0_synth["metadata"]
    assert r1_synth["metadata"].get("round1_marker") is True

    r0_verify = next(n for n in synced_r1["nodes"] if n.get("round_index", 0) == 0 and n["kind"] == "verify")
    r1_verify = next(n for n in synced_r1["nodes"] if n.get("round_index") == 1 and n["kind"] == "verify")
    assert r0_verify["review_outcome"] == "NEEDS_MORE_EVIDENCE"
    assert r1_verify["review_outcome"] == "PASSED"


def test_risk05_sqlite_fts5_and_cpu_min_cache_concurrency():
    searcher = SQLiteFTS5Searcher()
    assert hasattr(searcher, "_connection_lock")
    assert isinstance(searcher._connection_lock, type(threading.RLock()))

    docs = [
        {"id": f"d_{i}", "doc_id": "corpus.pdf", "title": f"Title {i}", "text": f"Transformer attention layer {i} benchmark", "page_no": i}
        for i in range(1, 40)
    ]
    searcher.fit(docs)

    errors: list[Exception] = []

    def worker(idx: int):
        try:
            for _ in range(15):
                res = searcher.search(f"Transformer attention {idx % 10}", top_k=5)
                assert isinstance(res, list)
        except Exception as exc:
            errors.append(exc)

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(worker, i) for i in range(16)]
        for f in futures:
            f.result()

    assert not errors
    searcher.close()

    agent = CPUMinRetrievalAgent()
    assert hasattr(agent, "_cache_lock")
    assert isinstance(agent._cache_lock, type(threading.RLock()))


def test_m1_branch_timeout_preserves_surviving_branches_and_no_redundant_single_pass_rerun(monkeypatch, tmp_path):
    """
    1. When one parallel branch times out, surviving branches still merge their seen_chunks & provenance.
    2. In graph.py, when Retrieval DAG completes normally with 0 hits, single-pass retrieval is NOT re-run.
    """
    import time

    def retrieve_with_timeout(q: str) -> dict[str, Any]:
        if q == "slow_branch":
            time.sleep(0.35)
        chunk = {"id": f"c_{q}", "doc_id": "p.pdf", "page_no": 1, "text": f"Evidence from {q}"}
        return {"retrieved_docs": [chunk], "seen_chunks": [chunk], "no_new_evidence": False}

    merged, trace = execute_retrieval_dag(
        ["fast_branch_1", "slow_branch", "fast_branch_2"],
        retrieve_with_timeout,
        budget=ResearchDAGBudget(max_concurrency=3, task_timeout_seconds=0.1, max_task_retries=0),
        round_index=1,
        previous_seen_chunks=[],
    )
    assert trace["status"] == "completed"
    assert trace["completed_branches"] == 2
    seen_ids = {c["id"] for c in merged["seen_chunks"]}
    assert seen_ids == {"c_fast_branch_1", "c_fast_branch_2"}
    assert merged["no_new_evidence"] is False

    # Verify graph.py does not re-run single-pass retrieval when DAG finishes cleanly with 0 hits
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    graph_mod = importlib.import_module("core.graph")

    call_log: list[list[str]] = []

    def fake_zero_hit_retrieval(state):
        call_log.append(list(state.get("sub_queries") or []))
        return {"retrieved_docs": [], "seen_chunks": [], "no_new_evidence": True}

    runners = {
        "retrieval": fake_zero_hit_retrieval,
        "reader": lambda s: {"source_units": []},
        "summarizer": lambda s: {"answer": "No info", "answer_status": "NOT_FOUND_IN_DOCUMENT", "claims": [], "evidence_spans": []},
        "fact_checker": lambda s: {"fact_check_label": "support", "verified_claims": [], "rejected_claims": [], "evidence_gaps": []},
        "writer": lambda s: {"final_report": "No info"},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, state: {})

    app = graph_mod.build_research_rag_graph(
        "hybrid",
        parallel_retrieval=True,
        auto_scout=False,
        planner_llm_fn=lambda _p: {
            "intent": "compare",
            "complexity": "HIGH",
            "sub_queries": ["q_branch_1", "q_branch_2"],
            "required_entities": [],
            "is_comparative_query": True,
        },
    )
    out = app.invoke({"query": "Compare q_branch_1 and q_branch_2", "pdf_paths": ["p.pdf"], "max_reflection_loops": 0})
    expected_branches = len(out.get("sub_queries") or [])
    # Each branch ran once with [branch_query]; single-pass fallback with all sub_queries must NOT run
    assert all(len(sq) == 1 for sq in call_log)
    assert len(call_log) == expected_branches


def test_m1_end_to_end_multi_round_langgraph_replan_preserves_round_history_and_seen_chunks(monkeypatch, tmp_path):
    """
    Full 2-round LangGraph run:
      - Round 0: 2 parallel retrieval branches -> reader -> summarizer -> fact_checker (not_enough_info + gap)
      - Replan creates Round 1 subgraph (r1_gap_investigate_1..2 -> r1_merge_evidence -> r1_synthesize -> r1_verify)
      - Round 1: parallel retrieval branches -> reader -> summarizer -> fact_checker (support) -> writer
    Verifies:
      - Cumulative seen_chunks retains chunks from all branches across both Round 0 and Round 1.
      - Round 0 verify node retains review_outcome == 'NEEDS_MORE_EVIDENCE'.
      - Round 1 verify node has review_outcome == 'PASSED'.
      - All nodes in both rounds reach COMPLETED status.
    """
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    graph_mod = importlib.import_module("core.graph")

    def fake_retrieval(state):
        q = state["query"]
        prev = list(state.get("seen_chunks") or [])
        chunk = {
            "id": f"chunk_{q.replace(' ', '_')}",
            "doc_id": "study.pdf",
            "page_no": 1 if "r0" in q else 2,
            "text": f"Verified finding for {q}.",
            "rrf_score": 0.9,
        }
        prev_ids = {c.get("id") for c in prev if isinstance(c, dict)}
        all_seen = [*prev, *( [chunk] if chunk["id"] not in prev_ids else [] )]
        return {
            "retrieved_docs": [chunk],
            "seen_chunks": all_seen,
            "no_new_evidence": False,
        }

    def fake_reader(state):
        return {"source_units": list(state.get("retrieved_docs") or [])}

    def fake_summarizer(state):
        units = state.get("source_units") or []
        u0 = units[0]
        return {
            "answer": u0["text"],
            "claims": [{"id": "c1", "text": u0["text"], "evidence_ids": ["sp1"]}],
            "evidence_spans": [
                {"id": "sp1", "source_unit_id": u0["id"], "doc_id": u0["doc_id"], "page_no": u0["page_no"], "quote": u0["text"]}
            ],
        }

    checker_calls = {"count": 0}

    def fake_checker(state):
        checker_calls["count"] += 1
        if checker_calls["count"] == 1:
            return {
                "fact_check_label": "not_enough_info",
                "verified_claims": [],
                "rejected_claims": [{"id": "c1", "text": state.get("answer", ""), "status": "NOT_ENOUGH_INFO", "reason": "Need latency"}],
                "evidence_gaps": [
                    {"missing_aspect": "Latency p95", "suggested_query": "r1_gap_latency_p95"},
                    {"missing_aspect": "Memory footprint", "suggested_query": "r1_gap_memory_footprint"},
                ],
            }
        return {
            "fact_check_label": "support",
            "verified_claims": [{"id": "c1", "text": state.get("answer", ""), "status": "SUPPORT", "evidence_ids": ["sp1"]}],
            "rejected_claims": [],
            "evidence_gaps": [],
        }

    runners = {
        "retrieval": fake_retrieval,
        "reader": fake_reader,
        "summarizer": fake_summarizer,
        "fact_checker": fake_checker,
        "writer": lambda s: {"final_report": s.get("answer", "")},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, state: {})

    app = graph_mod.build_research_rag_graph(
        "hybrid",
        parallel_retrieval=True,
        auto_scout=False,
        planner_llm_fn=lambda _p: {
            "intent": "compare",
            "complexity": "HIGH",
            "sub_queries": ["r0_sub_alpha", "r0_sub_beta"],
            "required_entities": [],
            "is_comparative_query": True,
        },
    )
    final_state = app.invoke({
        "query": "r0_primary_query",
        "pdf_paths": ["study.pdf"],
        "max_reflection_loops": 1,
    })

    # 1. Both rounds' chunks must be accumulated in seen_chunks
    seen_ids = {c["id"] for c in final_state.get("seen_chunks") or []}
    assert "chunk_r0_sub_alpha" in seen_ids
    assert "chunk_r0_sub_beta" in seen_ids
    assert "chunk_r1_gap_latency_p95" in seen_ids
    assert "chunk_r1_gap_memory_footprint" in seen_ids

    # 2. Check DAG manifest round 0 and round 1 nodes
    dag_manifest = final_state["document_schema"]["research_dag"]
    assert dag_manifest["round_index"] == 1
    assert dag_manifest["last_verify_outcome"] == "PASSED"

    r0_verify = next(n for n in dag_manifest["nodes"] if n.get("round_index", 0) == 0 and n["kind"] == "verify")
    r1_verify = next(n for n in dag_manifest["nodes"] if n.get("round_index") == 1 and n["kind"] == "verify")
    assert r0_verify["status"] == "COMPLETED"
    assert r0_verify["review_outcome"] == "NEEDS_MORE_EVIDENCE"
    assert r1_verify["status"] == "COMPLETED"
    assert r1_verify["review_outcome"] == "PASSED"
    assert all(n["status"] == "COMPLETED" for n in dag_manifest["nodes"])


