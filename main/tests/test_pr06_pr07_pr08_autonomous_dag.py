import asyncio
import importlib
from threading import Barrier

import fitz
import pytest

from core import db
from core.config import RunConfig, settings
from core.evidence_store import EvidenceStore
from core.graph import build_research_rag_graph, orchestrator_router_node
from core.research_dag import (
    ResearchDAGBudget,
    build_research_dag_plan,
    execute_retrieval_dag,
    replan_research_dag_for_gaps,
)
from core.supervisor import supervisor_planner


def test_pr06_orchestrator_uses_llm_planner_and_fails_open_to_deterministic():
    def fake_llm(_prompt: str):
        return {
            "intent": "compare",
            "complexity": "HIGH",
            "sub_queries": ["RAG architecture comparison", "GraphRAG indexing cost", "Agentic RAG latency"],
            "required_entities": ["RAG", "GraphRAG", "Agentic RAG"],
            "is_comparative_query": True,
            "is_numeric_query": True,
            "alpha": 0.65,
            "top_k": 8,
        }

    out = orchestrator_router_node({
        "query": "Compare traditional RAG, GraphRAG and Agentic RAG on architecture and cost",
        "retry_count": 0,
        "planner_llm_fn": fake_llm,
    })
    assert out["intent"] == "compare"
    assert out["complexity"] == "HIGH"
    assert len(out["sub_queries"]) >= 3
    assert out["reasoning_trace"][0]["planning_source"] == "llm_refined"
    dag = out["document_schema"]["research_dag"]
    assert dag["total_tasks"] >= 6
    assert any(n["kind"] == "merge" for n in dag["nodes"])

    # Broken LLM fails open to deterministic plan
    fallback = orchestrator_router_node({
        "query": "What is Self-Attention?",
        "retry_count": 0,
        "planner_llm_fn": lambda _p: "not valid json {{{",
    })
    assert fallback["reasoning_trace"][0]["planning_source"] == "deterministic"
    assert fallback["document_schema"]["research_dag"]["total_tasks"] >= 5


def test_pr07_retrieval_dag_runs_branches_in_parallel_and_tolerates_partial_failure():
    barrier = Barrier(2)

    def retrieve_branch(q: str):
        if q in {"branch_a", "branch_b"}:
            barrier.wait(timeout=5)
        if q == "branch_fail":
            raise RuntimeError("Simulated branch failure")
        return {
            "retrieved_docs": [
                {"id": "shared_chunk", "doc_id": "paper.pdf", "page_no": 2, "text": "Shared core fact.", "rrf_score": 0.8},
                {"id": f"only_{q}", "doc_id": "paper.pdf", "page_no": 3, "text": f"Unique to {q}.", "rrf_score": 0.5},
            ]
        }

    merged, trace = execute_retrieval_dag(
        ["branch_a", "branch_b", "branch_fail"],
        retrieve_branch,
        top_k=5,
        budget=ResearchDAGBudget(max_concurrency=3, task_timeout_seconds=10.0, max_task_retries=0),
    )
    assert trace["action"] == "research_dag_executed"
    assert trace["status"] == "completed"
    assert trace["completed_branches"] == 2
    docs = merged["retrieved_docs"]
    assert docs[0]["id"] == "shared_chunk"
    assert docs[0]["dag_branch_hits"] == 2


def test_pr07_replan_for_gaps_deduplicates_and_enforces_task_budget():
    base_plan = supervisor_planner.plan_query("What is AlphaGo Zero?")
    dag = build_research_dag_plan(base_plan, budget=ResearchDAGBudget(max_tasks=8))
    initial_manifest = dag.to_manifest()
    initial_count = initial_manifest["total_tasks"]
    max_tasks = initial_count + 3

    gaps = [
        {"missing_aspect": "MCTS rollouts", "suggested_query": "AlphaGo Zero MCTS simulation count"},
        {"missing_aspect": "MCTS duplicate", "suggested_query": "AlphaGo Zero MCTS simulation count"},
        {"missing_aspect": "Network weights", "suggested_query": "AlphaGo Zero shared neural network"},
    ]
    replanned = replan_research_dag_for_gaps(initial_manifest, gaps, budget=ResearchDAGBudget(max_tasks=max_tasks))
    assert replanned["added_gap_tasks"] == 3  # 2 unique gap tasks + 1 merge task
    assert replanned["total_tasks"] == max_tasks

    # Budget is now full; further replanning adds 0 tasks and flags budget_exhausted
    exhausted = replan_research_dag_for_gaps(
        replanned,
        [{"missing_aspect": "Extra", "suggested_query": "brand new aspect"}],
        budget=ResearchDAGBudget(max_tasks=max_tasks),
    )
    assert exhausted["added_gap_tasks"] == 0
    assert exhausted["budget_exhausted"] is True


def test_pr07_and_pr08_graph_wires_parallel_dag_and_evidence_store(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    graph_mod = importlib.import_module("core.graph")

    seen_queries = []

    def fake_retrieval(state):
        seen_queries.append(state["query"])
        return {
            "retrieved_docs": [
                {"id": f"doc_{len(seen_queries)}", "doc_id": "p.pdf", "page_no": 1, "text": f"Fact for {state['query']}.", "rrf_score": 0.9}
            ]
        }

    def fake_reader(state):
        return {"source_units": state.get("retrieved_docs") or []}

    def fake_summarizer(state):
        units = state.get("source_units") or []
        return {
            "answer": "Fact for sub_1.",
            "claims": [{"id": "c1", "text": "Fact for sub_1.", "evidence_ids": ["sp1"]}],
            "evidence_spans": [{"id": "sp1", "source_unit_id": units[0]["id"], "doc_id": "p.pdf", "page_no": 1, "quote": units[0]["text"]}],
        }

    def fake_checker(state):
        return {
            "fact_check_label": "support",
            "verified_claims": [{"id": "c1", "text": "Fact for sub_1.", "status": "SUPPORT", "evidence_ids": ["sp1"]}],
            "rejected_claims": [],
            "evidence_gaps": [],
        }

    runners = {
        "retrieval": fake_retrieval,
        "reader": fake_reader,
        "summarizer": fake_summarizer,
        "fact_checker": fake_checker,
        "writer": lambda s: {"final_report": s["answer"]},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, state: {})

    # Use mode="hybrid" so pilot_enabled (cpu_min-only) is False and our general Research DAG runs
    app = graph_mod.build_research_rag_graph(
        "hybrid",
        parallel_retrieval=True,
        auto_scout=False,
        planner_llm_fn=lambda _p: {
            "intent": "compare",
            "complexity": "HIGH",
            "sub_queries": ["sub_1", "sub_2"],
            "required_entities": ["sub_1", "sub_2"],
            "is_comparative_query": True,
        },
    )
    result = app.invoke({
        "query": "Compare sub_1 and sub_2",
        "pdf_paths": ["p.pdf"],
    })
    actions = [e.get("action") for e in result["reasoning_trace"]]
    assert "research_dag_executed" in actions
    assert "sub_1" in seen_queries and "sub_2" in seen_queries
    ev_store = result["document_schema"]["evidence_store"]
    assert ev_store["summary"]["verified_claim_count"] == 1
    assert ev_store["summary"]["verified_span_count"] == 1
    dag_out = result["document_schema"]["research_dag"]
    assert all(n["status"] == "COMPLETED" for n in dag_out["nodes"])
    assert dag_out["completed_tasks"] == dag_out["total_tasks"]


def test_pr08_evidence_store_and_sqlite_checkpoint_resume(monkeypatch, tmp_path):
    # Isolate workspace DB in tmp_path
    db_file = tmp_path / "data" / "workspace.db"
    monkeypatch.setattr(db, "DB_PATH", str(db_file))
    monkeypatch.setattr(db, "_db_initialized", False)
    db.init_db()

    # 1. Verify EvidenceStore contradiction & provenance tracking
    state = {
        "source_units": [
            {"id": "u1", "doc_id": "paper.pdf", "page_no": 4, "text": "The model achieves 91.2% accuracy."}
        ],
        "evidence_spans": [
            {"id": "s_ok", "source_unit_id": "u1", "doc_id": "paper.pdf", "page_no": 4, "quote": "91.2% accuracy"},
            {"id": "s_bad", "source_unit_id": "u1", "doc_id": "paper.pdf", "page_no": 4, "quote": "Fabricated quote"},
        ],
        "verified_claims": [{"id": "c1", "text": "Achieves 91.2% accuracy", "evidence_ids": ["s_ok"]}],
        "rejected_claims": [{"id": "c2", "text": "Achieves 50% accuracy", "status": "CONTRADICT", "reason": "Wrong value"}],
        "evidence_gaps": [{"missing_aspect": "Latency", "suggested_query": "model latency ms"}],
    }
    store = EvidenceStore.from_state(state)
    summary = store.summary()
    assert summary["verified_span_count"] == 1
    assert summary["span_count"] == 2
    assert summary["contradiction_count"] == 1
    assert summary["gap_count"] == 1

    # 2. Verify SQLite checkpoint persistence & recovery discovery
    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as doc:
        doc.new_page().insert_text((72, 72), "The model achieves 91.2% accuracy.")
        doc.save(pdf_path)

    import hashlib
    doc_sha = hashlib.sha256(pdf_path.read_bytes()).hexdigest()
    db.save_document(
        doc_id="doc_1",
        name="sample.pdf",
        path=str(pdf_path),
        size=pdf_path.stat().st_size,
        doc_type="pdf",
    )
    saved = db.save_run(
        run_id="run_interrupted_1",
        question="What accuracy does the model achieve?",
        status="processing",
        reasoning_trace=[],
        documents=[{"id": "doc_1", "name": "sample.pdf", "path": str(pdf_path), "sha256": doc_sha}],
    )
    assert db.save_run_checkpoint(
        saved["id"],
        1,
        "reader_node",
        {
            "query": saved["question"],
            "pdf_path": str(pdf_path),
            "pdf_paths": [str(pdf_path)],
            "source_units": state["source_units"],
            "max_reflection_loops": 0,
        },
        evidence_store=store.to_dict(),
        dag_state={"total_tasks": 5},
    )
    latest = db.get_latest_run_checkpoint(saved["id"])
    assert latest is not None
    assert latest["step_index"] == 1
    assert latest["node_name"] == "reader_node"
    assert latest["evidence_store"]["summary"]["contradiction_count"] == 1

    recoverable = db.list_recoverable_runs()
    assert any(r["run_id"] == "run_interrupted_1" for r in recoverable)

    # 3. Verify ResearchRunService.resume() creates a recovered run seeded from checkpoint
    from api.research_runs import ResearchRunService

    service = ResearchRunService()

    async def _run_resume():
        resumed = await service.resume("run_interrupted_1", config=RunConfig.capture(), mode="cpu_min")
        assert resumed.task is not None
        final = await asyncio.wait_for(resumed.result(), timeout=15)
        return resumed.record, final

    rec, final = asyncio.run(_run_resume())
    resumed_events = [e for e in rec["reasoning_trace"] if e.get("action") == "run_resumed_from_checkpoint"]
    assert len(resumed_events) == 1
    assert resumed_events[0]["source_run_id"] == "run_interrupted_1"
    assert resumed_events[0]["resumed_from_node"] == "reader_node"
    assert resumed_events[0]["resume_start_at_node"] == "summarizer_node"
    resumed_ckpts = db.list_run_checkpoints(rec["id"])
    assert resumed_ckpts
    # Because we resumed after reader_node, the first executed step in the resumed run is summarizer_node (step 2)
    assert resumed_ckpts[0]["node_name"] == "summarizer_node"
    assert resumed_ckpts[0]["step_index"] == 2
    executed_nodes = [c["node_name"] for c in resumed_ckpts]
    assert "orchestrator_router" not in executed_nodes
    assert "retrieval_node" not in executed_nodes
    assert "reader_node" not in executed_nodes


def test_p1b_execute_full_research_dag_end_to_end():
    from core.research_dag import execute_full_research_dag

    base_plan = supervisor_planner.plan_query("Compare Transformer and RNN on parallelization and memory")
    dag = build_research_dag_plan(base_plan)

    def retrieve_fn(q: str):
        return {
            "retrieved_docs": [
                {"id": f"c_{q[:8]}", "doc_id": "arch.pdf", "page_no": 1, "text": f"Evidence for {q}.", "rrf_score": 0.85}
            ]
        }

    def synthesize_fn(st):
        docs = st.get("retrieved_docs") or []
        return {
            "answer": f"Synthesized from {len(docs)} chunks.",
            "claims": [{"id": "c1", "text": "Synthesized claim."}],
        }

    def verify_fn(st):
        return {
            "fact_check_label": "support",
            "verified_claims": st.get("claims") or [],
            "rejected_claims": [],
        }

    def write_fn(st):
        return {"final_report": f"# Report\n\n{st.get('answer', '')}"}

    state_out, manifest_out = execute_full_research_dag(
        dag,
        {"query": "Compare Transformer and RNN on parallelization and memory", "top_k": 5},
        retrieve_fn=retrieve_fn,
        synthesize_fn=synthesize_fn,
        verify_fn=verify_fn,
        write_fn=write_fn,
    )
    assert state_out["final_report"].startswith("# Report")
    assert manifest_out["completed_tasks"] == manifest_out["total_tasks"]
    assert all(node["status"] == "COMPLETED" for node in manifest_out["nodes"])


def test_p2a_all_reader_modes_read_multiple_pdfs_with_provenance(tmp_path):
    import os
    from modes.gpu_finetune.reader import GPUFinetuneReaderAgent
    from modes.gpu_max.reader import GPUMaxReaderAgent

    pdf_a = tmp_path / "paper_a.pdf"
    pdf_b = tmp_path / "paper_b.pdf"
    with fitz.open() as doc_a:
        doc_a.new_page().insert_text((72, 72), "Paper A reports 94.5% precision on dataset X.")
        doc_a.save(pdf_a)
    with fitz.open() as doc_b:
        doc_b.new_page().insert_text((72, 72), "Paper B reports 88.1% precision on dataset Y.")
        doc_b.save(pdf_b)

    state = {
        "query": "Compare precision of Paper A and Paper B",
        "pdf_path": str(pdf_a),
        "pdf_paths": [str(pdf_a), str(pdf_b)],
        "retrieved_docs": [
            {"id": "chunk_a1", "doc_id": "paper_a.pdf", "page_no": 1, "text": "Paper A reports 94.5% precision on dataset X."}
        ],
    }
    for agent_cls in (GPUFinetuneReaderAgent, GPUMaxReaderAgent):
        out = agent_cls().run(state)
        units = out["source_units"]
        doc_stems = {os.path.splitext(str(u.get("doc_id") or ""))[0] for u in units}
        assert "paper_a" in doc_stems
        assert "paper_b" in doc_stems
        assert all(u.get("provenance") for u in units)


def test_p2b_deterministic_sha256_claim_ids():
    from core.state import AtomicClaim, ClaimCheckResult

    c1 = ClaimCheckResult(claim_text="The model achieves 91.2% accuracy.")
    c2 = ClaimCheckResult(claim_text="  The model achieves 91.2% accuracy.  ")
    assert c1.claim_id == c2.claim_id
    assert c1.claim_id.startswith("claim_")
    assert len(c1.claim_id) == len("claim_") + 12

    a1 = AtomicClaim(text="The model achieves 91.2% accuracy.")
    a2 = AtomicClaim(text="The model achieves 91.2% accuracy.")
    assert a1.id == a2.id == c1.claim_id

