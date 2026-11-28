"""
Comprehensive Acceptance Suite for Phase 1 & Phase 2 Multi-Agent Hub Architecture:
  1. Zero dead edges: `ALLOWED_HANDOFF_TARGETS == ROLE_DECISION_TARGETS` across all 6 specialists + Supervisor.
  2. True Hub-and-Spoke routing: every specialist returns to `supervisor_node` with typed `AgentResult` + `AgentDecision`.
  3. All 6 agents independently propose decisions (Scout, Retrieval, Reader, Summarizer, Fact-Checker, Writer)
     and Supervisor Gatekeeper approves or rejects with exact reasons.
  4. Real FastAPI `/api/runs` + real PyMuPDF PDF tests in `supervisor_mode="hub"`:
     - Distant Table 4 autonomous `Reader -> Retrieval` loop
     - Multi-document `comparison_plan=True` via `comparison_review_node` in Hub mode
     - Cumulative SQLite checkpoint persistence of `agent_messages`, `handoff_budgets`, `supervisor_mode="hub"` + resume
  5. Single Unified Planner: Supervisor invokes Kaggle `/api/router`; Retrieval never re-plans when Supervisor already planned.
  6. Selective LLM Arbiter: 0 calls on fast-path queries, bounded structured arbitration only on ambiguous conflict.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

import fitz
import pytest
from fastapi.testclient import TestClient

from api import app as api_module, auth
from core import db
from core.agent_decisions import ROLE_DECISION_TARGETS, decide_for_role
from core.config import settings
from core.coordination import ALLOWED_HANDOFF_TARGETS
from core.graph import build_research_rag_graph, orchestrator_router_node, resolve_next_node_after_checkpoint
from core.tools import clear_pdf_cache


def _make_html_pdf(path: Path, html_pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for html in html_pages:
            page = doc.new_page(width=595, height=842)
            page.insert_htmlbox(
                (48, 48, 547, 794),
                f'<div style="font-family:sans-serif; font-size:10.5pt; line-height:1.4;">{html}</div>',
            )
        doc.save(str(path))
    return path


def _poll_run(client: TestClient, run_id: str, timeout: float = 25.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] != "processing":
            return data
        time.sleep(0.03)
    pytest.fail(f"Run {run_id} did not complete within {timeout}s")


@pytest.fixture
def hub_api_client(monkeypatch, research_db, tmp_path):
    clear_pdf_cache()
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "hub-secret-test")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module.settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api_module.settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    monkeypatch.setattr(api_module, "_legacy_mode_override", "cpu_min")
    with TestClient(api_module.app, base_url="https://workspace.test") as client:
        assert client.post("/api/auth/login", json={"password": "hub-secret-test"}).status_code == 200
        yield client
    clear_pdf_cache()


def test_zero_dead_branches_policy_matrix_matches_role_deciders():
    assert ALLOWED_HANDOFF_TARGETS == ROLE_DECISION_TARGETS
    for role in ("scout", "retrieval", "reader", "summarizer", "fact_checker", "writer"):
        assert role in ROLE_DECISION_TARGETS


def test_all_six_specialists_emit_mandatory_decisions_and_hub_routes_every_step(monkeypatch):
    graph_mod = importlib.import_module("core.graph")
    calls: list[str] = []

    runners = {
        "retrieval": lambda s: calls.append("retrieval") or {"retrieved_docs": [{"id": "c1", "doc_id": "d1", "page_no": 1, "text": "Yield was 14.6 ms."}]},
        "reader": lambda s: calls.append("reader") or {"source_units": [{"id": "c1", "doc_id": "d1", "page_no": 1, "text": "Yield was 14.6 ms."}]},
        "summarizer": lambda s: calls.append("summarizer") or {"answer": "Yield was 14.6 ms [1].", "claims": [{"id": "cl1", "text": "Yield was 14.6 ms.", "evidence_ids": ["c1"]}]},
        "fact_checker": lambda s: calls.append("fact_checker") or {"fact_check_label": "support", "verified_claims": [{"id": "cl1", "claim_text": "Yield was 14.6 ms.", "status": "SUPPORT", "evidence_ids": ["c1"]}], "rejected_claims": []},
        "writer": lambda s: calls.append("writer") or {"final_report": "Yield was 14.6 ms [1]."},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, s: calls.append("scout") or {"pdf_path": "d1.pdf", "pdf_paths": ["d1.pdf"]})

    arbiter_calls: list[str] = []
    app = build_research_rag_graph(
        "cpu_min",
        supervisor_mode="hub",
        arbiter_llm_fn=lambda p: arbiter_calls.append(p) or '{"next_role":"writer"}',
    )
    out = app.invoke({"query": "What is the yield?", "pdf_paths": ["d1.pdf"], "max_reflection_loops": 2})

    assert calls == ["scout", "retrieval", "reader", "summarizer", "fact_checker", "writer"]
    assert out["supervisor_mode"] == "hub"
    # Fast-path deterministic routing: 0 extra LLM arbiter calls on clear progression!
    assert arbiter_calls == []

    hub_dispatches = [e for e in out["reasoning_trace"] if e.get("action") == "supervisor_hub_dispatched"]
    dispatched_roles = [e["next_role"] for e in hub_dispatches]
    assert dispatched_roles[:6] == ["scout", "retrieval", "reader", "summarizer", "fact_checker", "writer"]

    # Compact blackboard messages captured for all 6 agents + supervisor
    senders = {m["sender"] for m in out["agent_messages"]}
    assert {"supervisor", "scout", "retrieval", "reader", "summarizer", "fact_checker", "writer"}.issubset(senders)


def test_hub_autonomous_retrieval_to_scout_and_strategy_switch(monkeypatch, tmp_path):
    graph_mod = importlib.import_module("core.graph")
    pdf1 = _make_html_pdf(tmp_path / "p1.pdf", ["<p>Unrelated background overview without metrics.</p>"])
    pdf2 = _make_html_pdf(tmp_path / "p2.pdf", ["<p>Quantum-X throughput reached 982 ops/s.</p>"])

    retrieval_runs: list[dict[str, Any]] = []
    scout_runs: list[str] = []

    def fake_retrieval(state):
        retrieval_runs.append({"retry": state.get("retry_count", 0), "strategy": state.get("retrieval_strategy"), "pdfs": list(state.get("pdf_paths") or [])})
        if str(pdf2) in (state.get("pdf_paths") or []):
            return {"retrieved_docs": [{"id": "c_new", "doc_id": pdf2.stem, "page_no": 1, "text": "Quantum-X throughput reached 982 ops/s."}], "no_new_evidence": False}
        # First pass returns empty -> triggers autonomous retrieval -> scout when allow_scout_handoff=True
        return {"retrieved_docs": [], "no_new_evidence": True}

    def fake_scout(self, state):
        scout_runs.append(state.get("query", ""))
        if not state.get("pdf_paths"):
            return {"pdf_path": str(pdf2), "pdf_paths": [str(pdf2)], "scout_decision": "success"}
        return {"pdf_path": str(pdf1), "pdf_paths": [str(pdf1)]}

    runners = {
        "retrieval": fake_retrieval,
        "reader": lambda s: {"source_units": s.get("retrieved_docs") or []},
        "summarizer": lambda s: {"answer": "Quantum-X throughput reached 982 ops/s [1].", "claims": [{"id": "c1", "text": "Quantum-X throughput reached 982 ops/s.", "evidence_ids": ["c_new"]}]},
        "fact_checker": lambda s: {"fact_check_label": "support", "verified_claims": [{"id": "c1", "claim_text": "Quantum-X throughput reached 982 ops/s.", "status": "SUPPORT", "evidence_ids": ["c_new"]}], "rejected_claims": []},
        "writer": lambda s: {"final_report": s.get("answer", "")},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", fake_scout)

    app = build_research_rag_graph("cpu_min", supervisor_mode="hub", auto_scout=False, allow_scout_handoff=True)
    out = app.invoke({"query": "What is Quantum-X throughput?", "pdf_paths": [str(pdf1)], "max_reflection_loops": 2})

    assert len(retrieval_runs) == 2
    assert len(scout_runs) == 2
    handoffs = [e for e in out["reasoning_trace"] if e.get("action") == "agent_handoff"]
    assert any(h["sender"] == "retrieval" and h["recipient"] == "scout" for h in handoffs)
    assert "982" in out["final_report"]

    # Also verify retrieval -> retrieval strategy switch decision directly
    switch_dec = decide_for_role(
        "retrieval",
        {
            "query": "What is Hybrid-X latency?",
            "required_entities": ["Hybrid-X"],
            "pdf_paths": [str(pdf1)],
            "retry_count": 0,
            "enable_retrieval_strategy_switch": True,
        },
        {"retrieved_docs": [{"id": "x", "text": "Unrelated prose only."}]},
    )
    assert switch_dec.action == "request_more_evidence"
    assert switch_dec.target_agent == "retrieval"
    assert switch_dec.parameters.get("strategy_hint") == "broad_lexical_bilingual"


def test_hub_autonomous_writer_triggers_summarizer_revision_on_citation_audit_failure(monkeypatch):
    graph_mod = importlib.import_module("core.graph")
    sum_calls = {"n": 0}
    writer_calls = {"n": 0}

    def fake_summarizer(state):
        sum_calls["n"] += 1
        span = {
            "id": "ev_u1",
            "source_unit_id": "u1",
            "unit_id": "u1",
            "doc_id": "paperA",
            "page_no": 1,
            "quote": "Measured latency is 14.6 ms.",
        }
        if sum_calls["n"] == 1:
            # First draft has a hallucinated number (777 ms) -> Writer's citation audit flags missing_numeric_evidence
            return {
                "answer": "Measured latency is 777 ms [1].",
                "evidence_spans": [span],
                "claims": [{"id": "c1", "text": "Measured latency is 777 ms.", "evidence_ids": ["ev_u1"]}],
            }
        # Revised draft matches audited span strictly (14.6 ms)
        return {
            "answer": "Measured latency is 14.6 ms [1].",
            "evidence_spans": [span],
            "claims": [{"id": "c1", "text": "Measured latency is 14.6 ms.", "evidence_ids": ["ev_u1"]}],
        }

    def fake_checker(state):
        # Simulate passing checker so Writer's own citation_audit catches the ungrounded claim!
        return {
            "fact_check_label": "support",
            "verified_claims": list(state.get("claims") or []),
            "rejected_claims": [],
        }

    def fake_writer(state):
        writer_calls["n"] += 1
        return {"final_report": state.get("answer", "")}

    runners = {
        "retrieval": lambda s: {"retrieved_docs": [{"id": "u1", "doc_id": "paperA", "page_no": 1, "text": "Measured latency is 14.6 ms."}]},
        "reader": lambda s: {"source_units": [{"id": "u1", "doc_id": "paperA", "page_no": 1, "text": "Measured latency is 14.6 ms."}]},
        "summarizer": fake_summarizer,
        "fact_checker": fake_checker,
        "writer": fake_writer,
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, s: {"pdf_path": "paperA.pdf", "pdf_paths": ["paperA.pdf"]})

    app = build_research_rag_graph("cpu_min", supervisor_mode="hub")
    out = app.invoke({"query": "What is latency?", "pdf_paths": ["paperA.pdf"], "max_reflection_loops": 2})

    assert writer_calls["n"] == 2
    assert sum_calls["n"] == 2
    handoffs = [e for e in out["reasoning_trace"] if e.get("action") == "agent_handoff"]
    assert any(h["sender"] == "writer" and h["recipient"] == "summarizer" for h in handoffs)
    assert "14.6" in out["final_report"]

    # Also verify writer -> fact_checker decision when fact_check_label wasn't supported
    w_dec = decide_for_role(
        "writer",
        {
            "query": "What is latency?",
            "source_units": [{"id": "u1", "text": "14.6 ms"}],
            "evidence_spans": [{"id": "ev1", "quote": "14.6 ms"}],
            "fact_check_label": "not_enough_info",
        },
        {
            "verified_claims": [{"id": "c1", "text": "Latency is 999 ms"}],
            "reasoning_trace": [
                {
                    "action": "citation_audit",
                    "citation_audit": {
                        "claims": {"c1": {"status": "missing_numeric_evidence", "evidence_ids": []}},
                        "references": [],
                    },
                }
            ],
        },
    )
    assert w_dec.action == "request_verification"
    assert w_dec.target_agent == "fact_checker"


def test_single_unified_planner_calls_kaggle_router_and_retrieval_does_not_replan(monkeypatch):
    from modes.gpu_finetune.retrieval import GPUFinetuneRetrievalAgent
    from modes.hybrid.retrieval import HybridRetrievalAgent
    from utils.llm_client import llm_client

    router_calls: list[str] = []

    def fake_kaggle_specialist(*, endpoint, payload, **kwargs):
        router_calls.append(endpoint)
        return {
            "intent": "COMPARISON",
            "sub_queries": ["Compare Method A accuracy", "Compare Method B latency"],
            "alpha": 0.4,
            "top_k": 8,
            "reasoning": "Kaggle router tool plan",
        }

    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://kaggle.test")
    monkeypatch.setattr(llm_client, "call_kaggle_specialist", fake_kaggle_specialist)

    planned = orchestrator_router_node(
        {"query": "Compare Method A and Method B latency and accuracy across benchmarks"},
        execution_mode="gpu_finetune",
    )
    assert router_calls == ["api/router"]
    assert planned["document_schema"]["planner_source"] == "kaggle_router_tool"
    assert "Compare Method A accuracy" in planned["sub_queries"]

    # Now Retrieval receives Supervisor's state -> MUST NOT call `/api/router` or Gemini planning again!
    router_calls.clear()
    monkeypatch.setattr("modes.hybrid.retrieval.probe_all_capabilities", lambda: {"gemini": {"online": True}, "gpu_finetune": {"online": True}})
    monkeypatch.setattr(llm_client, "generate_json", lambda *a, **kw: pytest.fail("Retrieval must not re-run Gemini planner!"))

    hybrid_ret = HybridRetrievalAgent(enable_dense=False, enable_rerank=False)
    h_out = hybrid_ret.run({**planned, "query": "Compare Method A and Method B latency"})
    assert router_calls == []
    assert "supervisor:kaggle_router_tool" in str(h_out["reasoning_trace"][-1]["details"]["planning_backend"])

    gpu_ret = GPUFinetuneRetrievalAgent(enable_dense=False, enable_rerank=False)
    gpu_ret.run({**planned, "query": "Compare Method A and Method B latency"})
    assert router_calls == []


def test_hub_real_pdf_api_reader_handoff_comparison_and_checkpoint_accumulation(
    hub_api_client: TestClient, tmp_path: Path
):
    # 1. Real 10-page PDF distant Table 4 via FastAPI `/api/runs` in `supervisor_mode="hub"`
    table4 = (
        "<p><b>Table 4: Empirical Latency Breakdown</b></p>"
        "<table border='1'><tr><th>Engine</th><th>p50 (ms)</th><th>p99 (ms)</th></tr>"
        "<tr><td>Hybrid-X</td><td>8.2</td><td>14.6</td></tr></table>"
    )
    pages = [
        (
            "<h3>Section 1: Production Load Latency Overview</h3>"
            "<p>This paper evaluates the p99 latency of Hybrid-X under production load across multi-tenant clusters. "
            "For exact empirical p99 latency measurements of Hybrid-X under production load, please see Table 4.</p>"
        ),
        (
            "<h3>Section 2: Production Load Traffic Model for Hybrid-X</h3>"
            "<p>Our production load generator replays anonymized search queries to profile p99 latency of Hybrid-X "
            "without warm-cache bias. All production load traces keep arrival intervals steady while measuring Hybrid-X p99 latency.</p>"
        ),
        (
            "<h3>Section 3: Tail Latency Instrumentation for Hybrid-X</h3>"
            "<p>To capture p99 latency of Hybrid-X under production load accurately, each worker records monotonic timestamps "
            "at request admission and response flush. Hybrid-X production load experiments isolate queuing delay from parsing overhead.</p>"
        ),
        (
            "<h3>Section 4: Cluster Hardware for Hybrid-X Production Load</h3>"
            "<p>Every Hybrid-X production load run executes on isolated nodes so p99 latency reflects deterministic scheduling "
            "rather than noisy-neighbor contention. Production load monitors verify zero packet loss during Hybrid-X p99 latency tests.</p>"
        ),
        (
            "<h3>Section 5: Warmup Protocol for Hybrid-X</h3>"
            "<p>Before recording Hybrid-X p99 latency under production load, each instance completes a warmup pass so JIT caches stabilize. "
            "Production load benchmarks confirm reproducible Hybrid-X p99 latency distributions across repeated seeds.</p>"
        ),
        (
            "<h3>Section 6: Baseline Comparison Methodology</h3>"
            "<p>We compare Hybrid-X against baseline engines under identical production load to study p99 latency behavior. "
            "Qualitative latency trade-offs for Hybrid-X production load pipelines are discussed before reporting exact numbers.</p>"
        ),
        f"<h3>Appendix A</h3>{table4}",
        (
            "<h3>Section 7: Production Load Error Analysis for Hybrid-X</h3>"
            "<p>Under bursty production load, Hybrid-X p99 latency remains stable because bounded queues shed stale retries. "
            "Operators tuning Hybrid-X for production load prioritize tail p99 latency predictability.</p>"
        ),
        (
            "<h3>Section 8: Related Work on Hybrid-X Tail Latency</h3>"
            "<p>Prior retrieval studies examine production load p99 latency on single-stage indexes, whereas Hybrid-X coordinates "
            "structured verification under production load to guard p99 latency budgets.</p>"
        ),
        (
            "<h3>Section 9: Conclusions on Hybrid-X Production Load</h3>"
            "<p>Overall, Hybrid-X delivers dependable p99 latency under production load across heterogeneous workloads. "
            "Future work will test Hybrid-X p99 latency under geo-distributed production load.</p>"
        ),
    ]
    pdf_case2 = _make_html_pdf(tmp_path / "hub_case2.pdf", pages)
    db.save_document("doc_hub_c2", pdf_case2.name, str(pdf_case2), pdf_case2.stat().st_size)

    resp = hub_api_client.post(
        "/api/runs",
        json={
            "question": "What is the p99 latency of Hybrid-X under production load?",
            "document_ids": ["doc_hub_c2"],
            "mode": "cpu_min",
            "supervisor_mode": "hub",
            "max_iterations": 2,
            "auto_scout": False,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(hub_api_client, resp.json()["id"])
    assert run["status"] == "completed"
    assert "14.6" in run["final_report"]

    trace = run["reasoning_trace"]
    assert any(e.get("action") == "supervisor_hub_dispatched" for e in trace)
    assert any(e.get("action") == "agent_handoff" and e.get("sender") == "reader" and e.get("recipient") == "retrieval" for e in trace)

    # Verify cumulative SQLite checkpoint stored agent_messages + supervisor_mode="hub" + handoff_budgets
    ckpt = db.get_latest_run_checkpoint(run["id"])
    assert ckpt is not None
    ckpt_state = ckpt["state"]
    assert ckpt_state["supervisor_mode"] == "hub"
    assert len(ckpt_state.get("agent_messages") or []) >= 6
    assert ckpt_state.get("handoff_budgets", {}).get("evidence", 0) >= 1
    assert resolve_next_node_after_checkpoint("reader_node", ckpt_state) == "supervisor_node"

    # 2. Real 2-PDF comparison_plan=True via FastAPI `/api/runs` in `supervisor_mode="hub"`
    pdf_a = _make_html_pdf(tmp_path / "comp_a.pdf", ["<p>Paper A reports Model-Alpha accuracy of 91.2% on ImageNet.</p>"])
    pdf_b = _make_html_pdf(tmp_path / "comp_b.pdf", ["<p>Paper B reports Model-Beta accuracy of 94.5% on ImageNet.</p>"])
    db.save_document("doc_hub_a", pdf_a.name, str(pdf_a), pdf_a.stat().st_size)
    db.save_document("doc_hub_b", pdf_b.name, str(pdf_b), pdf_b.stat().st_size)

    comp_resp = hub_api_client.post(
        "/api/runs",
        json={
            "question": "Compare Model-Alpha and Model-Beta accuracy on ImageNet.",
            "document_ids": ["doc_hub_a", "doc_hub_b"],
            "mode": "cpu_min",
            "supervisor_mode": "hub",
            "comparison_plan": True,
            "max_iterations": 2,
            "auto_scout": False,
        },
    )
    assert comp_resp.status_code == 200
    comp_run = _poll_run(hub_api_client, comp_resp.json()["id"])
    assert comp_run["status"] == "completed"
    assert any(
        e.get("action") == "supervisor_hub_dispatched" and e.get("next_role") == "comparison_review"
        for e in comp_run["reasoning_trace"]
    )
    assert "91.2" in comp_run["final_report"] and "94.5" in comp_run["final_report"]


def test_hub_selective_llm_arbiter_invoked_only_on_ambiguous_conflict(monkeypatch):
    graph_mod = importlib.import_module("core.graph")
    arbiter_calls: list[str] = []

    runners = {
        "retrieval": lambda s: {"retrieved_docs": [{"id": "u1", "doc_id": "d1", "page_no": 1, "text": "Alpha achieved 90%."}]},
        "reader": lambda s: {"source_units": [{"id": "u1", "doc_id": "d1", "page_no": 1, "text": "Alpha achieved 90%."}]},
        "summarizer": lambda s: {"answer": "Alpha achieved 90%.", "claims": [{"id": "c1", "text": "Alpha achieved 90%.", "evidence_ids": ["u1"]}]},
        "fact_checker": lambda s: {
            "fact_check_label": "not_enough_info",
            "verified_claims": [{"id": "c1", "claim_text": "Alpha achieved 90%.", "status": "SUPPORT", "evidence_ids": ["u1"]}],
            "rejected_claims": [{"id": "c2", "claim_text": "Beta achieved 50%.", "status": "NOT_ENOUGH_INFO"}],
            "evidence_gaps": [{"missing_aspect": "Beta accuracy", "suggested_query": "Beta accuracy"}],
        },
        "writer": lambda s: {"final_report": "Published verified subset [1]."},
    }
    monkeypatch.setattr(graph_mod, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_mod.ScoutAgent, "run", lambda self, s: {"pdf_path": "d1.pdf", "pdf_paths": ["d1.pdf"]})

    # With max_reflection_loops=0, corrective handoff is rejected -> Supervisor faces ambiguous conflict
    # (verified + rejected + gaps) and invokes arbiter_llm_fn, which picks "writer".
    app = build_research_rag_graph(
        "cpu_min",
        supervisor_mode="hub",
        arbiter_llm_fn=lambda p: arbiter_calls.append(p) or json.dumps({"next_role": "writer", "rationale": "verified_subset_ready"}),
    )
    out = app.invoke({"query": "Compare Alpha and Beta", "pdf_paths": ["d1.pdf"], "max_reflection_loops": 0})

    assert len(arbiter_calls) == 1
    assert any(m.get("kind") == "arbitration" and m.get("recipient") == "writer" for m in out["agent_messages"])
    assert "Published verified subset" in out["final_report"]


def test_hub_checkpoint_interrupt_and_resume_preserves_messages_and_finishes(
    hub_api_client: TestClient, monkeypatch, tmp_path: Path
):
    """
    Simulates a mid-run crash inside `summarizer_node` in `supervisor_mode="hub"`,
    then calls `POST /api/runs/{id}/resume` and verifies:
      1. Completed nodes (`retrieval`, `reader`) are NOT re-executed on resume.
      2. Cumulative `agent_messages` from pre-crash checkpoints are preserved across resume.
      3. Resumed run completes cleanly through `supervisor_node` and publishes the verified figure.
    """
    graph_mod = importlib.import_module("core.graph")
    pdf = _make_html_pdf(
        tmp_path / "hub_resume.pdf",
        ["<p>Resilient-Net achieved 96.4% top-1 accuracy on Benchmark-Z.</p>"],
    )
    db.save_document("doc_hub_resume", pdf.name, str(pdf), pdf.stat().st_size)

    should_crash = {"flag": True}
    phase2_calls: list[str] = []
    orig_get_runners = graph_mod.get_agent_runners

    def wrapped_runners(mode):
        runners = orig_get_runners(mode)
        orig_ret, orig_read, orig_sum = runners["retrieval"], runners["reader"], runners["summarizer"]

        def ret_spy(s):
            if not should_crash["flag"]:
                phase2_calls.append("retrieval")
            return orig_ret(s)

        def read_spy(s):
            if not should_crash["flag"]:
                phase2_calls.append("reader")
            return orig_read(s)

        def sum_crash_once(s):
            if should_crash["flag"]:
                raise RuntimeError("Simulated mid-run crash in Hub summarizer")
            phase2_calls.append("summarizer")
            return orig_sum(s)

        return {**runners, "retrieval": ret_spy, "reader": read_spy, "summarizer": sum_crash_once}

    monkeypatch.setattr(graph_mod, "get_agent_runners", wrapped_runners)

    queued = hub_api_client.post(
        "/api/runs",
        json={
            "question": "What accuracy did Resilient-Net achieve on Benchmark-Z?",
            "document_ids": ["doc_hub_resume"],
            "mode": "cpu_min",
            "supervisor_mode": "hub",
            "max_iterations": 0,
            "auto_scout": False,
        },
    )
    assert queued.status_code == 200
    failed_run = _poll_run(hub_api_client, queued.json()["id"])
    assert failed_run["status"] == "failed"

    pre_ckpt = db.get_latest_run_checkpoint(failed_run["id"])
    assert pre_ckpt is not None
    assert pre_ckpt["state"]["supervisor_mode"] == "hub"
    pre_msg_count = len(pre_ckpt["state"].get("agent_messages") or [])
    assert pre_msg_count >= 4

    # Phase 2: Resume from checkpoint without re-running retrieval or reader!
    should_crash["flag"] = False
    resumed = hub_api_client.post(f"/api/runs/{failed_run['id']}/resume")
    assert resumed.status_code == 200
    done = _poll_run(hub_api_client, resumed.json()["id"])
    assert done["status"] == "completed"
    assert "96.4" in done["final_report"]
    assert "retrieval" not in phase2_calls
    assert "reader" not in phase2_calls
    assert "summarizer" in phase2_calls

    post_ckpt = db.get_latest_run_checkpoint(done["id"])
    assert post_ckpt is not None
    assert len(post_ckpt["state"].get("agent_messages") or []) > pre_msg_count

