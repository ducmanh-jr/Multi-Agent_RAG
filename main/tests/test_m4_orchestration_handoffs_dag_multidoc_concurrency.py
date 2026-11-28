"""
Milestone M4 Regression & Multi-Document Accuracy Suite:
1. Retrieval -> Scout AgentDecision wired through real LangGraph & FastAPI API (comparison_plan on/off, auto_scout on/off),
   recording full lifecycle trace: proposed/evaluated -> approved/rejected -> received -> outcome.
2. Unified single dispatcher (`dispatch_next_node`) shared identically by fresh runs, retries, and `resume`
   across all 6 specialist roles (`scout`, `retrieval`, `reader`, `summarizer`, `fact_checker`, `writer`).
3. Strict `task_id`-only DAG manifest synchronization (waiting parallel branches NEVER prematurely completed).
4. Multi-document per-document fan-out across 6 distinct bilingual PDFs (preserving full source lineage & accuracy)
   through both UI entry (`parallel_retrieval=True`) and default API entry (`parallel_retrieval=False`),
   plus 6-PDF `comparison_plan=True`.
5. Real 6-paper arXiv academic corpus test (`colbert`, `contriever`, `dpr`, `fid`, `rag`, `realm`, with clean-checkout fallback).
6. Mid-run worker crash/interrupt + truthful `POST /api/runs/{run_id}/resume` (no re-running completed nodes, custom agent_modes).
7. Timeout enforcement & shared budget-capped retries with `no_new_evidence` early stop.
8. Two concurrent multi-document (3+3 = 6 PDFs) API runs with overlapping identical filenames (`shared_study.pdf`)
   verifying zero document/cache/state leakage.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
from pathlib import Path
import time
from threading import Barrier
from typing import Any

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

from api import app as api_module, auth, research_runs as service_module
from core import db, graph as graph_module
from core.config import RunConfig, settings
from core.coordination import AgentDecision, evaluate_agent_handoff
from core.graph import (
    build_research_rag_graph,
    dispatch_next_node,
    resolve_next_node_after_checkpoint,
)
from core.research_dag import (
    ResearchDAGBudget,
    build_research_dag_plan,
    execute_document_fanout_dag,
    sync_dag_manifest_from_retrieval,
)
from core.research_tasks import TaskStatus
from core.supervisor import supervisor_planner
from modes.cpu_min.retrieval import CPUMinRetrievalAgent


def _make_pdf(path: Path, pages: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with fitz.open() as doc:
        for text in pages:
            page = doc.new_page(width=595, height=842)
            page.insert_htmlbox(
                (54, 54, 540, 780),
                f'<div style="font-size:11pt; line-height:1.35;">{text}</div>',
            )
        doc.save(str(path))
    return path


def _poll_run(client: TestClient, run_id: str, timeout: float = 20.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        resp = client.get(f"/api/runs/{run_id}")
        assert resp.status_code == 200
        data = resp.json()
        if data["status"] != "processing":
            return data
        time.sleep(0.02)
    pytest.fail(f"Run {run_id} did not finish within {timeout}s")


@pytest.fixture
def api_client(monkeypatch, research_db, tmp_path):
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "m4-test-secret")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module.settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api_module, "_legacy_mode_override", "cpu_min")
    with TestClient(api_module.app, base_url="https://workspace.test") as client:
        assert client.post("/api/auth/login", json={"password": "m4-test-secret"}).status_code == 200
        yield client


@pytest.mark.parametrize("auto_scout", [True, False])
@pytest.mark.parametrize("comparison_plan", [False, True])
def test_m4_retrieval_to_scout_handoff_wired_in_graph_and_api_with_comparison_on_off(
    api_client, monkeypatch, tmp_path, auto_scout, comparison_plan
):
    """
    Reproduces and verifies the M4 P1 orchestration fix:
    When Retrieval emits an AgentDecision proposing `scout` (`discover_additional_literature`),
    - if `auto_scout=True`: Supervisor approves, LangGraph routes `retrieval_node -> scout_node`,
      Scout discovers the missing PDF, Retrieval re-runs with both PDFs, and full lifecycle traces
      (`supervisor_decision_evaluated` -> `agent_handoff` -> `agent_handoff_received` -> `handoff_outcome_recorded`)
      are persisted.
    - if `auto_scout=False`: Supervisor rejects (`scout_handoff_disabled`), Scout is NOT re-invoked,
      and execution continues cleanly to Reader without external discovery.
    """
    local_pdf = _make_pdf(tmp_path / "local_base.pdf", ["Local study reports baseline yield of 41 tons/ha."])
    scouted_pdf = _make_pdf(tmp_path / "scouted_extra.pdf", ["External follow-up study reports upgraded yield of 67 tons/ha."])
    db.save_document("doc_local", local_pdf.name, str(local_pdf), local_pdf.stat().st_size)

    calls: list[str] = []
    scout_discovery_calls = {"count": 0}

    def fake_scout_run(self, state):
        selected = state.get("pdf_paths") or ([state["pdf_path"]] if state.get("pdf_path") else [])
        if selected:
            calls.append("scout_bypass")
            return {"pdf_path": selected[0], "pdf_paths": selected, "reasoning_trace": [{"action": "bypass_discovery"}]}
        calls.append("scout_discover")
        scout_discovery_calls["count"] += 1
        db.save_document("doc_scouted", scouted_pdf.name, str(scouted_pdf), scouted_pdf.stat().st_size)
        return {
            "scout_decision": "success",
            "pdf_path": str(scouted_pdf),
            "pdf_paths": [str(scouted_pdf)],
            "reasoning_trace": [{"action": "paper_discovered_and_ingested", "path": str(scouted_pdf)}],
        }

    def fake_retrieval(state):
        calls.append("retrieval")
        paths = list(state.get("pdf_paths") or [])
        if len(paths) == 1 and state.get("retry_count", 0) == 0:
            return {
                "retrieved_docs": [
                    {"id": "u_local", "doc_id": "local_base", "page_no": 1, "text": "Local study reports baseline yield of 41 tons/ha."}
                ],
                "agent_decision": {
                    "action": "discover_additional_literature",
                    "target_agent": "scout",
                    "reason": "Missing external follow-up yield study",
                    "suggested_queries": ["external follow-up yield study"],
                    "task_id": "investigate_1",
                },
            }
        docs = []
        if any(Path(p).name == "local_base.pdf" for p in paths):
            docs.append(
                {"id": "u_local", "doc_id": "local_base", "page_no": 1, "text": "Local study reports baseline yield of 41 tons/ha."}
            )
        if any(Path(p).name == "scouted_extra.pdf" for p in paths):
            docs.append(
                {"id": "u_scouted", "doc_id": "scouted_extra", "page_no": 1, "text": "External follow-up study reports upgraded yield of 67 tons/ha."}
            )
        return {"retrieved_docs": docs, "agent_decision": None}

    def fake_reader(state):
        calls.append("reader")
        return {"source_units": list(state.get("retrieved_docs") or []), "document_schema": {}}

    def fake_summarizer(state):
        calls.append("summarizer")
        units = list(state.get("source_units") or [])
        combined = " ".join(u["text"] for u in units)
        spans = [
            {"id": f"ev_{i}", "source_unit_id": u["id"], "doc_id": u["doc_id"], "page_no": 1, "quote": u["text"]}
            for i, u in enumerate(units, 1)
        ]
        claims = [
            {"id": f"c_{i}", "text": u["text"], "evidence_ids": [f"ev_{i}"]}
            for i, u in enumerate(units, 1)
        ]
        return {"answer": combined, "claims": claims, "evidence_spans": spans}

    def fake_checker(state):
        calls.append("fact_checker")
        claims = list(state.get("claims") or [])
        return {
            "fact_check_label": "support",
            "verified_claims": [{**c, "status": "SUPPORT"} for c in claims],
            "rejected_claims": [],
            "evidence_gaps": [],
        }

    from modes.cpu_min.writer import CPUMinWriterAgent

    real_writer = CPUMinWriterAgent()

    def fake_writer(state):
        calls.append("writer")
        return real_writer.run(state)

    monkeypatch.setattr(graph_module.ScoutAgent, "run", fake_scout_run)
    monkeypatch.setattr(
        graph_module,
        "get_agent_runners",
        lambda mode: {
            "retrieval": fake_retrieval,
            "reader": fake_reader,
            "summarizer": fake_summarizer,
            "fact_checker": fake_checker,
            "writer": fake_writer,
        },
    )
    monkeypatch.setattr(graph_module, "add_figure_evidence", lambda inputs, update, mode: update)

    resp = api_client.post(
        "/api/runs",
        json={
            "question": "Compare local baseline yield with external follow-up yield.",
            "document_ids": ["doc_local"],
            "auto_scout": auto_scout,
            "comparison_plan": comparison_plan,
            "mode": "cpu_min",
            "max_iterations": 2,
        },
    )
    assert resp.status_code == 200
    run = _poll_run(api_client, resp.json()["id"])
    assert run["status"] == "completed"
    trace = run["reasoning_trace"]

    eval_events = [e for e in trace if e.get("action") == "supervisor_decision_evaluated" and e.get("source_agent") == "retrieval"]
    assert len(eval_events) == 1
    assert eval_events[0]["task_id"] == "investigate_1"

    if auto_scout:
        assert eval_events[0]["approved"] is True
        assert scout_discovery_calls["count"] == 1
        assert calls[:4] == ["scout_bypass", "retrieval", "scout_discover", "retrieval"]
        assert "67 tons/ha" in run["final_report"]
        assert len(run["documents"]) == 2
        # Verify complete lifecycle: evaluated -> agent_handoff -> agent_handoff_received -> handoff_outcome_recorded
        handoffs = [e for e in trace if e.get("action") == "agent_handoff" and e.get("sender") == "retrieval"]
        receipts = [e for e in trace if e.get("action") == "agent_handoff_received" and e.get("recipient") == "scout"]
        outcomes = [e for e in trace if e.get("action") == "handoff_outcome_recorded" and e.get("recipient") == "scout"]
        assert len(handoffs) == 1 and handoffs[0]["recipient"] == "scout" and handoffs[0]["task_id"] == "investigate_1"
        assert len(receipts) == 1 and receipts[0]["signature"] == handoffs[0]["signature"]
        assert len(outcomes) == 1 and outcomes[0]["outcome_status"] == "discovered_new_documents"
    else:
        assert eval_events[0]["approved"] is False
        assert eval_events[0]["rejection_reason"] == "scout_handoff_disabled"
        assert scout_discovery_calls["count"] == 0
        assert "scout_discover" not in calls
        assert "41 tons/ha" in run["final_report"]
        assert "67 tons/ha" not in run["final_report"]


def test_m4_unified_six_role_dispatch_matches_fresh_run_and_checkpoint_resume(monkeypatch, tmp_path):
    """
    Verifies a SINGLE dispatch function (`dispatch_next_node`) governs fresh execution, retry loops,
    and `resolve_next_node_after_checkpoint` identically across all 6 roles (including Writer handoff!).
    """
    samples = [
        ("orchestrator_router", {}, False, "scout_node"),
        ("scout_node", {"reasoning_trace": [{"action": "bypass_discovery"}]}, False, "retrieval_node"),
        ("scout_node", {"reasoning_trace": [{"action": "discovery_failed"}]}, False, "__end__"),
        (
            "retrieval_node",
            {"reasoning_trace": [{"action": "agent_handoff", "sender": "retrieval", "recipient": "scout"}]},
            False,
            "scout_node",
        ),
        (
            "retrieval_node",
            {"reasoning_trace": [{"action": "agent_handoff", "sender": "retrieval", "recipient": "scout"}]},
            True,
            "scout_node",
        ),
        ("retrieval_node", {"reasoning_trace": []}, False, "reader_node"),
        (
            "reader_node",
            {"reasoning_trace": [{"action": "agent_handoff", "sender": "reader", "recipient": "retrieval"}]},
            False,
            "retrieval_node",
        ),
        (
            "summarizer_node",
            {"reasoning_trace": [{"action": "agent_handoff", "sender": "summarizer", "recipient": "summarizer"}]},
            False,
            "summarizer_node",
        ),
        (
            "fact_checker_node",
            {"reasoning_trace": [{"action": "agent_handoff", "sender": "fact_checker", "recipient": "scout"}]},
            False,
            "scout_node",
        ),
        ("fact_checker_node", {"fact_check_label": "support", "rejected_claims": []}, False, "writer_node"),
        ("fact_checker_node", {"fact_check_label": "contradict", "rejected_claims": [{"text": "bad"}]}, False, "escalation_node"),
        (
            "writer_node",
            {"reasoning_trace": [{"action": "agent_handoff", "sender": "writer", "recipient": "summarizer"}]},
            False,
            "summarizer_node",
        ),
        ("writer_node", {"reasoning_trace": []}, False, "__end__"),
    ]
    for node, state, cmp_flag, expected in samples:
        live_target = dispatch_next_node(node, state, comparison_plan=cmp_flag)
        resume_target = resolve_next_node_after_checkpoint(node, state, comparison_plan=cmp_flag)
        assert live_target == resume_target == expected

    # Verify Writer AgentDecision -> Summarizer works end-to-end inside compiled LangGraph
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    writer_calls = {"count": 0}
    summarizer_calls = {"count": 0}

    def fake_writer(state):
        writer_calls["count"] += 1
        if writer_calls["count"] == 1:
            return {
                "final_report": "Draft needing polish",
                "agent_decision": {
                    "action": "revise_draft",
                    "target_agent": "summarizer",
                    "reason": "Citation alignment failed in writer gate",
                    "focus": ["Align numeric claim with source span"],
                },
            }
        return {"final_report": "Verified final report: accuracy is 93.4%.", "agent_decision": None}

    def fake_summarizer(state):
        summarizer_calls["count"] += 1
        return {
            "answer": "Accuracy is 93.4%.",
            "claims": [{"id": "c1", "text": "Accuracy is 93.4%.", "evidence_ids": ["sp1"]}],
            "evidence_spans": [{"id": "sp1", "source_unit_id": "u1", "doc_id": "p.pdf", "page_no": 1, "quote": "Accuracy is 93.4%."}],
        }

    runners = {
        "retrieval": lambda s: {"retrieved_docs": [{"id": "u1", "doc_id": "p.pdf", "page_no": 1, "text": "Accuracy is 93.4%."}]},
        "reader": lambda s: {"source_units": s["retrieved_docs"]},
        "summarizer": fake_summarizer,
        "fact_checker": lambda s: {"fact_check_label": "support", "verified_claims": s["claims"], "rejected_claims": []},
        "writer": fake_writer,
    }
    monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_module.ScoutAgent, "run", lambda self, state: {})
    monkeypatch.setattr(graph_module, "add_figure_evidence", lambda inputs, update, mode: update)

    app = build_research_rag_graph("cpu_min", auto_scout=False)
    out = app.invoke({"query": "What is accuracy?", "pdf_paths": ["p.pdf"], "max_reflection_loops": 2})
    assert writer_calls["count"] == 2
    assert summarizer_calls["count"] == 2
    assert out["retry_count"] == 1
    assert "93.4%" in out["final_report"]


def test_m4_dag_manifest_sync_strictly_matches_task_id_never_completes_waiting_branches():
    """
    Reproduces and verifies the M4 P1 DAG sync fix:
    When 3 parallel investigation branches are PENDING and only 1 branch reports a result in `trace_event["tasks"]`,
    ONLY that matching `task_id` becomes COMPLETED; the other 2 waiting branches AND the `merge` node MUST stay PENDING!
    """
    plan = supervisor_planner.plan_query("Compare Alpha, Beta, and Gamma across accuracy, latency, and memory")
    plan.sub_queries = ["Alpha accuracy", "Beta latency", "Gamma memory"]
    manifest = build_research_dag_plan(plan, budget=ResearchDAGBudget(max_parallel_subqueries=3)).to_manifest()

    # Partial event: ONLY investigate_1 finished; investigate_2 and investigate_3 are still waiting
    partial_event = {
        "round_index": 0,
        "tasks": [
            {"id": "investigate_1", "query": "Alpha accuracy", "status": "COMPLETED", "attempts": 1, "hits": 3},
        ],
    }
    after_partial = sync_dag_manifest_from_retrieval(manifest, partial_event, retrieved_count=3, round_index=0)
    status_by_id = {n["id"]: n["status"] for n in after_partial["nodes"]}
    assert status_by_id["investigate_1"] == "COMPLETED"
    assert status_by_id["investigate_2"] == "PENDING"
    assert status_by_id["investigate_3"] == "PENDING"
    assert status_by_id["merge_evidence"] == "PENDING"
    assert after_partial["completed_tasks"] == 1

    # Second event: investigate_2 COMPLETED, investigate_3 FAILED -> all 3 deps terminal, so merge can COMPLETE
    second_event = {
        "round_index": 0,
        "tasks": [
            {"id": "investigate_2", "query": "Beta latency", "status": "COMPLETED", "attempts": 1, "hits": 2},
            {"id": "investigate_3", "query": "Gamma memory", "status": "FAILED", "attempts": 1, "hits": 0, "error": "TimeoutError"},
        ],
    }
    after_all = sync_dag_manifest_from_retrieval(after_partial, second_event, retrieved_count=5, round_index=0)
    status_after_all = {n["id"]: n["status"] for n in after_all["nodes"]}
    assert status_after_all["investigate_1"] == "COMPLETED"
    assert status_after_all["investigate_2"] == "COMPLETED"
    assert status_after_all["investigate_3"] == "FAILED"
    assert status_after_all["merge_evidence"] == "COMPLETED"
    assert after_all["completed_tasks"] == 3


def test_m4_multi_document_fanout_six_pdfs_preserves_full_lineage_and_accuracy(api_client, tmp_path):
    """
    Multi-document stress & accuracy test across 6 distinct bilingual PDFs:
    1. Per-document parallel fan-out (`execute_document_fanout_dag`) preserves full lineage
       (`source_path`, `doc_id`, `document_sha256`, `origin_task_id`, `provenance`) and 100% document coverage.
    2. End-to-end API `/api/runs` tested uniformly for BOTH UI entry (`parallel_retrieval=True`),
       default API entry (`parallel_retrieval=False`), AND 6-PDF `comparison_plan=True`.
    """
    corpus_specs = [
        ("doc_solar", "solar_efficiency_en.pdf", "Solar photovoltaic array S1 achieved 24.8% conversion efficiency over 120 test hours.", "24.8%"),
        ("doc_wind", "wind_turbine_en.pdf", "Offshore wind turbine W2 generated 14.6 MW peak output at 11.2 m/s wind speed.", "14.6 MW"),
        ("doc_hydro", "hydro_vietnam_vi.pdf", "Nhà máy thủy điện H3 tại Sơn La đạt hiệu suất tuabin 91.5% với lưu lượng 340 m3/s.", "91.5%"),
        ("doc_battery", "battery_storage_en.pdf", "Solid-state battery B4 retained 96.2% capacity after 1500 charge cycles.", "96.2%"),
        ("doc_geotherm", "geothermal_vi.pdf", "Hệ thống địa nhiệt G5 duy trì công suất ổn định 8.4 MW với nhiệt độ giếng 185 độ C.", "8.4 MW"),
        ("doc_hydrogen", "hydrogen_cell_en.pdf", "Proton-exchange hydrogen electrolyzer E6 operated at 73.9% thermal efficiency.", "73.9%"),
    ]
    doc_paths: list[str] = []
    sha_by_doc: dict[str, str] = {}
    for doc_id, fname, text, _metric in corpus_specs:
        pdf = _make_pdf(tmp_path / fname, [f"Energy Systems Benchmark Report: {fname}\n\n{text}"])
        doc_paths.append(str(pdf))
        sha_by_doc[doc_id] = hashlib.sha256(pdf.read_bytes()).hexdigest()
        db.save_document(doc_id, fname, str(pdf), pdf.stat().st_size)

    # 1. Verify execute_document_fanout_dag across all 6 PDFs preserves exact lineage for every document
    retriever = CPUMinRetrievalAgent()

    def _retrieve_one(path: str) -> dict[str, Any]:
        return retriever.run({
            "query": "Compare conversion efficiency, output, capacity, and performance across all six energy systems.",
            "pdf_path": path,
            "pdf_paths": [path],
            "top_k": 3,
        })

    fanout_out, fanout_trace = execute_document_fanout_dag(
        doc_paths,
        _retrieve_one,
        top_k=12,
        budget=ResearchDAGBudget(max_concurrency=3, task_timeout_seconds=30.0),
    )
    assert fanout_trace["action"] == "document_fanout_dag_executed"
    assert fanout_trace["status"] == "completed"
    assert fanout_trace["document_count"] == 6
    assert fanout_trace["completed_documents"] == 6
    merged_docs = fanout_out["retrieved_docs"]
    assert len(merged_docs) >= 6
    covered_paths = {c.get("source_path") for c in merged_docs}
    assert covered_paths == set(doc_paths)
    for chunk in merged_docs:
        assert chunk.get("origin_task_id", "").startswith("doc_investigate_")
        assert chunk.get("provenance")
        prov0 = chunk["provenance"][0]
        assert prov0.get("source_path") == chunk["source_path"]
        assert prov0.get("task_id") == chunk["origin_task_id"]
        assert prov0.get("document_sha256") == chunk.get("document_sha256")

    # 2. Verify uniform API behavior on 6 PDFs across UI parallel_retrieval=True, default API parallel_retrieval=False, and comparison_plan=True
    all_doc_ids = [spec[0] for spec in corpus_specs]
    for parallel_flag, comparison_flag in [(False, False), (True, False), (False, True)]:
        resp = api_client.post(
            "/api/runs",
            json={
                "question": "Compare efficiency and output reported across the six energy system studies.",
                "document_ids": all_doc_ids,
                "mode": "cpu_min",
                "parallel_retrieval": parallel_flag,
                "comparison_plan": comparison_flag,
            },
        )
        assert resp.status_code == 200
        run = _poll_run(api_client, resp.json()["id"], timeout=30.0)
        assert run["status"] == "completed"
        assert run["document_ids"] == all_doc_ids
        assert len(run["documents"]) == 6
        report = run["final_report"]
        cited_doc_ids = {ref["document_id"] for ref in run["citation_references"]}
        if comparison_flag:
            # Comparison plan enforces evidence & audited citations from ALL 6 documents!
            assert cited_doc_ids == set(all_doc_ids)
            for _did, _fn, _txt, expected_metric in corpus_specs:
                assert expected_metric in report, f"Missing {expected_metric} in 6-PDF comparison report"
            schedules = run["run_manifest"]["observed"]["comparison_schedules"]
            assert len(schedules) == 2
            assert schedules[0]["status"] == "COMPLETED"
            assert len(schedules[0]["required_sources"]) == 6
        else:
            # Non-comparison multi-doc summary covers all 6 documents cleanly
            assert len(cited_doc_ids) >= 5
            assert run["run_manifest"]["configuration"]["retrieval"]["parallel_enabled"] is parallel_flag


def test_m4_real_six_arxiv_papers_multi_document_accuracy(tmp_path):
    """
    Exercises multi-document fan-out retrieval & cross-paper QA accuracy on the 6 real arXiv PDFs
    (`colbert`, `contriever`, `dpr`, `fid`, `rag`, `realm`) when present in `main/artifacts/eval_pr05/pdfs_six/`,
    falling back seamlessly to 6 self-contained academic IR papers in clean-checkout environments.
    """
    real_dir = Path(__file__).resolve().parents[1] / "artifacts" / "eval_pr05" / "pdfs_six"
    real_files = sorted(real_dir.glob("*.pdf")) if real_dir.is_dir() else []

    if len(real_files) >= 6:
        six_paths = [str(p) for p in real_files[:6]]
        probes = [
            ("ColBERT late interaction MaxSim", "colbert"),
            ("Contriever unsupervised dense retrieval contrastive", "contriever"),
            ("Dense Passage Retrieval DPR dual-encoder BM25", "dpr"),
            ("Fusion-in-Decoder FiD generative open domain QA", "fid"),
            ("Retrieval-Augmented Generation RAG-Sequence RAG-Token", "rag"),
            ("REALM retrieval-augmented language model pre-training", "realm"),
        ]
    else:
        fallback_specs = [
            ("colbert-2004.12832v2.pdf", "ColBERT introduces late interaction using MaxSim over contextualized token embeddings for efficient retrieval."),
            ("contriever-2112.09118v2.pdf", "Contriever trains unsupervised dense information retrieval models via contrastive learning with inverse cloze tasks."),
            ("dpr-2004.04906v3.pdf", "Dense Passage Retrieval (DPR) uses a dual-encoder BERT architecture that outperforms BM25 on open-domain QA."),
            ("fid-2007.01282v1.pdf", "Fusion-in-Decoder (FiD) uses dense retrieval for open-domain QA by encoding retrieved passages independently and fusing them in the generative decoder."),
            ("rag-2005.11401v4.pdf", "Retrieval-Augmented Generation proposes RAG-Sequence and RAG-Token models combining parametric and non-parametric memory."),
            ("realm-2002.08909v1.pdf", "REALM augments masked language model pre-training with a latent neural knowledge retriever over Wikipedia."),
        ]
        six_paths = [str(_make_pdf(tmp_path / fn, [txt])) for fn, txt in fallback_specs]
        probes = [
            ("ColBERT late interaction MaxSim", "colbert"),
            ("Contriever unsupervised dense retrieval contrastive", "contriever"),
            ("Dense Passage Retrieval DPR dual-encoder BM25", "dpr"),
            ("Fusion-in-Decoder FiD generative open domain QA", "fid"),
            ("Retrieval-Augmented Generation RAG-Sequence RAG-Token", "rag"),
            ("REALM retrieval-augmented language model pre-training", "realm"),
        ]

    retriever = CPUMinRetrievalAgent()
    hits = 0
    for query, expected_stem in probes:
        res = retriever.run({"query": query, "pdf_paths": six_paths, "top_k": 5})
        top3_docs = [str(c.get("doc_id") or "").lower() for c in (res.get("retrieved_docs") or [])[:3]]
        if any(expected_stem in d for d in top3_docs):
            hits += 1
    assert hits == 6, f"Expected 6/6 accuracy across 6-paper corpus, got {hits}/6"

    # Also verify per-document fan-out over all 6 papers preserves every paper's provenance
    fanout, trace = execute_document_fanout_dag(
        six_paths,
        lambda p: retriever.run({"query": "dense retrieval pre-training open-domain QA", "pdf_path": p, "pdf_paths": [p], "top_k": 2}),
        top_k=12,
        budget=ResearchDAGBudget(max_concurrency=3, task_timeout_seconds=60.0),
    )
    assert trace["status"] == "completed"
    assert trace["completed_documents"] == 6
    assert {c.get("source_path") for c in fanout["retrieved_docs"]} == set(six_paths)


def test_m4_worker_interrupt_crash_and_truthful_resume_across_multiple_pdfs(api_client, monkeypatch, tmp_path):
    """
    Simulates a mid-run worker crash in `summarizer_node` across 4 PDFs with custom `agent_modes`,
    then calls `POST /api/runs/{run_id}/resume` WITHOUT mode override:
    - Exercises the fixed `RunConfig.with_overrides(...)` path.
    - Proves completed nodes (`orchestrator_router`, `scout_node`, `retrieval_node`, `reader_node`)
      are NOT re-executed on resume.
    - Verifies resumed run completes cleanly with multi-document citations.
    """
    pdfs = [
        _make_pdf(tmp_path / f"study_{i}.pdf", [f"Multi-center Study {i} achieved {80 + i * 3}.5% precision on cohort C{i}."])
        for i in range(1, 5)
    ]
    doc_ids = []
    for i, p in enumerate(pdfs, 1):
        did = f"doc_crash_{i}"
        db.save_document(did, p.name, str(p), p.stat().st_size)
        doc_ids.append(did)

    should_crash = {"flag": True}
    node_calls_phase2: list[str] = []
    orig_get_runners = graph_module.get_agent_runners

    def wrapped_runners(mode):
        runners = orig_get_runners(mode)
        orig_ret = runners["retrieval"]
        orig_read = runners["reader"]
        orig_sum = runners["summarizer"]

        def ret_spy(state):
            if not should_crash["flag"]:
                node_calls_phase2.append("retrieval")
            return orig_ret(state)

        def read_spy(state):
            if not should_crash["flag"]:
                node_calls_phase2.append("reader")
            return orig_read(state)

        def sum_crash_once(state):
            if should_crash["flag"]:
                raise RuntimeError("Simulated worker disconnect mid-run")
            node_calls_phase2.append("summarizer")
            return orig_sum(state)

        return {**runners, "retrieval": ret_spy, "reader": read_spy, "summarizer": sum_crash_once}

    monkeypatch.setattr(graph_module, "get_agent_runners", wrapped_runners)

    # Phase 1: Run with custom agent_modes -> crashes in summarizer_node after persisting checkpoints through reader_node
    queued = api_client.post(
        "/api/runs",
        json={
            "question": "Summarize the precision scores across all four multi-center studies.",
            "document_ids": doc_ids,
            "agent_modes": {
                "retrieval": "cpu_min",
                "reader": "cpu_min",
                "summarizer": "cpu_min",
                "fact_checker": "cpu_min",
                "writer": "cpu_min",
            },
            "max_iterations": 0,
        },
    )
    assert queued.status_code == 200
    src_id = queued.json()["id"]
    failed_run = _poll_run(api_client, src_id)
    assert failed_run["status"] == "failed"
    assert "Simulated worker disconnect" in (failed_run.get("error") or "")

    ckpts_resp = api_client.get(f"/api/runs/{src_id}/checkpoints")
    assert ckpts_resp.status_code == 200
    ckpts_json = ckpts_resp.json()
    assert ckpts_json["latest_node"] in {"reader_node", "supervisor_node"}
    assert "reader_node" in [c["node_name"] for c in ckpts_json["checkpoints"]]

    # Phase 2: Worker recovers -> call POST /api/runs/{id}/resume without explicit mode override
    should_crash["flag"] = False
    resumed_resp = api_client.post(f"/api/runs/{src_id}/resume")
    assert resumed_resp.status_code == 200
    resumed_id = resumed_resp.json()["id"]
    resumed_run = _poll_run(api_client, resumed_id)
    assert resumed_run["status"] == "completed"

    # Completed upstream nodes MUST NOT be re-executed on resume!
    assert "retrieval" not in node_calls_phase2
    assert "reader" not in node_calls_phase2
    assert "summarizer" in node_calls_phase2

    resumed_ckpts = db.list_run_checkpoints(resumed_id)
    resumed_nodes = [c["node_name"] for c in resumed_ckpts]
    assert "orchestrator_router" not in resumed_nodes
    assert "scout_node" not in resumed_nodes
    assert "retrieval_node" not in resumed_nodes
    assert "reader_node" not in resumed_nodes
    assert resumed_nodes[0] == "summarizer_node"
    assert resumed_run["final_report"]


def test_m4_timeout_budget_capped_retries_and_no_new_evidence_stop(api_client, monkeypatch, tmp_path):
    """
    Verifies:
    1. Timeout enforcement terminates cleanly with `TIMED_OUT` and zero leaked citations.
    2. Shared retry budget (default 2) + `no_new_evidence` growth gate prevents unbounded retry loops.
    """
    p1 = _make_pdf(tmp_path / "t1.pdf", ["Study T1 measured 12.4 ms latency."])
    p2 = _make_pdf(tmp_path / "t2.pdf", ["Study T2 measured 19.8 ms latency."])
    db.save_document("doc_t1", p1.name, str(p1), p1.stat().st_size)
    db.save_document("doc_t2", p2.name, str(p2), p2.stat().st_size)

    # 1. Timeout test via API
    class SlowGraph:
        async def astream(self, state, stream_mode):
            yield {"orchestrator_router": {"reasoning_trace": [{"action": "started"}]}}
            await asyncio.sleep(5.0)

    monkeypatch.setattr(service_module, "build_research_rag_graph", lambda **kwargs: SlowGraph())
    queued = api_client.post(
        "/api/runs",
        json={
            "question": "Compare T1 and T2 latency.",
            "document_ids": ["doc_t1", "doc_t2"],
            "mode": "cpu_min",
            "comparison_plan": True,
            "comparison_timeout_seconds": 0.15,
        },
    )
    assert queued.status_code == 200
    timed_out = _poll_run(api_client, queued.json()["id"])
    assert timed_out["status"] == "failed"
    assert timed_out["answer_status"] == "TIMED_OUT"
    assert timed_out["citation_references"] == []

    # 2. Retry budget & no_new_evidence growth gate (restoring real build_research_rag_graph)
    monkeypatch.undo()
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    retrieval_calls = {"count": 0}
    same_chunk = {"id": "u_static", "doc_id": "t1", "page_no": 1, "text": "Study T1 measured 12.4 ms latency."}

    def stagnant_retrieval(state):
        retrieval_calls["count"] += 1
        return {"retrieved_docs": [same_chunk]}

    def always_missing_checker(state):
        payload: dict[str, Any] = {
            "fact_check_label": "not_enough_info",
            "verified_claims": [],
            "rejected_claims": [],
            "evidence_gaps": [{"missing_aspect": f"round_{retrieval_calls['count']}", "suggested_query": f"energy consumption kWh {retrieval_calls['count']}"}],
        }
        if state.get("no_new_evidence"):
            # Even if an agent explicitly insists on another retrieval loop despite no_new_evidence,
            # Supervisor MUST reject it with `no_new_evidence_in_corpus` and stop cleanly!
            payload["agent_decision"] = {
                "action": "request_more_evidence",
                "target_agent": "retrieval",
                "reason": "Retry retrieval again despite stagnant corpus",
                "suggested_queries": ["energy consumption kWh forced retry"],
            }
        return payload

    runners = {
        "retrieval": stagnant_retrieval,
        "reader": lambda s: {"source_units": [same_chunk]},
        "summarizer": lambda s: {"answer": "Incomplete.", "claims": [], "evidence_spans": []},
        "fact_checker": always_missing_checker,
        "writer": lambda s: {"final_report": "Escalated low-confidence output."},
    }
    monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph_module.ScoutAgent, "run", lambda self, state: {})
    monkeypatch.setattr(graph_module, "add_figure_evidence", lambda inputs, update, mode: update)

    app = build_research_rag_graph("cpu_min", auto_scout=False)
    out = app.invoke({"query": "What was the energy consumption in kWh?", "pdf_paths": [str(p1)], "max_reflection_loops": 2})
    # Pass 0 runs retrieval (seen=[u_static]) -> checker requests retrieval (retry 1) ->
    # Pass 1 runs retrieval (returns same u_static -> graph sets no_new_evidence=True!) ->
    # checker proposes retrieval again -> Supervisor REJECTS (`no_new_evidence_in_corpus`) -> escalates!
    assert retrieval_calls["count"] >= 2
    assert out["retry_count"] == 1
    assert out["no_new_evidence"] is True
    assert out["confidence"] == "low"
    rejected_evals = [
        e for e in out["reasoning_trace"]
        if e.get("action") == "supervisor_decision_evaluated" and e.get("approved") is False
    ]
    assert any(e.get("rejection_reason") == "no_new_evidence_in_corpus" for e in rejected_evals)


def test_m4_two_concurrent_multi_doc_runs_isolated_even_with_identical_filenames(monkeypatch, research_db, tmp_path):
    """
    Runs TWO concurrent multi-document API requests simultaneously (3 PDFs each = 6 PDFs total),
    intentionally placing a DIFFERENT PDF with the EXACT SAME filename (`shared_study.pdf`) in both runs!
    Forces overlapping execution via a threading Barrier inside `CPUMinRetrievalAgent.run` and proves:
    - Zero document/cache/state leakage between Run A and Run B.
    - Each run cites ONLY its own documents, SHA-256 hashes, and numeric measurements.
    """
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "m4-concurrent")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module.settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api_module, "_legacy_mode_override", "cpu_min")

    # Run A: 3 PDFs (including dir_a/shared_study.pdf)
    pdf_a1 = _make_pdf(tmp_path / "dir_a" / "shared_study.pdf", ["AlphaCatalyst achieved 417.5 mg/L nitrate removal with 91.2% stability."])
    pdf_a2 = _make_pdf(tmp_path / "dir_a" / "alpha_thermal.pdf", ["AlphaCatalyst reactor operated at 312 Kelvin for 48 hours."])
    pdf_a3 = _make_pdf(tmp_path / "dir_a" / "alpha_cost.pdf", ["AlphaCatalyst synthesis cost was 18.5 USD per kilogram."])

    # Run B: 3 PDFs (including dir_b/shared_study.pdf — SAME filename, completely different content!)
    pdf_b1 = _make_pdf(tmp_path / "dir_b" / "shared_study.pdf", ["BetaPolymer achieved 823.9 mg/L sulfate capture with 76.4% yield."])
    pdf_b2 = _make_pdf(tmp_path / "dir_b" / "beta_thermal.pdf", ["BetaPolymer furnace operated at 695 Kelvin for 12 hours."])
    pdf_b3 = _make_pdf(tmp_path / "dir_b" / "beta_cost.pdf", ["BetaPolymer production cost was 64.2 USD per kilogram."])

    run_a_ids, run_b_ids = [], []
    run_a_shas, run_b_shas = set(), set()
    for idx, p in enumerate((pdf_a1, pdf_a2, pdf_a3), 1):
        did = f"doc_a_{idx}"
        db.save_document(did, p.name, str(p), p.stat().st_size)
        run_a_ids.append(did)
        run_a_shas.add(hashlib.sha256(p.read_bytes()).hexdigest())
    for idx, p in enumerate((pdf_b1, pdf_b2, pdf_b3), 1):
        did = f"doc_b_{idx}"
        db.save_document(did, p.name, str(p), p.stat().st_size)
        run_b_ids.append(did)
        run_b_shas.add(hashlib.sha256(p.read_bytes()).hexdigest())

    overlap_barrier = Barrier(2)
    orig_run = CPUMinRetrievalAgent.run
    first_hits = {"count": 0}

    def synchronized_run(self, state, *args, **kwargs):
        if state.get("retry_count", 0) == 0 and first_hits["count"] < 2:
            first_hits["count"] += 1
            try:
                overlap_barrier.wait(timeout=5.0)
            except Exception:
                pass
        return orig_run(self, state, *args, **kwargs)

    monkeypatch.setattr(CPUMinRetrievalAgent, "run", synchronized_run)

    async def _run_concurrent():
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="https://workspace.test") as client:
            login = await client.post("/api/auth/login", json={"password": "m4-concurrent"})
            assert login.status_code == 200
            req_a = client.post(
                "/api/research",
                json={
                    "query": "What nitrate removal in mg/L and stability did AlphaCatalyst achieve?",
                    "document_ids": run_a_ids,
                    "parallel_retrieval": True,
                },
            )
            req_b = client.post(
                "/api/research",
                json={
                    "query": "What sulfate capture in mg/L and yield did BetaPolymer achieve?",
                    "document_ids": run_b_ids,
                    "parallel_retrieval": True,
                },
            )
            resp_a, resp_b = await asyncio.gather(req_a, req_b)
            assert resp_a.status_code == 200 and resp_b.status_code == 200
            return resp_a.json(), resp_b.json()

    res_a, res_b = asyncio.run(_run_concurrent())

    # Strict isolation checks for Run A
    assert res_a["run_status"] == "completed"
    assert res_a["document_ids"] == run_a_ids
    assert {d["sha256"] for d in res_a["documents"]} == run_a_shas
    assert "417.5" in res_a["final_report"]
    assert "823.9" not in res_a["final_report"]
    assert "BetaPolymer" not in res_a["final_report"]
    assert set(ref["document_id"] for ref in res_a["citation_references"]).issubset(set(run_a_ids))

    # Strict isolation checks for Run B
    assert res_b["run_status"] == "completed"
    assert res_b["document_ids"] == run_b_ids
    assert {d["sha256"] for d in res_b["documents"]} == run_b_shas
    assert "823.9" in res_b["final_report"]
    assert "417.5" not in res_b["final_report"]
    assert "AlphaCatalyst" not in res_b["final_report"]
    assert set(ref["document_id"] for ref in res_b["citation_references"]).issubset(set(run_b_ids))
