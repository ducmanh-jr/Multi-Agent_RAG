import asyncio
import hashlib
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Event

import fitz
import httpx
import pytest
from fastapi.testclient import TestClient

from agents.agent0_scout import node as scout_module
from api import app as api_module, auth, research_runs as service_module
from api.run_manifest import get_run_manifest
from core import db, graph
from core.config import settings
from core.state import SourceUnit, compute_context_hash
from eval.run_benchmark import load_pilot
from modes.gpu_finetune.fact_checker import GPUFinetuneFactCheckerAgent
from modes.gpu_finetune.retrieval import GPUFinetuneRetrievalAgent
from modes.gpu_finetune.summarizer import GPUFinetuneSummarizerAgent
from modes.gpu_finetune.writer import GPUFinetuneWriterAgent
from utils.llm_client import llm_client


pytestmark = pytest.mark.integration


@pytest.fixture
def client(monkeypatch, research_db, tmp_path):
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "service-test")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})
    monkeypatch.setattr(api_module.settings, "UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setattr(api_module, "_legacy_mode_override", "cpu_min")
    with TestClient(api_module.app, base_url="https://workspace.test") as test_client:
        assert test_client.post("/api/auth/login", json={"password": "service-test"}).status_code == 200
        yield test_client


@pytest.fixture
def research_team(monkeypatch):
    observed = []

    def retrieve(state):
        observed.append(state["pdf_paths"])
        return {"retrieved_docs": []}

    monkeypatch.setattr(graph, "get_agent_runners", lambda mode: {
        "retrieval": retrieve,
        "reader": lambda state: {},
        "summarizer": lambda state: {},
        "fact_checker": lambda state: {"fact_check_label": "support"},
        "writer": lambda state: {"final_report": "Research complete."},
    })
    monkeypatch.setattr(graph, "add_figure_evidence", lambda inputs, update, mode: update)
    return observed


def _finish(client, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/api/runs/{run_id}")
        assert response.status_code == 200
        data = response.json()
        if data["status"] != "processing":
            return data
        time.sleep(0.01)
    pytest.fail(f"Run {run_id} did not reach a terminal status")


def test_run_reloads_all_sources_in_order_and_keeps_versions(client, research_team, research_db, tmp_path):
    paths = {}
    for identifier in ("first", "second"):
        path = tmp_path / f"{identifier}.md"
        path.write_text(f"Evidence from the {identifier} document.", encoding="utf-8")
        paths[identifier] = path
        db.save_document(identifier, path.name, str(path), path.stat().st_size, doc_type="md")
    queued = client.post("/api/runs", json={
        "question": "Compare both sources.",
        "document_ids": ["missing", "second", "first", "second"],
        "auto_scout": True,
        "mode": "cpu_min",
    }).json()
    result = _finish(client, queued["id"])
    assert result["status"] == "completed"
    assert result["document_id"] == "second"
    assert result["document_ids"] == ["second", "first"]
    assert [source["sha256"] for source in result["documents"]] == [
        hashlib.sha256(paths[identifier].read_bytes()).hexdigest() for identifier in ("second", "first")
    ]
    assert research_team == [[str(paths["second"].resolve()), str(paths["first"].resolve())]]
    manifest = result["run_manifest"]
    assert manifest["documents"] == result["documents"] == queued["run_manifest"]["documents"]
    assert manifest["configuration"]["auto_scout_requested"] is True
    assert manifest["configuration"]["auto_scout_enabled"] is False

    db.delete_document("second")
    assert client.get(f"/api/runs/{queued['id']}").json()["run_manifest"] == manifest
    session = client.get(f"/api/sessions/{queued['session_id']}").json()
    assert session["runs"][0]["document_ids"] == ["second", "first"]
    assert client.get("/api/runs").json()["runs"][0]["document_ids"] == ["second", "first"]
    assert client.get("/api/runs").json()["runs"][0]["run_manifest"] == manifest
    assert client.delete(f"/api/runs/{queued['id']}").status_code == 200
    with research_db.db_session() as conn:
        assert conn.execute("SELECT COUNT(*) FROM run_documents").fetchone()[0] == 0


def test_auto_scout_reports_waiting_without_opt_in(client, research_team, monkeypatch):
    def unexpected_scout(query):
        pytest.fail("No provider call should start without Auto-Scout opt-in")

    monkeypatch.setattr(scout_module, "analyze_intent", unexpected_scout)
    run = client.post("/api/runs", json={"query": "Research topic", "document_ids": ["invalid"]}).json()
    assert run["status"] == "completed" and run["answer_status"] == "WAITING_FOR_DOCUMENT"
    assert run["document_ids"] == []
    assert run["run_manifest"]["documents"] == []
    assert run["run_manifest"]["observed"]["retrieval_attempts"] == []
    assert client.get(f"/api/runs/{run['id']}").json()["answer_status"] == "WAITING_FOR_DOCUMENT"
    assert not research_team


@pytest.mark.parametrize("endpoint", ["/api/runs", "/api/research", "/api/research/stream"])
@pytest.mark.parametrize(("parallel_retrieval", "comparison_plan"), [(False, False), (True, False), (False, True)])
def test_cpu_min_answers_with_citations_from_both_pdfs(client, tmp_path, endpoint, parallel_retrieval, comparison_plan):
    questions = load_pilot(tmp_path)
    question = next(item for item in questions if item["id"] == "pilot-multi-en")
    for index, filename in enumerate(question["documents"]):
        pdf = tmp_path / filename
        db.save_document(f"doc-{index}", filename, str(pdf), pdf.stat().st_size)
    response = client.post(endpoint, json={
        "query": question["question"], "document_ids": ["doc-0", "doc-1"], "mode": "cpu_min",
        "parallel_retrieval": parallel_retrieval,
        "comparison_plan": comparison_plan,
    })
    assert response.status_code == 200
    immediate_audit = None
    immediate_manifest = None
    if endpoint == "/api/research/stream":
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        finished = events[-1]
        assert finished["event"] == "finished"
        run_id = finished["run_id"]
        immediate_audit = finished["citation_audit"]
        immediate_manifest = finished["run_manifest"]
        assert events[0]["run_manifest"]["status"] == "processing"
        if comparison_plan:
            assert all(event["data"]["partial_answer"] is None for event in events if event["event"] == "node_update")
    elif endpoint == "/api/research":
        direct = response.json()
        run_id = direct["run_id"]
        immediate_audit = direct["citation_audit"]
        immediate_manifest = direct["run_manifest"]
    else:
        run_id = response.json()["id"]
    result = _finish(client, run_id)
    assert result["status"] == "completed"
    assert "21 days" in result["final_report"] and "35 days" in result["final_report"]
    assert {item["document_id"] for item in result["citation_references"]} == {"doc-0", "doc-1"}
    audit = result["result"]["citation_audit"]
    manifest = result["run_manifest"]
    assert manifest["configuration"]["retrieval"]["parallel_enabled"] is parallel_retrieval
    assert manifest["configuration"]["comparison_plan_enabled"] is comparison_plan
    comparison_schedules = manifest["observed"]["comparison_schedules"]
    assert bool(comparison_schedules) is comparison_plan
    if comparison_plan:
        schedule, review = comparison_schedules
        assert manifest["configuration"]["comparison_timeout_seconds"] == 120
        assert 0 < schedule["remaining_seconds_at_start"] <= 120
        assert schedule["required_sources"] == ["orchard-trial", "greenhouse-trial"]
        assert [task["status"] for task in schedule["tasks"]] == ["COMPLETED"] * 5
        assert {
            task["id"]: task["depends_on_success"] for task in schedule["tasks"]
        } == {
            "retrieve-1": [], "retrieve-2": [], "read-1": ["retrieve-1"], "read-2": ["retrieve-2"],
            "collect_evidence": ["read-1", "read-2"],
        }
        assert sum(event["action"] == "cpu_min_reading_completed" for event in result["reasoning_trace"]) == 2
        assert review["stage"] == "review_and_audit" and review["status"] == "COMPLETED"
        assert [(task["id"], task["depends_on_success"], task["status"]) for task in review["tasks"]] == [
            ("collect_evidence", [], "COMPLETED"), ("summarize", ["collect_evidence"], "COMPLETED"),
            ("fact_check", ["summarize"], "COMPLETED"), ("audit_report", ["fact_check"], "COMPLETED"),
        ]
    schedules = manifest["observed"]["retrieval_schedules"]
    assert bool(schedules) is parallel_retrieval
    if parallel_retrieval:
        assert schedules[0]["stage"] == "primary"
        assert schedules[0]["max_concurrency"] == 2
        tasks = schedules[0]["tasks"]
        assert [(task["id"], task["status"]) for task in tasks] == [
            ("fts5", "COMPLETED"), ("bm25", "COMPLETED"), ("fuse", "COMPLETED"),
        ]
        assert tasks[-1]["depends_on_success"] == ["fts5", "bm25"]
    if immediate_manifest is not None:
        assert immediate_manifest == manifest
    if immediate_audit is not None:
        assert immediate_audit == audit
    provenance = audit["evidence_provenance"]
    spans = {span["id"]: span for span in result["evidence_spans"]}
    documents = {document["id"]: document for document in result["documents"]}
    assert set(provenance) == {
        eid for check in audit["claims"].values() for eid in check["evidence_ids"]
    }
    for ref in result["citation_references"]:
        evidence_id = ref["evidence_id"]
        entry = provenance[evidence_id]
        unit = entry["source_unit"]
        span = spans[evidence_id]
        document = documents[ref["document_id"]]
        assert entry["status"] == "linked"
        assert entry["document"] == {key: document[key] for key in ("id", "name", "sha256")}
        assert entry["document"]["sha256"] == hashlib.sha256((tmp_path / document["name"]).read_bytes()).hexdigest()
        assert unit["page_no"] == span["page_no"] == ref["page"]
        assert unit["id"] == span["source_unit_id"]
        assert unit["context_hash"] == compute_context_hash(unit["text"], unit["doc_id"], unit["page_no"])
        assert " ".join(span["quote"].split()) in " ".join(unit["text"].split())

    db.delete_document("doc-0")
    pdf = tmp_path / question["documents"][0]
    db.save_document("replacement", pdf.name, str(pdf), pdf.stat().st_size)
    refreshed = client.get(f"/api/runs/{run_id}").json()
    assert refreshed["result"]["citation_audit"] == audit
    assert {ref["document_id"] for ref in refreshed["citation_references"]} == {"doc-1"}
    exported = client.get(f"/api/runs/{run_id}/export?format=json").json()
    assert exported["citation_audit"] == audit
    assert exported["run_manifest"] == manifest
    session = client.get(f"/api/sessions/{result['session_id']}").json()
    assert session["runs"][0]["citation_audit"] == audit
    assert session["runs"][0]["run_manifest"] == manifest


@pytest.mark.parametrize("fault", ["missing_selection", "blank_pdf", "missing_citation", "source_changed"])
def test_comparison_requires_evidence_from_both_sources(client, tmp_path, monkeypatch, fault):
    from modes.cpu_min.writer import CPUMinWriterAgent

    question = next(item for item in load_pilot(tmp_path) if item["id"] == "pilot-multi-en")
    for index, filename in enumerate(question["documents"]):
        if index == 1 and fault == "missing_selection":
            continue
        pdf = tmp_path / filename
        if index == 1 and fault == "blank_pdf":
            with fitz.open() as document:
                document.new_page()
                document.save(pdf)
        db.save_document(f"doc-{index}", filename, str(pdf), pdf.stat().st_size)

    if fault == "missing_citation":
        write = CPUMinWriterAgent.run

        def write_one_claim(self, state):
            return write(self, {**state, "verified_claims": state["verified_claims"][:1]})

        monkeypatch.setattr(CPUMinWriterAgent, "run", write_one_claim)

    if fault == "source_changed":
        pin = service_module.pin_citation_provenance

        def change_before_pin(traces, state, sources):
            (tmp_path / question["documents"][1]).write_bytes(b"Different document version")
            pin(traces, state, sources)

        monkeypatch.setattr(service_module, "pin_citation_provenance", change_before_pin)

    queued = client.post("/api/runs", json={
        "query": question["question"], "document_ids": ["doc-0", "doc-1"],
        "mode": "cpu_min", "comparison_plan": True,
    }).json()
    result = _finish(client, queued["id"])
    assert result["status"] == "completed"
    assert result["answer_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["citation_references"] == result["verified_claims"] == []
    assert "21 days" not in result["final_report"] and "35 days" not in result["final_report"]
    trace = result["reasoning_trace"]
    refusal, = [event for event in trace if event["action"] == "comparison_evidence_incomplete"]
    assert refusal["reason"] == {
        "missing_selection": "selected_source_unavailable",
        "blank_pdf": "required_retrieval_incomplete",
        "missing_citation": "missing_audited_source",
        "source_changed": "source_snapshot_unconfirmed",
    }[fault]
    schedules = result["run_manifest"]["observed"]["comparison_schedules"]
    if fault == "missing_selection":
        assert schedules == []
    elif fault == "blank_pdf":
        assert schedules[0]["tasks"][-1]["status"] == "SKIPPED"
        assert schedules[0]["tasks"][-1]["blocked_by"] == ["read-2"]
        assert schedules[0]["tasks"][-2]["blocked_by"] == ["retrieve-2"]
        assert sum(event["action"] == "cpu_min_reading_completed" for event in trace) == 1
        assert not any(event.get("agent") == "CPUMinSummarizerAgent" for event in trace)
    else:
        assert schedules[0]["tasks"][-1]["status"] == "COMPLETED"


def _comparison_input(tmp_path, timeout_seconds):
    question = next(item for item in load_pilot(tmp_path) if item["id"] == "pilot-multi-en")
    for index, filename in enumerate(question["documents"]):
        path = tmp_path / filename
        db.save_document(f"doc-{index}", filename, str(path), path.stat().st_size)
    return {
        "query": question["question"], "document_ids": ["doc-0", "doc-1"],
        "mode": "cpu_min", "comparison_plan": True, "comparison_timeout_seconds": timeout_seconds,
    }


def test_comparison_persistence_requires_a_completed_review_schedule(client, tmp_path, monkeypatch):
    pin = service_module.pin_citation_provenance

    def remove_review_proof(traces, state, sources):
        pin(traces, state, sources)
        traces[:] = [event for event in traces if not (
            event.get("action") == "comparison_task_schedule" and event["details"]["stage"] == "review_and_audit"
        )]

    monkeypatch.setattr(service_module, "pin_citation_provenance", remove_review_proof)
    run_id = client.post("/api/runs", json=_comparison_input(tmp_path, 20)).json()["id"]
    result = _finish(client, run_id)
    assert result["answer_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["verified_claims"] == result["citation_references"] == []
    assert "21 days" not in result["final_report"] and "35 days" not in result["final_report"]
    refusal, = [event for event in result["reasoning_trace"] if event["action"] == "comparison_evidence_incomplete"]
    assert refusal["reason"] == "comparison_review_incomplete"


@pytest.mark.parametrize("endpoint", ["/api/runs", "/api/research", "/api/research/stream"])
def test_comparison_timeout_is_visible_through_all_apis(client, tmp_path, monkeypatch, endpoint):
    class StalledGraph:
        async def astream(self, state, stream_mode):
            yield {"retrieval_node": {"reasoning_trace": [{"action": "comparison_started"}]}}
            await asyncio.Event().wait()

    monkeypatch.setattr(service_module, "build_research_rag_graph", lambda **kwargs: StalledGraph())
    response = client.post(endpoint, json=_comparison_input(tmp_path, 0.2))
    if endpoint == "/api/runs":
        run_id = response.json()["id"]
    elif endpoint == "/api/research":
        assert response.status_code == 504 and response.json()["answer_status"] == "TIMED_OUT"
        run_id = response.json()["run_id"]
    else:
        events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
        assert events[-1]["event"] == "error" and events[-1]["answer_status"] == "TIMED_OUT"
        run_id = events[0]["run_id"]
    result = _finish(client, run_id)
    assert result["status"] == "failed" and result["answer_status"] == "TIMED_OUT"
    assert not result["final_report"] and result["verified_claims"] == result["citation_references"] == []
    assert result["run_manifest"]["configuration"]["comparison_timeout_seconds"] == 0.2


def test_comparison_handoff_uses_the_original_deadline(client, tmp_path, monkeypatch):
    from core.comparison_plan import get_comparison_budget
    from modes.cpu_min.retrieval import CPUMinRetrievalAgent
    from modes.cpu_min.summarizer import CPUMinSummarizerAgent

    retrieve = CPUMinRetrievalAgent.run_comparison
    entered, release, finished = Event(), Event(), Event()
    deadlines = []

    def slow_retry(self, state, plan, reader):
        budget = get_comparison_budget()
        deadlines.append(budget.deadline_at)
        if state["retry_count"]:
            entered.set()
            try:
                assert release.wait(5)
                return {
                    "final_report": "Late comparison",
                    "reasoning_trace": [{"action": "comparison_task_schedule", "details": {"status": "COMPLETED"}}],
                }
            finally:
                finished.set()
        return retrieve(self, state, plan, reader)

    def request_more_context(self, state):
        return {
            "critic_action": "request_more_context",
            "critic_feedback": {"missing_key_points": ["compare the duration of both trials"]},
        }

    monkeypatch.setattr(CPUMinRetrievalAgent, "run_comparison", slow_retry)
    monkeypatch.setattr(CPUMinSummarizerAgent, "run", request_more_context)
    run_id = client.post("/api/runs", json={**_comparison_input(tmp_path, 2), "max_iterations": 1}).json()["id"]
    try:
        assert entered.wait(3)
        assert len(deadlines) == 2 and deadlines[0] == deadlines[1]
        timed_out = _finish(client, run_id)
        assert not finished.is_set()
        assert timed_out["answer_status"] == "TIMED_OUT"
        assert not timed_out["final_report"]
        requests = [event for event in timed_out["reasoning_trace"] if event["action"] == "agent_handoff"]
        assert [event["recipient"] for event in requests] == ["retrieval"]
    finally:
        release.set()
    assert finished.wait(3)
    assert not client.get(f"/api/runs/{run_id}").json()["final_report"]


def test_cancelled_comparison_does_not_wait_for_or_publish_a_late_writer(client, tmp_path, monkeypatch):
    from modes.cpu_min.writer import CPUMinWriterAgent

    entered, release, finished = Event(), Event(), Event()

    def slow_writer(self, state):
        entered.set()
        try:
            assert release.wait(5)
            return {"final_report": "Late comparison"}
        finally:
            finished.set()

    monkeypatch.setattr(CPUMinWriterAgent, "run", slow_writer)
    run_id = client.post("/api/runs", json=_comparison_input(tmp_path, 20)).json()["id"]
    try:
        assert entered.wait(3)
        assert client.post(f"/api/runs/{run_id}/cancel").json()["status"] == "cancelled"
        deadline = time.monotonic() + 3
        while run_id in api_module.research_runs.tasks and time.monotonic() < deadline:
            time.sleep(0.01)
        assert run_id not in api_module.research_runs.tasks
        assert not finished.is_set()
    finally:
        release.set()
    assert finished.wait(3)
    result = client.get(f"/api/runs/{run_id}").json()
    assert result["status"] == "cancelled" and not result["final_report"]
    assert result["verified_claims"] == result["citation_references"] == []
    assert len(result["run_manifest"]["observed"]["comparison_schedules"]) == 1


@pytest.mark.parametrize("terminal", ["timed_out", "cancelled"])
def test_interrupted_review_cannot_advance_to_the_writer(client, tmp_path, monkeypatch, terminal):
    from modes.cpu_min.fact_checker import CPUMinFactCheckerAgent
    from modes.cpu_min.writer import CPUMinWriterAgent

    entered, release, finished, writer_called = Event(), Event(), Event(), Event()
    check = CPUMinFactCheckerAgent.run

    def slow_checker(self, state):
        entered.set()
        try:
            assert release.wait(8)
            return check(self, state)
        finally:
            finished.set()

    def writer(self, state):
        writer_called.set()
        return {"final_report": "Late comparison"}

    monkeypatch.setattr(CPUMinFactCheckerAgent, "run", slow_checker)
    monkeypatch.setattr(CPUMinWriterAgent, "run", writer)
    run_id = client.post("/api/runs", json=_comparison_input(tmp_path, 2 if terminal == "timed_out" else 20)).json()["id"]
    try:
        assert entered.wait(3)
        if terminal == "cancelled":
            assert client.post(f"/api/runs/{run_id}/cancel").json()["status"] == "cancelled"
        result = _finish(client, run_id)
        if terminal == "timed_out":
            assert result["status"] == "failed" and result["answer_status"] == "TIMED_OUT"
        else:
            assert result["status"] == "cancelled"
        assert not finished.is_set()
        assert not result["final_report"] and result["verified_claims"] == result["citation_references"] == []
    finally:
        release.set()
    assert finished.wait(3)
    assert not writer_called.wait(0.2)
    assert not client.get(f"/api/runs/{run_id}").json()["final_report"]


def test_sqlite_rejects_publication_after_deadline(research_db):
    db.save_run("expired-publication", "Compare")
    assert not db.update_run_result(
        "expired-publication", "completed", final_report="Late report", deadline_at=time.monotonic() - 1,
    )
    assert db.get_run("expired-publication")["status"] == "processing"
    assert db.cancel_run("expired-publication") == "cancelled"


@pytest.mark.parametrize(("endpoint", "scenario"), [
    ("/api/runs", "over"),
    ("/api/research", "over"),
    ("/api/research/stream", "over"),
    ("/api/research", "next_call"),
    ("/api/research", "exact"),
    ("/api/research", "missing"),
    ("/api/research", "unlimited_missing"),
])
def test_run_token_budget_accounts_for_actual_usage_and_blocks_publication(client, tmp_path, monkeypatch, endpoint, scenario):
    source = tmp_path / "sample.txt"
    source.write_text("Research evidence.", encoding="utf-8")
    db.save_document("sample", source.name, str(source), source.stat().st_size)
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "fixture-private-key")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "not-a-live-key")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "")
    actual_total = 14 if scenario == "over" else 10
    reports_usage = scenario not in {"missing", "unlimited_missing"}
    requested = []

    def inference(request):
        requested.append(request.url.host)
        assert request.url.host == "generativelanguage.googleapis.com"
        data = {"candidates": [{"content": {"parts": [{"text": "Model draft"}]}}]}
        if reports_usage:
            data["usageMetadata"] = {"promptTokenCount": 7, "candidatesTokenCount": 2, "totalTokenCount": actual_total}
        return httpx.Response(200, json=data)

    class GeneratingGraph:
        async def astream(self, state, stream_mode):
            yield {"summarizer_node": {"answer": "Provisional draft"}}
            try:
                await asyncio.to_thread(llm_client._call_gemini, "private prompt for fixture")
                if scenario == "next_call":
                    await asyncio.to_thread(llm_client._call_openai_compatible, "follow-up", None, 0.2, 16, False)
            except RuntimeError:
                # Even when an agent catches the budget signal, its fallback cannot call another provider.
                try:
                    await asyncio.to_thread(llm_client._call_openai_compatible, "fallback", None, 0.2, 16, False)
                except RuntimeError:
                    pass
            yield {"writer_node": {"final_report": "Model draft"}}

    monkeypatch.setattr(service_module, "build_research_rag_graph", lambda **kwargs: GeneratingGraph())
    expected_status = {
        "over": "TOKEN_BUDGET_EXHAUSTED", "next_call": "TOKEN_BUDGET_EXHAUSTED",
        "missing": "TOKEN_USAGE_UNAVAILABLE",
    }.get(scenario)
    with httpx.Client(transport=httpx.MockTransport(inference)) as provider:
        monkeypatch.setattr(llm_client, "http_client", provider)
        response = client.post(endpoint, json={
            "query": "Research a topic", "document_ids": ["sample"], "mode": "gemini",
            "token_budget": None if scenario == "unlimited_missing" else 10,
        })
        if endpoint == "/api/runs":
            run_id = response.json()["id"]
        elif endpoint == "/api/research/stream":
            events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]
            assert events[-1]["event"] == "error" and events[-1]["answer_status"] == expected_status
            assert all(item["data"]["partial_answer"] is None for item in events if item["event"] == "node_update")
            run_id = events[0]["run_id"]
        else:
            code = 422 if scenario in {"over", "next_call"} else (502 if scenario == "missing" else 200)
            assert response.status_code == code
            run_id = response.json()["run_id"]
        result = _finish(client, run_id)
    assert requested == ["generativelanguage.googleapis.com"]
    assert result["answer_status"] == expected_status
    assert result["status"] == ("failed" if expected_status else "completed")
    assert bool(result["final_report"]) is (expected_status is None)
    usage = result["run_manifest"]["observed"]["token_usage"]
    assert usage["total_tokens"] == (actual_total if reports_usage else 0)
    assert usage["reported_calls"] == int(reports_usage)
    assert usage["unreported_calls"] == int(not reports_usage)
    assert usage["complete"] is reports_usage
    assert usage["calls"][0]["operation"] == "generateContent"
    assert "fixture-private-key" not in str(usage) and "private prompt" not in str(usage)


def test_manifest_allowlists_settings_and_keeps_effective_limits(client, research_team, tmp_path, monkeypatch):
    sensitive = {
        "GEMINI_API_KEY": "gemini-private-key",
        "DEEPSEEK_API_KEY": "deepseek-private-key",
        "OPENAI_API_KEY": "openai-private-key",
        "COHERE_API_KEY": "cohere-private-key",
        "REMOTE_INFERENCE_URL": "https://private-user:private-pass@worker.invalid/?token=private-token",
        "LOCAL_CPU_URL": "http://private-ollama-host.invalid:11434",
        "CORPUS_PATH": str(tmp_path / "private-corpus.jsonl"),
    }
    for key, value in sensitive.items():
        monkeypatch.setattr(settings, key, value)
    monkeypatch.setattr(settings, "MAX_ORCHESTRATOR_RETRY", 1)
    source = tmp_path / "public-paper.txt"
    source.write_text("Calibration evidence.", encoding="utf-8")
    db.save_document("paper", source.name, str(source), source.stat().st_size)

    queued = client.post("/api/runs", json={
        "query": "What was measured?", "document_id": "paper", "mode": "cpu_min", "max_iterations": 5,
    })
    assert queued.status_code == 200
    run_id = queued.json()["id"]
    result = _finish(client, run_id)
    config = result["run_manifest"]["configuration"]
    assert config == queued.json()["run_manifest"]["configuration"]
    assert config["remote_inference_configured"] is True
    assert config["local_cpu_configured"] is True
    assert config["limits"]["requested_reflection_loops"] == 5
    assert config["limits"]["reflection_loops"] == 1
    assert set(config["agent_modes"].values()) == {"cpu_min"}
    assert all(event["details"]["action"] != "run_manifest" for event in result["events"])
    exported = client.get(f"/api/runs/{run_id}/export?format=json")
    assert exported.status_code == 200
    for marker in [*sensitive.values(), "private-user", "private-pass", "private-token", "private-corpus.jsonl"]:
        assert marker not in queued.text and marker not in exported.text


@pytest.mark.parametrize("embedding_unavailable", [False, True])
def test_manifest_records_real_retrieval_outcome(client, tmp_path, monkeypatch, embedding_unavailable):
    pdf = tmp_path / "calibration.pdf"
    with fitz.open() as document:
        for sentence in ("The calibration used two control samples.", "The calibration used a baseline temperature."):
            document.new_page().insert_text((72, 72), sentence)
        document.save(pdf)
    db.save_document("calibration", pdf.name, str(pdf), pdf.stat().st_size)
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://worker.invalid")
    monkeypatch.setattr(settings, "EMBEDDING_MODEL", "BAAI/bge-m3")
    monkeypatch.setattr(settings, "RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
    monkeypatch.setattr(settings, "GPU_FINETUNE_DENSE_ENABLED", True)
    monkeypatch.setattr(settings, "GPU_FINETUNE_RERANK_ENABLED", True)
    monkeypatch.setattr(settings, "GPU_FINETUNE_RERANK_KEEP_PREFIX", 0)
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "unused.jsonl"))
    monkeypatch.setattr(graph, "get_agent_runners", lambda mode: {
        "retrieval": GPUFinetuneRetrievalAgent(enable_remote_router=False).run,
        "reader": lambda state: {},
        "summarizer": lambda state: {},
        "fact_checker": lambda state: {"fact_check_label": "support"},
        "writer": lambda state: {"final_report": "Research complete."},
    })
    monkeypatch.setattr(graph, "add_figure_evidence", lambda inputs, update, mode: update)

    def inference(request):
        body = json.loads(request.content)
        if request.url.path == "/api/embed":
            if embedding_unavailable:
                return httpx.Response(503)
            return httpx.Response(200, json={
                "model": "BAAI/bge-m3", "revision": "a" * 40,
                "embeddings": [[1.0, *([0.0] * 1023)] for _ in body["texts"]],
            })
        assert request.url.path == "/api/rerank"
        assert len(body["candidates"]) == 2
        return httpx.Response(200, json={
            "model": "BAAI/bge-reranker-v2-m3", "revision": "b" * 40,
            "scores": [0, 1], "ranked_indices": [1, 0],
        })

    with httpx.Client(transport=httpx.MockTransport(inference)) as worker:
        monkeypatch.setattr(llm_client, "http_client", worker)
        queued = client.post("/api/runs", json={
            "query": "What calibration conditions were used?", "document_id": "calibration", "mode": "gpu_finetune",
            "parallel_retrieval": True,
        }).json()
        assert queued["run_manifest"]["observed"]["retrieval_attempts"] == []
        completed = _finish(client, queued["id"])
    assert completed["status"] == "completed"
    manifest = completed["run_manifest"]
    assert manifest["configuration"]["retrieval"]["dense_enabled"] is True
    assert manifest["configuration"]["retrieval"]["rerank_enabled"] is True
    assert manifest["configuration"]["retrieval"]["parallel_requested"] is True
    assert manifest["configuration"]["retrieval"]["parallel_enabled"] is False
    assert manifest["observed"]["retrieval_schedules"] == []
    assert manifest["observed"]["generator_revision"] is None
    assert manifest["observed"]["verifier_revision"] is None
    attempt, = manifest["observed"]["retrieval_attempts"]
    assert attempt["retrieved_count"] == 2
    if embedding_unavailable:
        assert attempt["dense"] == {"status": "fallback", "reason": "embedding_http_503"}
        assert attempt["rerank"] == {"status": "skipped", "reason": "dense_fallback"}
    else:
        assert attempt["dense"]["model"] == "BAAI/bge-m3"
        assert attempt["dense"]["revision"] == "a" * 40
        assert attempt["rerank"] == {
            "status": "used", "model": "BAAI/bge-reranker-v2-m3", "revision": "b" * 40,
            "changed": True, "kept_prefix": 0,
        }


@pytest.mark.parametrize("scenario", ["reported", "fallback", "legacy", "nli_unavailable", "mismatch", "adapter_changed"])
def test_manifest_records_the_worker_used_by_summarizer_and_fact_checker(client, tmp_path, monkeypatch, scenario):
    statement = "The pilot study lasted 7 days."
    path = tmp_path / "paper.txt"
    path.write_text(statement, encoding="utf-8")
    db.save_document("paper", path.name, str(path), path.stat().st_size)
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "https://model-observation.invalid")
    monkeypatch.setattr(graph, "get_agent_runners", lambda mode: {
        "retrieval": lambda state: {},
        "reader": lambda state: {"source_units": [SourceUnit("unit", statement, "paper").to_dict()]},
        "summarizer": GPUFinetuneSummarizerAgent().run,
        "fact_checker": GPUFinetuneFactCheckerAgent().run,
        "writer": GPUFinetuneWriterAgent().run,
    })

    generation = {
        "answer": statement,
        "claims": [{"text": statement, "unit_id": "unit"}],
        "evidence_spans": [{"quote": statement, "unit_id": "unit"}],
    }
    base_revision, adapter_digest, nli_digest = "a" * 40, "b" * 64, "c" * 64
    generation_count = 0
    visited = []

    def infer(request):
        nonlocal generation_count
        endpoint = request.url.path
        visited.append(endpoint)
        if endpoint == "/api/factcheck":
            body = json.loads(request.content)
            assert body["claims"] == [statement]
            if scenario == "nli_unavailable":
                return httpx.Response(200, headers={"X-Research-Model-Status": "unavailable"}, json={
                    "evaluations": [{"claim": statement, "verdict": "NOT_ENOUGH_INFO", "confidence": 0.0}],
                })
            headers = {} if scenario == "legacy" else {
                "X-Research-Model": "research-finetuned-nli", "X-Research-Revision": nli_digest,
            }
            return httpx.Response(200, headers=headers, json={
                "evaluations": [{"claim": statement, "verdict": "SUPPORT", "confidence": 0.98}],
            })
        if endpoint == "/api/summarize/critic":
            rewrite = scenario == "adapter_changed" and generation_count == 1
            return httpx.Response(200, json={
                "is_supported": not rewrite, "action": "rewrite" if rewrite else "accept",
                "issues_found": ["Review the wording"] if rewrite else [],
            })
        assert endpoint in {"/api/summarize/generator", "/generate"}
        if scenario == "fallback" and endpoint == "/api/summarize/generator":
            return httpx.Response(502)
        generation_count += 1
        headers = {
            "X-Research-Model": "Qwen/Qwen2.5-7B-Instruct", "X-Research-Revision": base_revision,
            "X-Research-Adapter": "critic" if scenario == "mismatch" else "generator",
            "X-Research-Adapter-SHA256": adapter_digest if generation_count == 1 else "d" * 64,
        }
        if scenario == "legacy":
            # A generated JSON field cannot supply trusted checkpoint identity.
            return httpx.Response(200, json={**generation, "model": "false-model", "revision": "f" * 40})
        body = generation if endpoint != "/generate" else {"text": json.dumps(generation)}
        return httpx.Response(200, json=body, headers=headers)

    with httpx.Client(transport=httpx.MockTransport(infer)) as worker:
        monkeypatch.setattr(llm_client, "http_client", worker)
        queued = client.post("/api/runs", json={
            "query": "How many days did the pilot study last?", "document_id": "paper",
            "mode": "gpu_finetune", "max_iterations": 0,
        }).json()
        completed = _finish(client, queued["id"])
    assert completed["status"] == "completed"
    manifest = completed["run_manifest"]
    observed = manifest["observed"]
    assert observed["generator_revision"] == (None if scenario in {"legacy", "mismatch", "adapter_changed"} else base_revision)
    assert observed["verifier_revision"] == (None if scenario in {"legacy", "nli_unavailable"} else nli_digest)
    assert ("/generate" in visited) == (scenario == "fallback")
    if scenario == "nli_unavailable":
        assert completed["verified_claims"] == []
    else:
        assert any(claim["status"] == "SUPPORT" for claim in completed["verified_claims"])
    generator = [event for event in observed["worker_inference"] if event["component"] == "generator"]
    verifier, = [event for event in observed["worker_inference"] if event["component"] == "verifier"]
    if scenario == "fallback":
        assert [event["status"] for event in generator] == ["http_error", "reported"]
        assert generator[0]["http_status"] == 502
        assert generator[1]["endpoint"] == "/generate"
    if scenario in {"legacy", "mismatch"}:
        assert generator == [{"component": "generator", "status": "unreported", "endpoint": "/api/summarize/generator"}]
    else:
        assert generator[-1]["adapter"] == "generator"
        assert [event["adapter_sha256"] for event in generator if event["status"] == "reported"] == (
            [adapter_digest, "d" * 64] if scenario == "adapter_changed" else [adapter_digest]
        )
    if scenario == "nli_unavailable":
        assert verifier == {"component": "verifier", "status": "unavailable", "endpoint": "/api/factcheck"}
    exported = client.get(f"/api/runs/{queued['id']}/export?format=json").json()
    assert exported["run_manifest"] == manifest


@pytest.mark.parametrize(("decision", "has_candidates", "can_download", "expected"), [
    ("SEARCH", True, True, None),
    ("CLARIFY", False, False, "CLARIFICATION_NEEDED"),
    ("REJECT", False, False, "OUT_OF_SCOPE"),
    ("SEARCH", False, False, "NO_RELEVANT_PAPERS"),
    ("SEARCH", True, False, "DOWNLOAD_FAILED"),
])
def test_scout_shares_the_run_and_stops_without_readable_evidence(
    client, research_team, monkeypatch, tmp_path, decision, has_candidates, can_download, expected,
):
    monkeypatch.setattr(scout_module, "analyze_intent", lambda _: {
        "decision": decision, "perspectives": [{"angle": "Primary", "query": "test topic"}],
        "academic_query": "test topic", "message": "Please specify the research question.",
    })
    monkeypatch.setattr(scout_module, "search_arxiv_papers", lambda *args, **kwargs: [{}] if has_candidates else [])
    monkeypatch.setattr(scout_module, "rank_and_select_papers", lambda *args, **kwargs: [{"title": "Found paper"}])
    monkeypatch.setattr(scout_module, "search_gemini_academic_papers", lambda *args, **kwargs: [])
    pdf = tmp_path / "scouted.pdf"

    def download(paper):
        if not can_download:
            raise OSError("Offline test fixture")
        with fitz.open() as document:
            document.new_page().insert_text((72, 72), "Published research evidence.")
            document.save(pdf)
        db.save_document("scouted", "Found paper", str(pdf), pdf.stat().st_size)
        return {**paper, "doc_id": "scouted", "local_pdf_path": str(pdf)}

    monkeypatch.setattr(scout_module, "download_and_register_paper", download)
    queued = client.post("/api/runs", json={"query": "Find evidence", "mode": "cpu_min", "auto_scout": True}).json()
    result = _finish(client, queued["id"])
    assert result["status"] == "completed"
    assert result["answer_status"] == expected
    manifest = result["run_manifest"]
    assert queued["run_manifest"]["documents"] == []
    assert manifest["configuration"]["auto_scout_enabled"] is True
    assert manifest["documents"] == result["documents"]
    assert len(client.get("/api/runs").json()["runs"]) == 1
    if expected is None:
        assert result["document_ids"] == ["scouted"]
        assert result["document_id"] == "scouted"
        assert research_team == [[str(pdf.resolve())]]
        assert result["final_report"] == "Research complete."
        assert client.get(f"/api/sessions/{result['session_id']}").json()["session"]["document_id"] == "scouted"
    else:
        assert result["final_report"]
        assert result["document_ids"] == []
        assert not research_team


def test_legacy_and_sse_return_ids_of_persisted_runs(client, research_team, tmp_path):
    text_file = tmp_path / "local.txt"
    text_file.write_text("Local source text.", encoding="utf-8")
    synchronous = client.post("/api/research", json={"query": "Read this", "pdf_path": str(text_file)}).json()
    assert synchronous["status"] == "support"
    assert synchronous["run_status"] == "completed"
    assert synchronous["document_ids"][0].startswith("local_")
    recorded = client.get(f"/api/runs/{synchronous['run_id']}").json()
    assert recorded["document_ids"] == synchronous["document_ids"]

    streamed = client.post("/api/research/stream", json={"query": "Read this too", "pdf_path": str(text_file)})
    assert streamed.status_code == 200
    events = [json.loads(line[6:]) for line in streamed.text.splitlines() if line.startswith("data: ")]
    assert events[0]["event"] == "started" and events[-1]["event"] == "finished"
    assert all(event["run_id"] == events[0]["run_id"] for event in events)
    assert any(event["node"] == "scout_node" for event in events if event["event"] == "node_update")
    assert events[-1]["document_ids"] == synchronous["document_ids"]
    assert client.get(f"/api/runs/{events[0]['run_id']}").json()["final_report"] == events[-1]["final_report"]


def test_legacy_failure_preserves_http_error_and_run_record(client, monkeypatch):
    def unavailable(**kwargs):
        raise RuntimeError("Test graph failed")

    monkeypatch.setattr(service_module, "build_research_rag_graph", unavailable)
    response = client.post("/api/research", json={"query": "Question"})
    assert response.status_code == 500
    run = client.get(f"/api/runs/{response.json()['run_id']}").json()
    assert run["status"] == "failed"
    assert run["error"] == "Test graph failed"
    assert run["run_manifest"] == response.json()["run_manifest"]
    assert run["run_manifest"]["status"] == "failed"


def test_cancel_during_a_node_keeps_progress_and_blocks_late_result(monkeypatch, research_db):
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "service-test")
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "_sessions", {})

    async def case():
        entered, release = asyncio.Event(), asyncio.Event()

        class PausedGraph:
            async def astream(self, state, stream_mode):
                yield {"scout_node": {"reasoning_trace": [{"action": "search_started"}]}}
                entered.set()
                await release.wait()
                yield {"writer_node": {"final_report": "Late answer", "reasoning_trace": [{"action": "late_writer"}]}}
                pytest.fail("Another step was requested after the run was cancelled")

        monkeypatch.setattr(service_module, "build_research_rag_graph", lambda **kwargs: PausedGraph())
        transport = httpx.ASGITransport(app=api_module.app)
        async with httpx.AsyncClient(transport=transport, base_url="https://workspace.test") as client:
            await client.post("/api/auth/login", json={"password": "service-test"})
            response = await client.post("/api/runs", json={"query": "Run", "auto_scout": True})
            run_id = response.json()["id"]
            task = api_module.research_runs.tasks[run_id]
            try:
                await asyncio.wait_for(entered.wait(), 5)
                cancelled = await client.post(f"/api/runs/{run_id}/cancel")
                assert cancelled.json()["status"] == "cancelled"
            finally:
                release.set()
            result = await asyncio.wait_for(task, 5)
            assert result.record["status"] == "cancelled"
            assert result.record["final_report"] == ""
            assert [event["action"] for event in result.record["reasoning_trace"]] == ["run_manifest", "search_started"]
            manifest = (await client.get(f"/api/runs/{run_id}")).json()["run_manifest"]
            assert manifest["status"] == "cancelled"
            assert manifest["configuration"] == response.json()["run_manifest"]["configuration"]
            assert (await client.post("/api/runs/missing/cancel")).status_code == 404

    asyncio.run(case())


def test_sqlite_first_terminal_transition_wins(research_db):
    for index in range(6):
        run_id = f"race-{index}"
        db.save_run(run_id, "Test")
        barrier = Barrier(2)

        def complete():
            barrier.wait(timeout=5)
            return db.update_run_result(run_id, "completed", final_report="Saved answer")

        def cancel():
            barrier.wait(timeout=5)
            return db.cancel_run(run_id)

        with ThreadPoolExecutor(max_workers=2) as pool:
            completed, cancelled = pool.submit(complete), pool.submit(cancel)
            expected = "completed" if completed.result() else "cancelled"
            assert cancelled.result() == expected
        assert db.cancel_run(run_id) == expected
        assert db.get_run(run_id)["final_report"] == ("Saved answer" if expected == "completed" else None)


def test_existing_database_keeps_legacy_primary_id_without_inventing_hash(monkeypatch, tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE research_runs (
                id TEXT PRIMARY KEY, question TEXT NOT NULL, status TEXT, created_at TEXT, final_report TEXT,
                verified_claims TEXT, rejected_claims TEXT, evidence_spans TEXT, reasoning_trace TEXT, error TEXT,
                document_id TEXT, session_id TEXT
            )
        """)
        connection.execute("""
            INSERT INTO research_runs (id, question, status, created_at, final_report, document_id)
            VALUES ('old', 'Old question', 'completed', '2026-01-01', 'Old report', 'known-source')
        """)
    monkeypatch.setattr(db, "DB_PATH", str(path))
    monkeypatch.setattr(db, "_db_initialized", False)
    db.init_db()
    stored = db.get_run("old")
    assert stored["final_report"] == "Old report"
    assert stored["document_ids"] == ["known-source"]
    assert stored["documents"] == []
    assert get_run_manifest(stored) is None
