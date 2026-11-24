from pathlib import Path
from threading import Barrier

import pytest

from core.config import RunConfig
from core.graph import build_research_rag_graph
from core.state import compute_context_hash
from eval.run_benchmark import load_pilot
from modes.cpu_min.fact_checker import CPUMinFactCheckerAgent
from modes.cpu_min.reader import CPUMinReaderAgent
from modes.cpu_min.summarizer import CPUMinSummarizerAgent
from modes.cpu_min.writer import CPUMinWriterAgent


pytestmark = pytest.mark.integration


def _compare(tmp_path):
    question = next(item for item in load_pilot(tmp_path) if item["id"] == "pilot-multi-en")
    paths = [str(tmp_path / name) for name in question["documents"]]
    config = RunConfig.capture(corpus_path=str(tmp_path / "missing.jsonl"))
    return build_research_rag_graph("cpu_min", run_config=config, comparison_plan=True).invoke({
        "query": question["question"], "pdf_paths": paths, "pdf_path": paths[0], "max_reflection_loops": 1,
    })


def _schedules(result):
    return [
        entry["details"] for entry in result["reasoning_trace"] if entry["action"] == "comparison_task_schedule"
    ]


@pytest.mark.parametrize(
    ("fault", "reason", "statuses"),
    [
        ("empty_draft", "synthesis_incomplete", ("FAILED", "SKIPPED", "SKIPPED")),
        ("duplicate_claim_id", "synthesis_incomplete", ("FAILED", "SKIPPED", "SKIPPED")),
        ("unchecked_claim", "fact_check_incomplete", ("COMPLETED", "FAILED", "SKIPPED")),
        ("changed_claim", "fact_check_incomplete", ("COMPLETED", "FAILED", "SKIPPED")),
        ("writer_error", "audit_report_failed", ("COMPLETED", "COMPLETED", "FAILED")),
    ],
)
def test_comparison_review_blocks_dependents_when_a_stage_fails(tmp_path, monkeypatch, fault, reason, statuses):
    summarize = CPUMinSummarizerAgent.run
    check = CPUMinFactCheckerAgent.run
    calls = []

    def synthesize(self, state):
        calls.append("summarize")
        update = summarize(self, state)
        if fault == "empty_draft":
            update["answer"] = ""
        elif fault == "duplicate_claim_id":
            first, second, *_ = update["claims"]
            second["claim_id"] = first.get("claim_id") or first["id"]
        return update

    def verify(self, state):
        calls.append("fact_check")
        update = check(self, state)
        assert update["fact_check_label"] == "support"
        if fault == "unchecked_claim":
            update["verified_claims"].pop()
        elif fault == "changed_claim":
            update["verified_claims"][0]["claim_text"] = "The trial lasted 999 days."
        return update

    def write(self, state):
        calls.append("audit_report")
        raise RuntimeError("A draft must never be published after a failed task")

    monkeypatch.setattr(CPUMinSummarizerAgent, "run", synthesize)
    monkeypatch.setattr(CPUMinFactCheckerAgent, "run", verify)
    monkeypatch.setattr(CPUMinWriterAgent, "run", write)
    result = _compare(tmp_path)
    review = _schedules(result)[-1]
    assert review["stage"] == "review_and_audit"
    assert [(task["id"], task["status"]) for task in review["tasks"][1:]] == list(zip(
        ("summarize", "fact_check", "audit_report"), statuses, strict=True,
    ))
    for previous, task in zip(review["tasks"], review["tasks"][1:]):
        assert task["depends_on_success"] == [previous["id"]]
        if task["status"] == "SKIPPED":
            assert task["blocked_by"] == [previous["id"]] and task["attempts"] == 0
    assert calls == [task["id"] for task in review["tasks"][1:] if task["attempts"]]
    assert result["answer_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["verified_claims"] == result["evidence_spans"] == []
    assert not any(entry["action"] == "citation_audit" for entry in result["reasoning_trace"])
    assert result["reasoning_trace"][-1]["reason"] == reason


def test_verifier_handoff_restarts_review_with_a_fresh_draft(tmp_path, monkeypatch):
    summarize = CPUMinSummarizerAgent.run
    verify = CPUMinFactCheckerAgent.run
    write = CPUMinWriterAgent.run
    calls = []

    def summarize_again(self, state):
        calls.append("summarize")
        return summarize(self, state)

    def check_again(self, state):
        calls.append("check")
        if not state.get("retry_count"):
            return {"fact_check_label": "contradict", "rejected_claims": [
                {"status": "CONTRADICT", "claim_text": "Compare the duration of both trials"},
            ]}
        assert not state.get("rejected_claims")
        return verify(self, state)

    def audited_writer(self, state):
        calls.append("write")
        return write(self, state)

    monkeypatch.setattr(CPUMinSummarizerAgent, "run", summarize_again)
    monkeypatch.setattr(CPUMinFactCheckerAgent, "run", check_again)
    monkeypatch.setattr(CPUMinWriterAgent, "run", audited_writer)
    result = _compare(tmp_path)
    evidence, first_review, second_review = _schedules(result)
    assert [task["status"] for task in first_review["tasks"]] == ["COMPLETED", "COMPLETED", "FAILED", "SKIPPED"]
    assert second_review["status"] == "COMPLETED" and second_review["retry_count"] == 1
    assert second_review["required_sources"] == evidence["required_sources"]
    assert calls == ["summarize", "check", "summarize", "check", "write"]
    assert "21 days" in result["final_report"] and "35 days" in result["final_report"]


def test_comparison_readers_run_concurrently_with_single_document_inputs(tmp_path, monkeypatch):
    read = CPUMinReaderAgent.run
    barrier = Barrier(2)
    called = []

    def scoped_reader(self, state, *, allow_pdf_fallback=True):
        path, = state["pdf_paths"]
        source = Path(path).stem
        assert state["pdf_path"] == path and not allow_pdf_fallback
        assert {chunk["doc_id"] for chunk in state["retrieved_docs"]} == {source}
        called.append(source)
        barrier.wait(timeout=5)
        return read(self, state, allow_pdf_fallback=allow_pdf_fallback)

    monkeypatch.setattr(CPUMinReaderAgent, "run", scoped_reader)
    result = _compare(tmp_path)
    assert sorted(called) == ["greenhouse-trial", "orchard-trial"]
    assert all(unit["source_path"] in result["pdf_paths"] for unit in result["source_units"])
    assert _schedules(result)[0]["status"] == "COMPLETED"
    assert "21 days" in result["final_report"] and "35 days" in result["final_report"]


@pytest.mark.parametrize("fault", ["empty", "foreign_source", "changed_page", "changed_text", "invalid_hash"])
def test_comparison_stops_before_synthesis_when_one_reader_has_no_traceable_evidence(
    tmp_path, monkeypatch, fault,
):
    read = CPUMinReaderAgent.run

    def corrupted_reader(self, state, *, allow_pdf_fallback=True):
        update = read(self, state, allow_pdf_fallback=allow_pdf_fallback)
        if Path(state["pdf_path"]).stem != "greenhouse-trial":
            return update
        if fault == "empty":
            return {**update, "source_units": []}
        changed = {
            "foreign_source": {"doc_id": "orchard-trial"},
            "changed_page": {"page_no": 99},
            "changed_text": {"text": "The greenhouse trial lasted 999 days."},
            "invalid_hash": {},
        }[fault]
        unit = {**update["source_units"][0], **changed}
        unit["context_hash"] = (
            "tampered" if fault == "invalid_hash" else compute_context_hash(unit["text"], unit["doc_id"], unit["page_no"])
        )
        return {**update, "source_units": [unit]}

    def unexpected_synthesis(self, state):
        pytest.fail("The Summarizer received a comparison with an invalid Reader branch")

    monkeypatch.setattr(CPUMinReaderAgent, "run", corrupted_reader)
    monkeypatch.setattr(CPUMinSummarizerAgent, "run", unexpected_synthesis)
    result = _compare(tmp_path)
    assert result["answer_status"] == "INSUFFICIENT_EVIDENCE"
    assert result["source_units"] == result["verified_claims"] == []
    assert "999" not in result["final_report"]
    schedule, = _schedules(result)
    statuses = {task["id"]: task["status"] for task in schedule["tasks"]}
    assert statuses == {
        "retrieve-1": "COMPLETED", "retrieve-2": "COMPLETED", "read-1": "COMPLETED",
        "read-2": "FAILED", "collect_evidence": "SKIPPED",
    }
    assert schedule["tasks"][-1]["blocked_by"] == ["read-2"]


@pytest.mark.parametrize("fail_reread", [False, True])
def test_reader_handoff_uses_scoped_rankings_without_retrieving_again(tmp_path, monkeypatch, fail_reread):
    read = CPUMinReaderAgent.run
    summarize = CPUMinSummarizerAgent.run
    calls = {}
    summaries = []

    def partial_reader(self, state, *, allow_pdf_fallback=True):
        source = Path(state["pdf_path"]).stem
        assert state["pdf_paths"] == [state["pdf_path"]] and not allow_pdf_fallback
        calls[source] = calls.get(source, 0) + 1
        update = read(self, state, allow_pdf_fallback=allow_pdf_fallback)
        if source == "orchard-trial":
            if calls[source] == 1:
                assert len(update["source_units"]) > 1
                update["source_units"] = update["source_units"][:-1]
            elif fail_reread:
                update["source_units"] = []
        return update

    def request_reread(self, state):
        summaries.append(state.get("retry_count", 0))
        if not state.get("retry_count"):
            return {"critic_action": "request_more_context", "critic_feedback": {"missing_key_points": ["orchard trial"]}}
        return summarize(self, state)

    monkeypatch.setattr(CPUMinReaderAgent, "run", partial_reader)
    monkeypatch.setattr(CPUMinSummarizerAgent, "run", request_reread)
    result = _compare(tmp_path)
    assert calls == {"orchard-trial": 2, "greenhouse-trial": 2}
    schedules = _schedules(result)
    first, reread = [schedule for schedule in schedules if schedule["stage"] != "review_and_audit"]
    assert (first["stage"], first["retry_count"]) == ("retrieve_and_read", 0)
    assert (reread["stage"], reread["retry_count"]) == ("read_handoff", 1)
    assert [task["id"] for task in reread["tasks"]] == ["read-1", "read-2", "collect_evidence"]
    request, = [event for event in result["reasoning_trace"] if event["action"] == "agent_handoff"]
    assert request["recipient"] == "reader"
    reviews = [schedule for schedule in schedules if schedule["stage"] == "review_and_audit"]
    assert [(item["retry_count"], item["status"]) for item in reviews] == (
        [(0, "SKIPPED")] if fail_reread else [(0, "SKIPPED"), (1, "COMPLETED")]
    )
    if fail_reread:
        assert summaries == [0]
        assert result["answer_status"] == "INSUFFICIENT_EVIDENCE" and result["source_units"] == []
        assert reread["tasks"][-1]["blocked_by"] == ["read-1"]
    else:
        assert summaries == [0, 1]
        assert "21 days" in result["final_report"] and "35 days" in result["final_report"]
