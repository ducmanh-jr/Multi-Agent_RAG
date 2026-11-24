"""Exercise the compiled graph with controlled evidence and reviewer decisions."""

import importlib

import pytest

from core.config import settings

pytestmark = pytest.mark.integration


@pytest.fixture
def graph_module(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "BASE_DIR", str(tmp_path))
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "empty.jsonl"))
    monkeypatch.setattr(settings, "MAX_ORCHESTRATOR_RETRY", 3)
    return importlib.import_module("core.graph")


def run_team(graph, monkeypatch, *, checker, summarizer=None, reader=None, retrieval=None, budget=3):
    calls = []

    def observe(role, callback):
        def run(state):
            calls.append(role)
            return callback(state)
        return run

    runners = {
        "retrieval": observe("retrieval", retrieval or (lambda s: {})),
        "reader": observe("reader", reader or (lambda s: {"source_units": [{"id": "one", "text": "Yield was 14."}]})),
        "summarizer": observe("summarizer", summarizer or (lambda s: {"answer": "Yield was 14."})),
        "fact_checker": observe("fact_checker", checker),
        "writer": observe("writer", lambda s: {"final_report": s["answer"]}),
    }
    monkeypatch.setattr(graph, "get_agent_runners", lambda mode: runners)
    monkeypatch.setattr(graph.ScoutAgent, "run", lambda self, state: {})
    app = graph.build_research_rag_graph("cpu_min")
    state = app.invoke({
        "query": "What was the yield?", "pdf_paths": ["trial.pdf"], "max_reflection_loops": budget,
    })
    return calls, state


def test_checker_delegates_revision_and_verifies_new_draft(graph_module, monkeypatch):
    def summarize(state):
        if not state.get("retry_count"):
            return {"answer": "Yield was 40.", "evidence_spans": [{"evidence_id": "old"}]}
        feedback = state["critic_feedback"]["revision_note"]
        assert "Yield was 40." in feedback
        assert state["query"] == "What was the yield?" and state["pdf_paths"] == ["trial.pdf"]
        return {"answer": state["source_units"][0]["text"]}

    def check(state):
        if state["answer"] == "Yield was 40.":
            return {"fact_check_label": "contradict", "rejected_claims": [{"claim_text": state["answer"]}]}
        assert state["rejected_claims"] == [] and state["evidence_spans"] == []
        return {"fact_check_label": "support", "verified_claims": [{"claim_text": state["answer"]}]}

    calls, result = run_team(graph_module, monkeypatch, checker=check, summarizer=summarize)
    assert calls == ["retrieval", "reader", "summarizer", "fact_checker", "summarizer", "fact_checker", "writer"]
    assert result["final_report"] == "Yield was 14."
    assert result["retry_count"] == 1 and result["confidence"] == "high"
    request, receipt = [e for e in result["reasoning_trace"] if e.get("action", "").startswith("agent_handoff")]
    assert request["sender"] == "fact_checker" and request["recipient"] == "summarizer"
    assert receipt["action"] == "agent_handoff_received" and receipt["signature"] == request["signature"]


def test_checker_delegates_unread_chunk_before_search(graph_module, monkeypatch):
    docs = [{"id": "one", "text": "Method."}, {"id": "two", "text": "Observed yield."}]

    def read(state):
        return {"source_units": state["retrieved_docs"][:1]}

    def check(state):
        if state["source_units"][0]["id"] == "two":
            return {"fact_check_label": "support"}
        return {"fact_check_label": "not_enough_info", "evidence_gaps": [{"suggested_query": "observed yield"}]}

    calls, result = run_team(
        graph_module, monkeypatch, retrieval=lambda s: {"retrieved_docs": docs}, reader=read, checker=check,
    )
    assert calls.count("reader") == 2 and calls.count("retrieval") == 1
    assert [unit["id"] for unit in result["source_units"]] == ["two", "one"]
    assert result["retry_count"] == 1


def test_cpu_min_corrects_a_wrong_number_from_the_selected_pdf(graph_module, monkeypatch, tmp_path):
    from eval.run_benchmark import load_pilot
    from modes.cpu_min.summarizer import CPUMinSummarizerAgent

    load_pilot(tmp_path)
    summarize = CPUMinSummarizerAgent.run

    def faulty_first_draft(self, state):
        if state.get("retry_count"):
            return summarize(self, state)
        claim = "The orchard trial ran for 999 days."
        return {"answer": claim, "claims": [{"id": "claim_1", "text": claim}]}

    monkeypatch.setattr(CPUMinSummarizerAgent, "run", faulty_first_draft)
    result = graph_module.build_research_rag_graph("cpu_min").invoke({
        "query": "How many days did the orchard trial run?", "pdf_path": str(tmp_path / "orchard-trial.pdf"),
    })
    requests = [event for event in result["reasoning_trace"] if event.get("action") == "agent_handoff"]
    assert [(event["sender"], event["recipient"]) for event in requests] == [("fact_checker", "summarizer")]
    assert "999" in requests[0]["focus"][0]
    assert "21 days" in result["final_report"] and "999" not in result["final_report"]
    assert result["fact_check_label"] == "support" and result["evidence_spans"][0]["page_no"] == 1


def test_parallel_retrieval_honors_the_existing_handoff_budget(graph_module, monkeypatch, tmp_path):
    from eval.run_benchmark import load_pilot
    from modes.cpu_min.summarizer import CPUMinSummarizerAgent

    load_pilot(tmp_path)
    summarize = CPUMinSummarizerAgent.run

    def first_draft_requests_search(self, state):
        if state.get("retry_count"):
            assert state["sub_queries"][0] == "duration of the orchard trial"
            return summarize(self, state)
        return {
            "critic_action": "request_more_context",
            "critic_feedback": {"missing_key_points": ["duration of the orchard trial"]},
        }

    monkeypatch.setattr(CPUMinSummarizerAgent, "run", first_draft_requests_search)
    result = graph_module.build_research_rag_graph("cpu_min", parallel_retrieval=True).invoke({
        "query": "How many days did the orchard trial run?",
        "pdf_path": str(tmp_path / "orchard-trial.pdf"),
        "max_reflection_loops": 1,
    })
    trace = result["reasoning_trace"]
    requests = [event for event in trace if event.get("action") == "agent_handoff"]
    assert [(request["sender"], request["recipient"]) for request in requests] == [("summarizer", "retrieval")]
    primary_rounds = [
        event["details"]["retry_count"] for event in trace
        if event.get("action") == "retrieval_task_schedule" and event["details"]["stage"] == "primary"
    ]
    assert primary_rounds == [0, 1]
    assert result["retry_count"] == 1
    assert result["fact_check_label"] == "support" and "21 days" in result["final_report"]


def test_critic_requests_targeted_search_before_checking(graph_module, monkeypatch):
    def retrieve(state):
        if state.get("retry_count"):
            assert state["sub_queries"][0] == "measured yield"
            assert state["pdf_paths"] == ["trial.pdf"]
        return {}

    def summarize(state):
        if state.get("retry_count"):
            return {"answer": "Yield was 14."}
        return {"critic_action": "request_more_context", "critic_feedback": {"missing_key_points": ["measured yield"]}}

    calls, result = run_team(
        graph_module, monkeypatch, retrieval=retrieve, summarizer=summarize,
        checker=lambda s: {"fact_check_label": "support"},
    )
    assert calls == ["retrieval", "reader", "summarizer", "retrieval", "reader", "summarizer", "fact_checker", "writer"]
    assert result["retry_count"] == 1


@pytest.mark.parametrize(("budget", "revisions"), [(0, 0), (3, 1)])
def test_unresolved_review_stops_and_marks_published_report(graph_module, monkeypatch, budget, revisions):
    calls, result = run_team(
        graph_module, monkeypatch, budget=budget,
        checker=lambda s: {"fact_check_label": "contradict", "rejected_claims": [{"claim_text": "Disputed value"}]},
    )
    assert calls.count("summarizer") == revisions + 1 and calls.count("writer") == 1
    assert result["retry_count"] == revisions and result["confidence"] == "low"
    assert "Độ tin cậy thấp" in result["final_report"]


@pytest.mark.parametrize(("decision", "status"), [("clarify", "CLARIFICATION_NEEDED"), ("reject", "OUT_OF_SCOPE")])
def test_scout_can_return_to_user_before_retrieval(graph_module, monkeypatch, decision, status):
    scout = importlib.import_module("agents.agent0_scout.node")
    monkeypatch.setattr(scout, "analyze_intent", lambda q: {"decision": decision, "message": "Please specify a topic."})
    monkeypatch.setattr(graph_module, "get_agent_runners", lambda mode: {})
    result = graph_module.build_research_rag_graph("cpu_min").invoke({"query": "AI"})
    assert result["answer_status"] == status and result["final_report"] == "Please specify a topic."
    assert result["retry_count"] == 0
