"""Contracts shared by discovery, local planning and enhanced retrieval."""
import importlib
import json
import sys
from types import SimpleNamespace

import pytest

from agents.agent0_scout.query_expander import analyze_intent
from core.config import settings
from core.supervisor import supervisor_planner


@pytest.mark.parametrize("query,decision", [("AI", "clarify"), ("tìm bài hay", "reject"), ("???", "reject")])
def test_scout_stops_before_search(monkeypatch, query, decision):
    from agents.agent0_scout import node

    def unexpected(*args, **kwargs):
        pytest.fail("A terminal decision attempted literature discovery")
    monkeypatch.setattr(node, "search_arxiv_papers", unexpected)
    monkeypatch.setattr(node, "search_gemini_academic_papers", unexpected)
    state = node.ScoutAgent().run({"query": query})
    assert state["scout_decision"] == decision
    assert state["final_report"]


def test_scout_routes_humanities_and_recent_search(monkeypatch):
    from agents.agent0_scout.arxiv_client import search_arxiv_papers

    assert analyze_intent("Tìm tài liệu về lịch sử Việt Nam")["domain"] == "NON_STEM"
    recent, = [p for p in analyze_intent("Transformer")["perspectives"] if p["angle"] == "Recent"]
    searches = []
    fake_sdk = SimpleNamespace(
        Client=lambda **kw: SimpleNamespace(results=lambda search: []),
        Search=lambda **args: searches.append(args),
        SortCriterion=SimpleNamespace(SubmittedDate="date", Relevance="relevance"),
        SortOrder=SimpleNamespace(Descending="desc"),
    )
    monkeypatch.setitem(sys.modules, "arxiv", fake_sdk)
    search_arxiv_papers(recent["query"], recent=True)
    assert searches[0]["sort_by"] == "date" and searches[0]["query"] == "Transformer"


def test_single_method_limitation_and_explicit_comparison():
    assert supervisor_planner.plan_query("Nhược điểm của phương pháp Transformer là gì?").intent == "limitation"
    assert supervisor_planner.plan_query("So sánh nhược điểm của Transformer và LSTM").intent == "compare"


@pytest.mark.parametrize("mode,class_name", [
    ("gpu_finetune", "GPUFinetuneRetrievalAgent"),
])
def test_retrieval_scopes_to_every_selected_pdf(monkeypatch, tmp_path, mode, class_name):
    module = importlib.import_module(f"modes.{mode}.retrieval")
    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "absent"))
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "")
    monkeypatch.setattr(module, "load_or_parse_selected_pdf", lambda p: [
        {"id": f"chunk-{p}", "doc_id": p, "text": f"latency {p}"}
    ] if p != "unreadable" else [])
    agent = getattr(module, class_name)()
    agent.corpus_docs = [{"id": "outside", "text": "latency from unrelated paper"}]
    state = {"query": "latency", "pdf_path": "first", "pdf_paths": ["first", "second"]}
    result = agent.run(state)
    assert {d["doc_id"] for d in result["retrieved_docs"]} == {"first", "second"}
    assert agent.run({**state, "pdf_paths": ["unreadable"]})["retrieved_docs"] == []


def test_hybrid_plan_is_validated_and_changes_ranking(monkeypatch, tmp_path):
    from modes.hybrid import retrieval

    monkeypatch.setattr(settings, "CORPUS_PATH", str(tmp_path / "absent"))
    monkeypatch.setattr(retrieval, "probe_all_capabilities", lambda: {"gemini": {"online": True}})
    monkeypatch.setattr(retrieval, "generate_research_perspectives", lambda q: ([], []))
    plan = {"sub_queries": ["duration"], "required_entities": ["benchmark"], "alpha": 0.0}

    def generate(prompt, **kwargs):
        assert json.loads(prompt)["question"] == "latency"
        assert "sub_queries" in kwargs["system_instruction"]
        return plan

    monkeypatch.setattr(retrieval.llm_client, "generate_json", generate)
    docs = [{"id": token, "doc_id": token, "text": f"benchmark {token}"} for token in ("latency", "duration")]
    monkeypatch.setattr(retrieval, "load_or_parse_selected_pdf", lambda p: docs)
    agent = retrieval.HybridRetrievalAgent()
    state = {"query": "latency", "pdf_path": "paper"}
    for weight, first in [(0.0, "latency"), (1.0, "duration")]:
        plan["alpha"] = weight
        result = agent.run(state)
        assert result["required_entities"] == ["benchmark"]
        assert result["sub_queries"][0] == "latency"
        assert result["retrieved_docs"][0]["id"] == first

    plan["alpha"] = 5  # reject the entire response, including its entities and sub-queries
    fallback = agent.run(state)
    assert "duration" not in fallback["sub_queries"]
    assert "benchmark" not in fallback["required_entities"]
