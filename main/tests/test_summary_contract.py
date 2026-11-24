"""Contracts at the model/Reader boundary; these tests make no provider calls."""

import json

import pytest

from agents.agent3_summarizer.components.critic import critique_summary_and_claims
from agents.agent3_summarizer.components.generator import bind_evidence_spans, bind_summary_claims
from core.schemas import SummarizerCriticOutput
from modes.gemini.summarizer import GeminiSummarizerAgent
from utils.llm_client import llm_client


QUOTE = "The trial measured 42 samples."
UNIT = {"id": "unit-A", "doc_id": "trial", "page_no": 3, "text": QUOTE}
STATE = {"query": "How many samples were measured in the trial?", "source_units": [UNIT]}


def test_quote_binding_uses_unique_source_and_rejects_wrong_location():
    units = [UNIT, {**UNIT, "id": "unit-B", "doc_id": "another"}]
    assert not bind_evidence_spans([QUOTE, {"quote": QUOTE, "unit_id": "invalid"}], units)
    assert not bind_evidence_spans([{"quote": QUOTE, "unit_id": "unit-A", "page_no": 9}], units)
    assert not bind_evidence_spans(["2 samples"], [UNIT])
    spans = bind_evidence_spans([{"quote": QUOTE, "unit_id": "unit-A"}], units)
    claim, = bind_summary_claims([{"text": "42 samples were measured.", "unit_id": "unit-A"}], spans)
    assert spans[0]["doc_id"] == "trial" and spans[0]["page_no"] == 3
    assert spans[0]["id"] in claim["evidence_ids"]
    assert not bind_summary_claims([{"text": QUOTE, "unit_id": "invalid"}], spans)


@pytest.mark.parametrize("response", [{}, {"is_supported": "false", "action": "approve"},
                                     {"is_supported": True, "action": "nonsense"},
                                     {"is_supported": False, "action": "accept"},
                                     {"is_supported": True, "action": "approve", "unsupported_claims": ["wrong quote"]}])
def test_invalid_or_conflicting_review_never_approves(response):
    out = SummarizerCriticOutput.from_dict(response)
    assert not out.is_supported and out.action != "approve"


def test_legacy_valid_critic_result_and_rewrite_feedback():
    assert SummarizerCriticOutput.from_dict({"is_supported": True, "action": "accept"}).action == "approve"
    out = SummarizerCriticOutput.from_dict({"action": "rewrite", "critique": "Wrong page", "issues_found": ["Missing unit"]})
    assert out.critique == "Wrong page" and out.issues_found == ["Missing unit"]


@pytest.mark.parametrize("decision", [{}, {"approved_claim_ids": [], "critic_warnings": []}])
def test_empty_local_critic_response_never_certifies_input_claims(monkeypatch, decision):
    monkeypatch.setattr(llm_client, "generate_json", lambda *args, **kwargs: decision)
    answer, selected, warnings = critique_summary_and_claims(QUOTE, [{"id": "c1", "text": QUOTE}], [])
    assert not answer and not selected and warnings


@pytest.mark.parametrize("status", ["NOT_FOUND_IN_DOCUMENT", "CLARIFICATION_NEEDED"])
def test_terminal_summary_has_no_manufactured_evidence(monkeypatch, status):
    response = {"answer": "Insufficient information.", "answer_status": status, "claims": [], "evidence_spans": [],
                "is_supported": False, "critic_action": "request_more_context"}
    monkeypatch.setattr(llm_client, "_call_gemini", lambda **kw: json.dumps(response))
    r = GeminiSummarizerAgent().run(STATE)
    assert r["answer_status"] == status and r["critic_action"] != "approve"
    assert r["claims"] == [] and r["evidence_spans"] == []


def test_gemini_accepts_grounded_structured_answer(monkeypatch):
    response = {"answer": QUOTE, "answer_status": "ANSWERED", "claims": [{"text": QUOTE, "unit_id": UNIT["id"]}],
                "evidence_spans": [{"quote": QUOTE, "unit_id": UNIT["id"]}], "is_supported": True, "critic_action": "approve"}
    monkeypatch.setattr(llm_client, "_call_gemini", lambda **kw: json.dumps(response))
    result = GeminiSummarizerAgent().run(STATE)
    assert result["critic_action"] == "approve" and result["answer"] == QUOTE
    assert result["evidence_spans"][0]["id"] in result["claims"][0]["evidence_ids"]


@pytest.mark.parametrize("mode,cls_name", [
    ("gpu_finetune", "GPUFinetuneSummarizerAgent"),
    ("gpu_max", "GPUMaxSummarizerAgent"),
])
def test_gpu_modes_preserve_unit_and_verifiable_span_link(monkeypatch, mode, cls_name):
    from importlib import import_module

    module = import_module(f"modes.{mode}.summarizer")
    agent = getattr(module, cls_name)()
    monkeypatch.setattr(agent.settings, "REMOTE_INFERENCE_URL", "http://unused.invalid")
    data = {"answer": QUOTE, "claims": [{"text": QUOTE, "unit_id": UNIT["id"]}],
            "evidence_spans": [{"quote": QUOTE, "unit_id": UNIT["id"]}],
            "is_supported": True, "critic_action": "approve"}
    if mode == "gpu_max":
        monkeypatch.setattr(llm_client, "call_gpu_max_32b", lambda **kw: {"parsed": data})
    else:
        def specialist(endpoint, **kwargs):
            if endpoint.endswith("critic"):
                return {"is_supported": True, "action": "accept"}
            return data
        monkeypatch.setattr(llm_client, "call_kaggle_specialist", specialist)
    result = agent.run(STATE)
    assert result["critic_action"] == "approve"
    assert result["evidence_spans"][0]["source_unit_id"] == UNIT["id"]
    assert result["evidence_spans"][0]["id"] in result["claims"][0]["evidence_ids"]
