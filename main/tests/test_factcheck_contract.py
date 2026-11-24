"""Exercise provider response contracts at the actual fact-checker entry points."""
import json

import pytest

from core.config import settings
from core.verifier import review_claims
from utils.llm_client import llm_client


CLAIM = "Model A was faster than Model B."
SOURCE = "Model A was slower than Model B."


def state(answer=CLAIM, source=SOURCE):
    return {"query": "How did the model speeds compare?", "answer": answer,
            "claims": [{"id": "measurement", "text": answer}],
            "source_units": [{"id": "model-p1", "doc_id": "paper", "page_no": 1, "text": source}]}


@pytest.mark.parametrize("output", [[], [{"claim_id": "unknown", "status": "SUPPORT", "quote": SOURCE}],
                                    [{"claim_id": "measurement", "status": "UNSUPPORTED", "quote": SOURCE}],
                                    [{"claim_id": "measurement", "status": "SUPPORT", "quote": "bogus"}],
                                    [{"claim_id": "measurement", "status": "SUPPORT", "quote": SOURCE}] * 2])
@pytest.mark.parametrize("mode", ["gemini", "gpu_max"])
def test_invalid_batch_cannot_support(monkeypatch, output, mode):
    from modes.gemini.fact_checker import GeminiFactCheckerAgent
    from modes.gpu_max.fact_checker import GPUMaxFactCheckerAgent
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "http://example.invalid")
    monkeypatch.setattr(llm_client, "_call_gemini", lambda **kw: json.dumps({"results": output}))
    monkeypatch.setattr(llm_client, "call_gpu_max_32b", lambda **kw: {"parsed": {"results": output}})
    checker = GeminiFactCheckerAgent() if mode == "gemini" else GPUMaxFactCheckerAgent()
    result = checker.run(state())
    assert not result["verified_claims"]
    assert result["fact_check_label"] == "not_enough_info"
    assert [c["claim_id"] for c in result["rejected_claims"]] == ["measurement"]


def test_all_claims_including_ninth_checked_with_input_ids():
    data = state(source=CLAIM)
    data["claims"] = [{"id": f"custom-{n}", "text": CLAIM} for n in range(9)]
    def review(pairs):
        return [{"claim_id": p["claim_id"], "status": "SUPPORT", "quote": p["evidence"]} for p in pairs]
    result = review_claims(data, review)
    assert [c["claim_id"] for c in result["verified_claims"]] == [f"custom-{n}" for n in range(9)]
    assert all(c["evidence_ids"] == ["model-p1"] for c in result["verified_claims"])


@pytest.mark.parametrize("mode", ["gpu_finetune"])
@pytest.mark.parametrize("label,reason,overall", [
    ("CONTRADICT", "BGE NLI Head: CONTRADICT", "contradict"),
    ("NOT_ENOUGH_INFO", "BGE NLI Head: NOT_ENOUGH_INFO", "not_enough_info"),
    ("SUPPORT", "mock/fallback", "not_enough_info"),
    ("SUPPORT", "BGE NLI Head: SUPPORT", "support"),
])
def test_worker_verdict_contract(mode, label, reason, overall, monkeypatch):
    from modes.gpu_finetune.fact_checker import GPUFinetuneFactCheckerAgent
    checker = GPUFinetuneFactCheckerAgent()
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "http://example.invalid")
    def mock_worker(payload, fallback_prompt, **kw):
        assert SOURCE in fallback_prompt and CLAIM in fallback_prompt
        assert payload["claims"] == [CLAIM] and payload["context"] == SOURCE
        return {"evaluations": [{"claim": CLAIM, "verdict": label, "reason": reason, "confidence": 0.85}]}
    monkeypatch.setattr(llm_client, "call_kaggle_specialist", mock_worker)
    result = checker.run(state())
    assert result["fact_check_label"] == overall
    checked, = result["verified_claims"] + result["rejected_claims"]
    assert checked["claim_id"] == "measurement"


@pytest.mark.parametrize("mode", ["cpu_min", "cpu_max"])
def test_offline_overlapping_words_and_false_refusal(mode):
    from modes.cpu_min.fact_checker import CPUMinFactCheckerAgent
    from modes.cpu_max.fact_checker import CPUMaxFactCheckerAgent
    agent = CPUMinFactCheckerAgent() if mode == "cpu_min" else CPUMaxFactCheckerAgent()
    assert not agent.run(state())["verified_claims"]
    assert agent.run(state(source=CLAIM))["verified_claims"][0]["claim_id"] == "measurement"
    refusal = state(answer="Thời gian không được đề cập.", source="The trial lasted 14 days.")
    refusal["query"] = "How long did the trial last?"
    result = agent.run(refusal)
    assert result["fact_check_label"] == "not_enough_info" and result["evidence_gaps"]


def test_table_uses_the_page_from_which_it_was_extracted(monkeypatch, tmp_path):
    from modes.gpu_finetune import reader
    pdf = tmp_path / "example.pdf"
    pdf.touch()
    monkeypatch.setattr(settings, "REMOTE_INFERENCE_URL", "")
    monkeypatch.setattr(reader, "extract_table_or_equations_tool", lambda path, page: f"Table from {page}")
    result = reader.GPUFinetuneReaderAgent().run({"pdf_path": str(pdf), "retrieved_docs": [
        {"id": f"u{p}", "page_no": p, "doc_id": "paper", "text": "Passage."} for p in (2, 5)]})
    tables = [(u["page_no"], u["text"]) for u in result["source_units"] if u["id"].startswith("table_")]
    assert tables == [(2, "Table from 2"), (5, "Table from 5")]
