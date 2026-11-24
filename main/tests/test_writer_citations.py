from agents.agent5_writer.components.report_builder import build_academic_report
from core.citation_audit import write_with_citation_audit
from modes.cpu_min.writer import CPUMinWriterAgent


def test_findings_and_bibliography_share_the_linked_source_pages():
    spans = [
        {"id": "a1", "doc_id": "paper-a", "page_no": 2, "quote": "First passage."},
        {"id": "a2", "doc_id": "paper-a", "page_no": 2, "quote": "Second passage."},
        {"id": "b", "document_id": "paper-b", "page": "2", "quote": "Third passage."},
        {"id": "c", "doc_id": "paper-c", "page_no": 5, "quote": "Only retrieved."},
    ]
    report, data = build_academic_report(
        query="What did the papers report?",
        verified_claims=[
            {"claim": "Paper B states the main observation.", "evidence_ids": ["b"]},
            {"claim": "Paper A describes two outcomes.", "evidence_ids": ["a1", "a2"]},
            {"claim": "The older result has no explicit link.", "evidence": "paper-a"},
        ],
        rejected_claims=[],
        retrieved_docs=[{"doc_id": "paper-c", "page_no": 5, "text": "Only retrieved."}],
        evidence_spans=spans,
    )

    assert "Paper B states the main observation. [2]" in report
    assert "Paper A describes two outcomes. [1]\n" in report
    assert "The older result has no explicit link." not in data["academic_report"]
    assert "The older result has no explicit link. *(Trích dẫn chưa xác minh)*" in data["audit_ledger"]
    assert data["citation_renderer"] == "academic_report"
    assert data["citation_references"] == [
        {"ref_id": 1, "document_id": "paper-a", "page": 2, "evidence_id": "a1"},
        {"ref_id": 2, "document_id": "paper-b", "page": 2, "evidence_id": "b"},
    ]


def test_source_markers_are_literal_and_markdown_code_is_preserved():
    answer = "A long source sentence with old markers [2], [u1] and [1, 2].\n\n`values[1]`\n```python\nvalues[1]\n```"
    report, data = build_academic_report(
        query="How are the values stored?",
        verified_claims=[{"text": answer, "evidence_ids": ["ev"]}],
        rejected_claims=[],
        retrieved_docs=[],
        evidence_spans=[{"id": "ev", "doc_id": "paper", "page_no": 1, "quote": answer}],
        answer=answer,
    )

    assert r"\[2], \[u1] and \[1, 2]" in data["answer"]
    assert "`values[1]`" in report
    assert "```python\nvalues[1]\n```" in report
    assert "[1] *Kho tài liệu nội bộ" not in report
    assert [r["evidence_id"] for r in data["citation_references"]] == ["ev"]


def test_an_ambiguous_evidence_id_cannot_be_cited():
    _, data = build_academic_report(
        query="What did the paper report?",
        verified_claims=[{"claim": "An evidence link should resolve to only one source.", "evidence_ids": ["ev"]}],
        rejected_claims=[],
        retrieved_docs=[],
        evidence_spans=[
            {"id": "ev", "doc_id": "paper-a", "page_no": 2, "quote": "First passage."},
            {"id": "ev", "doc_id": "paper-b", "page_no": 2, "quote": "Second passage."},
        ],
    )

    assert data["citation_references"] == []


def test_cpu_writer_publishes_audited_links_only():
    quote_a = "The first treatment improved recall by 20%."
    quote_b = "The second treatment improved recall by 30%."
    state = {
        "query": "By how much did each treatment improve recall?",
        "claims": [
            {"id": "claim-a", "text": quote_a, "evidence_ids": ["unit-b"]},
            {"id": "claim-b", "text": quote_b, "evidence_ids": ["unit-b"]},
        ],
        "verified_claims": [
            {"claim_id": "claim-a", "claim_text": quote_a, "status": "SUPPORT"},
            {"claim_id": "claim-b", "claim_text": quote_b, "status": "SUPPORT"},
        ],
        "source_units": [
            {"id": "unit-a", "doc_id": "paper-a", "page_no": 2, "text": quote_a},
            {"id": "unit-b", "doc_id": "paper-b", "page_no": 2, "text": quote_b},
        ],
        "evidence_spans": [
            {"id": "ev-a", "source_unit_id": "unit-a", "doc_id": "paper-a", "page_no": 2, "quote": quote_a},
            {"id": "ev-b", "source_unit_id": "unit-b", "doc_id": "paper-b", "page_no": 2, "quote": quote_b},
        ],
    }

    output = write_with_citation_audit(state, CPUMinWriterAgent().run)

    assert f"{quote_a} *(Trích dẫn chưa xác minh)*" in output["final_report"]
    assert f"{quote_b} [1]" in output["final_report"]
    audit = output["reasoning_trace"][-1]["citation_audit"]
    assert audit["claims"]["claim-a"]["evidence_ids"] == []
    assert audit["references"] == [{"ref_id": 1, "document_id": "paper-b", "page": 2, "evidence_id": "ev-b"}]
    assert "evidence_ids" not in state["verified_claims"][0]

def test_disputed_draft_and_unlinked_text_stay_out_of_the_published_answer():
    quote = "The trial lasted 14 days."
    state = {
        "query": "How long did the trial last?",
        "answer": "The trial measured productivity for 999 days.",
        "source_units": [{"id": "u", "text": quote, "doc_id": "trial", "page_no": 2}],
        "evidence_spans": [{"id": "e", "source_unit_id": "u", "quote": quote, "doc_id": "trial", "page_no": 2}],
        "verified_claims": [{"text": quote, "evidence_ids": ["e"]}],
        "rejected_claims": [{"text": "The trial lasted 999 days.", "status": "CONTRADICT"}],
    }
    result = write_with_citation_audit(state, CPUMinWriterAgent().run)
    assert result["answer"] == quote + " [1]"
    assert "999" not in result["written_report_data"]["academic_report"]
    assert "999" in result["written_report_data"]["audit_ledger"]
    state["verified_claims"] = []
    result = write_with_citation_audit(state, CPUMinWriterAgent().run)
    assert result["confidence"] == "low"
    assert "999" not in result["answer"]
    assert result["written_report_data"]["status"] == "INSUFFICIENT_EVIDENCE"


def test_gpu_writer_accepts_only_a_permutation_of_verified_findings(monkeypatch):
    from modes.gpu_finetune import writer

    agent = writer.GPUFinetuneWriterAgent()
    monkeypatch.setattr(agent.settings, "REMOTE_INFERENCE_URL", "http://worker.test")
    facts = ["Yield was 14 tonnes.", "Duration was 18 days."]
    state = {
        "query": "Give duration and yield.", "answer": "Incorrect preliminary estimate: 999.",
        "verified_claims": [{"text": fact, "evidence_ids": [f"e{i}"]} for i, fact in enumerate(facts)],
        "evidence_spans": [{"id": f"e{i}", "quote": fact, "page_no": i+1, "doc_id": "paper"}
                           for i, fact in enumerate(facts)],
    }
    for order, first in [([1, 0], facts[1]), ([42], facts[0]), ([1, 1], facts[0])]:
        def remote(**kwargs):
            assert "999" not in str(kwargs)
            assert kwargs["payload"]["claims"][0]["sources"][0]["page"] == 1
            return {"claim_order": order, "report": "Invented value: 777."}
        monkeypatch.setattr(writer.llm_client, "call_kaggle_specialist", remote)
        result = agent.run(state)
        assert result["written_report_data"]["answer"].startswith(first)
        assert "777" not in result["final_report"] and "999" not in result["final_report"]
