import json
from pathlib import Path

import pytest

from eval.metrics import evaluate_single_prediction
from eval.run_benchmark import _summarize_records, write_results


def test_pr05_hallucination_and_unit_mismatch_audit_on_holdout_schema():
    question = {
        "id": "holdout-numeric-sample",
        "category": "numeric",
        "language": "en",
        "documents": ["beir-2104.08663v4.pdf"],
        "gold_values": [{"name": "latency", "value": 20, "unit": "ms"}],
        "gold_sources": [
            {"document": "beir-2104.08663v4.pdf", "page": 8, "anchors": ["DBPedia", "20ms"]}
        ],
    }
    source_unit = {
        "id": "u_1",
        "doc_id": "beir-2104.08663v4.pdf",
        "page_no": 8,
        "text": "On DBPedia, BM25 achieves 20ms latency at 12 km/h conveyor speed.",
    }
    # Case 1: Correct grounded answer + valid provenance + zero orphan markers
    good_state = {
        "answer": "On DBPedia, BM25 achieves 20ms latency at 12 km/h conveyor speed. [1]",
        "final_report": "On DBPedia, BM25 achieves 20ms latency at 12 km/h conveyor speed. [1]",
        "source_units": [source_unit],
        "retrieved_docs": [source_unit],
        "evidence_spans": [
            {
                "id": "sp_1",
                "source_unit_id": "u_1",
                "doc_id": "beir-2104.08663v4.pdf",
                "page_no": 8,
                "quote": "On DBPedia, BM25 achieves 20ms latency",
            }
        ],
        "reasoning_trace": [
            {
                "action": "citation_audit",
                "citation_audit": {
                    "references": [
                        {
                            "ref_id": 1,
                            "document_id": "beir-2104.08663v4.pdf",
                            "page": 8,
                            "evidence_id": "sp_1",
                        }
                    ]
                },
            }
        ],
    }
    good = evaluate_single_prediction(question, good_state, 0.25)
    assert good["groundedness"] == 1.0
    assert good["hallucination_rate"] == 0.0
    assert good["citation_f1"] == 1.0
    assert good["citation_provenance_rate"] == 1.0
    assert good["orphan_citation_count"] == 0
    assert good["valid_citation_ratio"] == 1.0
    assert good["unit_mismatch_count"] == 0
    assert good["numeric_exactness"] == 1.0

    # Case 2: Compound-unit mismatch (m/h vs km/h), forged span quote, and orphan [9] marker
    bad_state = {
        "answer": "On DBPedia, BM25 achieves 20ms latency at 12 m/h conveyor speed. [1] [9]",
        "final_report": "On DBPedia, BM25 achieves 20ms latency at 12 m/h conveyor speed. [1] [9]",
        "source_units": [source_unit],
        "retrieved_docs": [source_unit],
        "evidence_spans": [
            {
                "id": "sp_bad",
                "source_unit_id": "u_1",
                "doc_id": "beir-2104.08663v4.pdf",
                "page_no": 8,
                "quote": "Fabricated sentence not in the PDF at all.",
            }
        ],
        "reasoning_trace": [
            {
                "action": "citation_audit",
                "citation_audit": {
                    "references": [
                        {
                            "ref_id": 1,
                            "document_id": "beir-2104.08663v4.pdf",
                            "page": 8,
                            "evidence_id": "sp_bad",
                        }
                    ]
                },
            }
        ],
    }
    bad = evaluate_single_prediction(question, bad_state, 0.30)
    assert bad["unit_mismatch_count"] >= 1
    assert bad["numeric_exactness"] == 0.0
    assert bad["citation_provenance_rate"] == 0.0
    assert bad["orphan_citation_count"] == 1
    assert bad["valid_citation_ratio"] == pytest.approx(0.5)


def test_pr05_insufficient_evidence_category_abstention_vs_hallucination():
    trap = {
        "id": "holdout-insufficient-01",
        "category": "insufficient_evidence",
        "language": "en",
        "documents": ["beir-2104.08663v4.pdf"],
        "gold_sources": [],
    }
    abstained = evaluate_single_prediction(
        trap,
        {
            "answer_status": "NOT_FOUND_IN_DOCUMENT",
            "answer": "Chưa đủ bằng chứng trong tài liệu đã chọn để xác nhận câu trả lời.",
            "claims": [{"id": "c1", "claim": "Chưa đủ bằng chứng trong tài liệu đã chọn để xác nhận câu trả lời."}],
            "source_units": [{"id": "u1", "doc_id": "beir-2104.08663v4.pdf", "page_no": 1, "text": "BEIR benchmark."}],
        },
        0.1,
    )
    assert abstained["abstention_accuracy"] == 1.0
    assert abstained["hallucination_rate"] == 0.0
    assert abstained["groundedness"] == 1.0
    assert abstained["numeric_exactness"] is None

    hallucinated = evaluate_single_prediction(
        trap,
        {"answer": "The authors paid $420 per query on cloud GPUs."},
        0.1,
    )
    assert hallucinated["abstention_accuracy"] == 0.0
    assert hallucinated["hallucination_rate"] == 1.0


def test_pr05_merged_provenance_passes_citation_provenance_audit():
    question = {"id": "multi-prov", "category": "multi_document", "language": "en", "gold_sources": []}
    merged_unit = {
        "id": "u_merged",
        "doc_id": "paper_a.pdf",
        "page_no": 5,
        "text": "Self-attention relates different positions of a single sequence.",
        "provenance": [
            {"doc_id": "paper_a.pdf", "page_no": 5, "source_unit_id": "u_merged"},
            {"doc_id": "paper_b.pdf", "page_no": 12, "source_unit_id": "u_b_12"},
        ],
    }
    state = {
        "answer": "Self-attention relates different positions of a single sequence.",
        "source_units": [merged_unit],
        "evidence_spans": [
            {
                "id": "sp_b",
                "source_unit_id": "u_merged",
                "doc_id": "paper_b.pdf",
                "page_no": 12,
                "quote": "Self-attention relates different positions",
            }
        ],
    }
    scored = evaluate_single_prediction(question, state, 0.1)
    assert scored["citation_provenance_rate"] == 1.0


def test_pr05_benchmark_report_writes_audit_columns_and_breakdowns(tmp_path: Path):
    records = [
        {
            "id": "q1", "category": "numeric", "language": "en",
            "groundedness": 1.0, "hallucination_rate": 0.0, "retrieval_recall": 1.0,
            "citation_f1": 1.0, "citation_provenance_rate": 1.0, "valid_citation_ratio": 1.0,
            "orphan_citation_count": 0, "unit_mismatch_count": 0,
            "numeric_exactness": 1.0, "abstention_accuracy": 1.0,
            "latency_sec": 0.4, "retry_count": 0,
        },
        {
            "id": "q2", "category": "vietnamese_english", "language": "vi",
            "groundedness": 0.5, "hallucination_rate": 0.5, "retrieval_recall": 0.5,
            "citation_f1": 0.5, "citation_provenance_rate": 1.0, "valid_citation_ratio": 0.5,
            "orphan_citation_count": 1, "unit_mismatch_count": 1,
            "numeric_exactness": 0.0, "abstention_accuracy": 1.0,
            "latency_sec": 0.8, "retry_count": 1,
        },
    ]
    summary = {
        "mode": "cpu_min",
        "num_questions": 2,
        **_summarize_records(records),
        "by_category": {
            "numeric": _summarize_records([records[0]]),
            "vietnamese_english": _summarize_records([records[1]]),
        },
        "by_language": {
            "en": _summarize_records([records[0]]),
            "vi": _summarize_records([records[1]]),
        },
        "records": records,
        "errors": [],
    }
    write_results(tmp_path, {"dataset": "holdout.jsonl"}, [summary])
    report_md = (tmp_path / "benchmark_report.md").read_text(encoding="utf-8")
    payload = json.loads((tmp_path / "benchmark_results.json").read_text(encoding="utf-8"))

    assert "Hallucination" in report_md
    assert "Citation provenance" in report_md
    assert "category:vietnamese_english" in report_md
    assert "language:vi" in report_md
    assert payload["summaries"][0]["orphan_citation_total"] == 1
    assert payload["summaries"][0]["unit_mismatch_total"] == 1
