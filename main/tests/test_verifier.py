"""
Automated Unit-Aware Numerical Verifier & Citation Provenance Suite (Đợt 4 — Criteria C & D).
"""

from __future__ import annotations

from core.state import SourceUnit, compute_context_hash
from core.verifier import (
    check_numerical_and_date_consistency,
    verify_evidence_span_provenance,
)


def test_numerical_verifier_unit_equivalence_and_strict_mismatch():
    # 85% vs 85% -> Match
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        "Độ chính xác đạt 85%.", "The model achieves 85% accuracy."
    )
    assert num_ok is True and date_ok is True and missing == []

    # 85% vs 85 percent -> Match
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        "Độ chính xác đạt 85%.", "The model achieves 85 percent accuracy."
    )
    assert num_ok is True and date_ok is True and missing == []

    # 85% vs 85 phần trăm -> Match
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        "Độ chính xác đạt 85%.", "Mô hình đạt độ chính xác 85 phần trăm trên tập kiểm thử."
    )
    assert num_ok is True and date_ok is True and missing == []

    # 85% vs bare 85 -> MUST NOT auto-match
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        "Độ chính xác đạt 85%.", "The dataset contains 85 samples."
    )
    assert num_ok is False
    assert "85%" in missing

    # 85 km vs 85 m -> MUST NOT match
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        "Khoảng cách là 85 km.", "The measured distance is 85 m."
    )
    assert num_ok is False
    assert "85 km" in missing

    # 2025 vs 2024 -> MUST NOT match
    num_ok, date_ok, missing = check_numerical_and_date_consistency(
        "Nghiên cứu công bố năm 2025.", "The study was published in 2024."
    )
    assert date_ok is False
    assert "2025" in missing


def test_numerical_verifier_metric_context_anchor_check():
    long_context = (
        "In our evaluation, BERT achieves 91.2% accuracy on the classification benchmark. "
        + ("Additional background discussion on architecture and training stability. " * 6)
        + "Meanwhile, latency measured for RoBERTa is 45 ms on a single GPU."
    )
    # Correct metric + number association
    ok_good, _, _ = check_numerical_and_date_consistency(
        "BERT achieves 91.2% accuracy.", long_context
    )
    assert ok_good is True

    # Wrong metric/model association (45 ms attached to BERT far away from BERT)
    ok_wrong_anchor, _, _ = check_numerical_and_date_consistency(
        "BERT achieves 45 ms latency.", long_context
    )
    assert ok_wrong_anchor is False


def test_citation_provenance_verifier_single_and_multi_source():
    quote = "Self-attention connects all positions with O(1) sequential operations."
    u1 = SourceUnit(id="u1", text=quote, doc_id="doc_1", page_no=4)
    u2 = SourceUnit(id="u2", text=quote, doc_id="doc_2", page_no=7)
    u1.merge_provenance(u2)
    units = [u1.to_dict()]

    # Valid citation to doc_1 page 4
    ok1, _ = verify_evidence_span_provenance(
        {
            "quote": quote,
            "doc_id": "doc_1",
            "page_no": 4,
            "unit_id": "u1",
            "context_hash": compute_context_hash(quote, "doc_1", 4),
        },
        units,
    )
    assert ok1 is True

    # Valid citation to merged doc_2 page 7
    ok2, _ = verify_evidence_span_provenance(
        {
            "quote": quote,
            "doc_id": "doc_2",
            "page_no": 7,
            "unit_id": "u2",
            "context_hash": compute_context_hash(quote, "doc_2", 7),
        },
        units,
    )
    assert ok2 is True

    # Wrong page number must FAIL
    ok_bad_page, _ = verify_evidence_span_provenance(
        {"quote": quote, "doc_id": "doc_1", "page_no": 99, "unit_id": "u1"},
        units,
    )
    assert ok_bad_page is False

    # Fabricated quote not in source must FAIL
    ok_bad_quote, _ = verify_evidence_span_provenance(
        {"quote": "Fabricated quote not in PDF", "doc_id": "doc_1", "page_no": 4},
        units,
    )
    assert ok_bad_quote is False
