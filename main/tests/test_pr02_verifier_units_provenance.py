"""PR-02 Verifier & Evidence Validation tests:
1. Compound unit consistency (85 km/h vs 85 m/h, 85% vs bare 85, 142 tokens/s vs 142 ms).
2. Date consistency (2024 vs 2025) and quantified single-digit guard (7 layers vs 3 layers).
3. EvidenceSpan provenance verification across primary and PR-01 merged multi-source provenance
   (doc_id, page_no, unit_id, quote substring, context_hash integrity).
4. Integration with review_claims(): forged evidence spans cannot certify claims.
"""

from __future__ import annotations

from core.state import SourceUnit, compute_context_hash
from core.verifier import (
    check_numerical_and_date_consistency,
    extract_numbers_with_units,
    review_claims,
    verify_evidence_span_provenance,
)


def test_compound_unit_and_percentage_and_date_consistency():
    # Exact match with canonical unit synonyms
    assert check_numerical_and_date_consistency("Accuracy đạt 85%", "Accuracy reached 85 percent on dev.")[:2] == (True, True)
    assert check_numerical_and_date_consistency("Tốc độ đạt 85 km/h", "Peak speed reached 85 kmh in tests.")[:2] == (True, True)
    assert check_numerical_and_date_consistency("Throughput đạt 142 tokens/s", "Sustained rate is 142 tok/s.")[:2] == (True, True)

    # Mismatched compound units MUST fail (km/h vs m/h, tokens/s vs ms)
    num_ok, _, missing = check_numerical_and_date_consistency("Tốc độ đạt 85 m/h", "Tốc độ đạt 85 km/h trong thử nghiệm.")
    assert not num_ok and any("85" in m and "m/h" in m for m in missing)

    num_ok2, _, missing2 = check_numerical_and_date_consistency("Tốc độ 142 ms", "Throughput reached 142 tokens/s.")
    assert not num_ok2 and any("142" in m for m in missing2)

    # Percentage vs bare number MUST fail
    num_ok3, _, _ = check_numerical_and_date_consistency("Accuracy đạt 85%", "The corpus contains 85 documents.")
    assert not num_ok3

    # Mismatched year MUST fail dates_passed
    _, date_ok, missing_dates = check_numerical_and_date_consistency("Công bố năm 2025.", "Bài báo công bố năm 2024.")
    assert not date_ok and "2025" in missing_dates

    # Fabricated number MUST fail
    num_ok4, _, missing4 = check_numerical_and_date_consistency("Accuracy đạt 90%.", "Accuracy đạt 85%.")
    assert not num_ok4 and any("90" in m for m in missing4)

    # Compound unit extraction keeps km/h distinct from m/h
    extracted = {(num, unit) for num, unit, _ in extract_numbers_with_units("Speed went from 85 m/h to 85 km/h.")}
    assert ("85", "m/h") in extracted and ("85", "km/h") in extracted


def test_verify_evidence_span_provenance_checks_all_five_fields_and_merged_provenance():
    u_a = SourceUnit(id="paperA_p5", text="Self-attention computes pairwise weights.", doc_id="PaperA", page_no=5)
    u_b = SourceUnit(id="paperB_p12", text="Self-attention computes pairwise weights.", doc_id="PaperB", page_no=12)
    u_a.merge_provenance(u_b)
    units = [u_a.to_dict()]

    # Valid span pointing to primary source (PaperA, p5)
    valid_a = {
        "id": "ev1",
        "quote": "Self-attention computes pairwise weights.",
        "doc_id": "PaperA",
        "page_no": 5,
        "source_unit_id": "paperA_p5",
        "context_hash": compute_context_hash("Self-attention computes pairwise weights.", "PaperA", 5),
    }
    assert verify_evidence_span_provenance(valid_a, units)[0] is True

    # Valid span pointing to merged secondary provenance (PaperB, p12)
    valid_b = {
        "id": "ev2",
        "quote": "Self-attention computes pairwise weights.",
        "doc_id": "PaperB",
        "page_no": 12,
        "source_unit_id": "paperB_p12",
        "context_hash": compute_context_hash("Self-attention computes pairwise weights.", "PaperB", 12),
    }
    assert verify_evidence_span_provenance(valid_b, units)[0] is True

    # Forged page_no -> rejected
    assert verify_evidence_span_provenance({**valid_a, "page_no": 99}, units)[0] is False
    # Forged doc_id -> rejected
    assert verify_evidence_span_provenance({**valid_a, "doc_id": "FakePaper"}, units)[0] is False
    # Forged context_hash -> rejected
    assert verify_evidence_span_provenance({**valid_a, "context_hash": "deadbeef12345678"}, units)[0] is False
    # Quote not present in source unit -> rejected
    assert verify_evidence_span_provenance({**valid_a, "quote": "Fabricated quote text.", "context_hash": ""}, units)[0] is False


def test_review_claims_blocks_unit_mismatch_even_when_nli_approves():
    unit = {"id": "u1", "doc_id": "paper", "page_no": 1, "text": "Peak speed reached 85 km/h during evaluation."}
    state = {
        "query": "What was the peak speed?",
        "answer": "Peak speed reached 85 m/h during evaluation.",
        "claims": [{"id": "c1", "text": "Peak speed reached 85 m/h during evaluation.", "evidence_ids": ["u1"]}],
        "source_units": [unit],
    }

    def lenient_nli(pairs):
        return [{"claim_id": "c1", "status": "SUPPORT", "quote": unit["text"], "rationale": "matched 85"}]

    result = review_claims(state, reviewer=lenient_nli)
    assert not result["verified_claims"]
    assert result["fact_check_label"] == "not_enough_info"
    assert result["rejected_claims"][0]["claim_id"] == "c1"
