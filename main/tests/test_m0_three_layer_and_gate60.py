"""Tests for M0 3-Layer Evaluator, fixed strict_metrics / metrics, and 60-case mechanism gate."""

from __future__ import annotations

import pytest

from eval.m0_regression_gate import GATE_DATASET_PATH, verify_and_time_m0_gate
from eval.metrics import evaluate_single_prediction
from eval.strict_metrics import compute_strict_abstention_accuracy, compute_strict_groundedness
from eval.three_layer_evaluator import (
    audit_abstention_and_display,
    audit_citation_regions,
    calibrate_nli_thresholds,
    check_directed_relation,
    check_math_equivalence_with_domain,
    clopper_pearson_lower_bound,
    compute_cohens_kappa,
    summarize_blind_pairwise,
    verify_derived_calculation,
)


def test_strict_groundedness_fixes_short_wrong_sentence_and_faithful_vi_translation():
    """
    Exact regression from KE_HOACH_TATMANH.md Section 1:
    Source: 'The trial lasted exactly fourteen consecutive days.'
    - Faithful VI translation -> MUST score 1.0 (previously 0.0)
    - Short wrong sentence 'It took 500 days.' (<= 25 chars) -> MUST score 0.0 (previously 1.0)
    """
    evidence = [{"text": "The trial lasted exactly fourteen consecutive days."}]
    assert compute_strict_groundedness("Thử nghiệm kéo dài đúng 14 ngày liên tiếp.", evidence) == 1.0
    assert compute_strict_groundedness("Thử nghiệm kéo dài đúng mười bốn ngày liên tiếp.", evidence) == 1.0
    assert compute_strict_groundedness("It took 500 days.", evidence) == 0.0

    # Negation inversion must score 0.0
    neg_evidence = [{"text": "It is false that treatment works."}]
    assert compute_strict_groundedness("Treatment works.", neg_evidence) == 0.0


def test_pilot_abstention_accepts_insufficient_evidence_and_blocks_leaky_status_display():
    """
    Exact regression from KE_HOACH_TATMANH.md Section 1:
    - Honest INSUFFICIENT_EVIDENCE on trap question -> abstention_accuracy = 1.0, hallucination_rate = 0.0
    - Dishonest status=INSUFFICIENT_EVIDENCE while text states fabricated trap number -> abstention_accuracy = 0.0
    """
    trap = {
        "id": "pilot-unanswerable-vi",
        "category": "adversarial_trap",
        "language": "vi",
        "is_unanswerable": True,
        "forbidden_entities": ["nitrat"],
        "forbidden_numbers": [45],
    }
    honest = evaluate_single_prediction(
        trap,
        {
            "answer_status": "INSUFFICIENT_EVIDENCE",
            "answer": "Chưa đủ bằng chứng trong tài liệu đã chọn để xác nhận câu trả lời.",
        },
        0.1,
    )
    assert honest["abstention_accuracy"] == 1.0
    assert honest["hallucination_rate"] == 0.0

    leaky = evaluate_single_prediction(
        trap,
        {
            "answer_status": "INSUFFICIENT_EVIDENCE",
            "answer": "Nồng độ nitrat trong đất là 45 mg/L.",
        },
        0.1,
    )
    assert leaky["abstention_accuracy"] == 0.0
    assert compute_strict_abstention_accuracy("", is_unanswerable=True) == 0.0


def test_derived_calculation_distinguishes_percentage_points_relative_pct_and_splits():
    spec_pp = {
        "op": "diff_pp",
        "operands": [
            {"entity": "Method B", "metric": "accuracy", "split": "val", "value": 92.0, "unit": "%"},
            {"entity": "Method A", "metric": "accuracy", "split": "val", "value": 81.0, "unit": "%"},
        ],
        "expected_value": 11.0,
        "expected_unit": "pp",
    }
    assert verify_derived_calculation("Phương pháp B cao hơn phương pháp A 11 điểm phần trăm.", spec_pp)["passed"] is True
    assert verify_derived_calculation("Phương pháp A thấp hơn phương pháp B 11 điểm phần trăm.", spec_pp)["passed"] is True

    # Wrong unit: 11% instead of 11 percentage points
    res_pct = verify_derived_calculation("Phương pháp B cao hơn phương pháp A 11%.", spec_pp)
    assert res_pct["passed"] is False and "unit_mismatch" in res_pct["reason"]

    # Relative % (13.58%) when percentage points requested
    res_rel = verify_derived_calculation("Phương pháp B cao hơn phương pháp A 13.58%.", spec_pp)
    assert res_rel["passed"] is False

    # Subject swap ("A cao hơn B 11 điểm phần trăm")
    res_swap = verify_derived_calculation("Phương pháp A cao hơn phương pháp B 11 điểm phần trăm.", spec_pp)
    assert res_swap["passed"] is False and "subject_swapped" in res_swap["reason"]

    # Cross-split subtraction rejected unless explicitly allowed
    cross_split_spec = {
        "op": "diff_pp",
        "operands": [
            {"entity": "Method B", "metric": "accuracy", "split": "validation", "value": 92.0, "unit": "%"},
            {"entity": "Method A", "metric": "accuracy", "split": "test", "value": 79.0, "unit": "%"},
        ],
    }
    res_split = verify_derived_calculation("Method B exceeds Method A by 13 percentage points.", cross_split_spec)
    assert res_split["passed"] is False and "split_mismatch" in res_split["reason"]


def test_domain_preserving_math_ast_guards_subscripts_inequalities_and_nonzero_domain():
    # x^2 = 4 vs x_2 = 4
    assert check_math_equivalence_with_domain("Root condition: $x^2 = 4$", "x^2 = 4")[0] is True
    assert check_math_equivalence_with_domain("Root condition: $x_2 = 4$", "x^2 = 4")[0] is False

    # Inequalities x < 3 and y > 2 (flipped 3 > x is equivalent; x > 3 is false)
    assert check_math_equivalence_with_domain("$3 > x$ and $y > 2$", "x < 3 and y > 2")[0] is True
    assert check_math_equivalence_with_domain("$x > 3$ and $y > 2$", "x < 3 and y > 2")[0] is False

    # x / x = 1 requires explicit x != 0 domain constraint
    ok_with_domain, _ = check_math_equivalence_with_domain(
        "Ta có $x / x = 1$ với $x \\neq 0$.",
        "x / x = 1",
        required_domain_constraints=["x!=0"],
    )
    assert ok_with_domain is True

    ok_without_domain, reason = check_math_equivalence_with_domain(
        "Ta có $x / x = 1$ với mọi x.",
        "x / x = 1",
        required_domain_constraints=["x!=0"],
    )
    assert ok_without_domain is False and "missing_domain_constraint" in reason


def test_layer3_statistics_calibration_and_full_60_case_mechanism_gate():
    # Clopper-Pearson 95% one-sided lower bound for 60/60 is ~0.9513 (95.1%)
    assert clopper_pearson_lower_bound(60, 60, alpha=0.05) == pytest.approx(0.9513, abs=1e-4)
    assert 0.5 < compute_cohens_kappa([1, 1, 0, 0, 1, 0], [1, 1, 0, 0, 0, 0]) < 1.0

    pairwise = summarize_blind_pairwise([
        {"accuracy": "a", "citation_quality": "a"},
        {"accuracy": "tie", "citation_quality": "b"},
    ])
    assert pairwise["dimensions"]["accuracy"]["win_rate_a"] == 0.5
    assert pairwise["dimensions"]["accuracy"]["tie_rate"] == 0.5

    # Run full 60-case M0 regression gate & persist m0_regression_gate_60.json
    gate_report = verify_and_time_m0_gate(run_cpu_baseline_slice=True, write_dataset_file=True)
    assert GATE_DATASET_PATH.is_file()
    assert gate_report["pdf_count"] == 10
    assert gate_report["total_cases"] == 60
    assert gate_report["good_passed"] == 60, gate_report["unexpected_good_failures"]
    assert gate_report["bad_traps_caught"] == 60, gate_report["missed_bad_traps"]
    assert gate_report["passed"] is True
    assert gate_report["slice_10_cpu_min_diagnostic"]["questions_timed"] == 10
