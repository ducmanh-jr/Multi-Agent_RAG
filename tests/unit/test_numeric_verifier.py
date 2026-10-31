"""
Unit tests for Deterministic Numerical Verifier:
  - Number extraction
  - Quantitative grounding against source context
"""

from agents.agent4_fact_checker.components.numeric_verifier import (
    extract_numbers,
    verify_numbers_in_claim,
)


def test_extract_numbers():
    text = "Achieved 40.97% Hit@5 with 18,587 vectors across 3.5 seconds."
    numbers = extract_numbers(text)
    assert "40.97%" in numbers
    assert "5" in numbers
    assert "18587" in numbers
    assert "3.5" in numbers


def test_verify_numbers_grounded():
    claim = "The accuracy reached 85.5% on the test set."
    context = "Our model demonstrated an accuracy reached 85.5% on the evaluation test set."
    is_valid, reason = verify_numbers_in_claim(claim, context)
    assert is_valid is True


def test_verify_numbers_hallucinated():
    claim = "The accuracy reached 99.9% on the test set."
    context = "Our model demonstrated an accuracy reached 85.5% on the evaluation test set."
    is_valid, reason = verify_numbers_in_claim(claim, context)
    assert is_valid is False
    assert "99.9%" in reason


def test_verify_numbers_percentage_ratio_normalization():
    claim = "The model achieved 40.97% accuracy on QASPER."
    context = "On the QASPER dataset, dense retrieval achieved 0.4097 accuracy."
    is_valid, reason = verify_numbers_in_claim(claim, context)
    assert is_valid is True


def test_verify_numbers_memory_unit_normalization():
    claim = "Peak memory consumption was reduced to 1.5 GB."
    context = "The footprint was measured at 1500 MB under max load."
    is_valid, reason = verify_numbers_in_claim(claim, context)
    assert is_valid is True


def test_verify_numbers_rounding_tolerance():
    claim = "The system achieved 41% accuracy on the benchmark."
    context = "The exact measured score was 40.97% on the benchmark."
    is_valid, reason = verify_numbers_in_claim(claim, context)
    assert is_valid is True
