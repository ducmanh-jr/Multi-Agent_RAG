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
