"""
Unit tests for Bilingual Date & Temporal Consistency Verifier.
"""

from agents.agent4_fact_checker.components.dates import (
    extract_date_mentions,
    verify_dates_in_claim,
)


def test_extract_dates_english():
    text = "The paper was published on September 15, 2026 by DeepMind."
    mentions = extract_date_mentions(text)
    assert len(mentions) > 0
    assert mentions[0].year == 2026


def test_extract_dates_vietnamese():
    text = "Nghiên cứu công bố ngày 15 tháng 9 năm 2026."
    mentions = extract_date_mentions(text)
    assert len(mentions) > 0
    assert mentions[0].year == 2026


def test_verify_dates_grounded():
    claim = "The model was released in 2023."
    context = "QLoRA was first proposed in 2023 for efficient fine-tuning."
    is_valid, reason = verify_dates_in_claim(claim, context)
    assert is_valid is True


def test_verify_dates_hallucinated():
    claim = "The model was released in 1985."
    context = "QLoRA was first proposed in 2023 for efficient fine-tuning."
    is_valid, reason = verify_dates_in_claim(claim, context)
    assert is_valid is False
    assert "1985" in reason
