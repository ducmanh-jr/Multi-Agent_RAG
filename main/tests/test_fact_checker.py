"""
Automated Fact-Checker Per-Claim & Citation Provenance Suite (Đợt 4 — Criteria B, C, F).

Verifies:
  1. Independent per-claim evaluation fields: status, numerical_passed, dates_passed, provenance_ok.
  2. Nuanced EvidenceStatus aggregation: FOUND, PARTIAL, NOT_FOUND, CONFLICTING_EVIDENCE
     (never conflating missing evidence with contradiction, nor 1 minor issue with full conflict).
  3. Citation Provenance verification against single-source and merged multi-PDF provenance,
     including context_hash integrity.
  4. Truncated context_coverage does NOT downgrade FOUND to PARTIAL when all claims are supported.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from agents.agent4_fact_checker.node import FactCheckerAgent
from core.state import SourceUnit, compute_context_hash


def _make_fc_with_judge(side_effect_fn) -> FactCheckerAgent:
    fc = FactCheckerAgent()
    fc.judge = MagicMock()
    fc.judge.judge_claim.side_effect = side_effect_fn
    return fc


def test_fact_checker_found_even_when_context_truncated():
    u1 = SourceUnit(
        id="u1",
        text="BERT achieves 92.5% accuracy in 2019 on the benchmark.",
        doc_id="paper_a",
        page_no=3,
    ).to_dict()
    quote = "BERT achieves 92.5% accuracy in 2019"
    span = {
        "id": "ev_1",
        "quote": quote,
        "source_unit_id": "u1",
        "doc_id": "paper_a",
        "page_no": 3,
        "context_hash": compute_context_hash(quote, "paper_a", 3),
    }

    fc = _make_fc_with_judge(
        lambda c, ev, ctx: {"status": "support", "confidence": 0.95, "evidence": quote, "reasoning": "Supported"}
    )
    out = fc.run(
        {
            "query": "What is BERT accuracy?",
            "answer": "BERT achieves 92.5% accuracy in 2019.",
            "claims": [{"id": "c1", "text": "BERT achieves 92.5% accuracy in 2019.", "evidence_ids": ["ev_1"]}],
            "evidence_spans": [span],
            "source_units": [u1],
            "context_coverage": {"total_units": 50, "included_units": 10, "omitted_units": 40, "truncated": True},
        }
    )

    assert out["evidence_status"] == "FOUND"
    assert out["fact_check_label"] == "support"
    assert len(out["verified_claims"]) == 1
    vc = out["verified_claims"][0]
    assert vc["numerical_passed"] is True
    assert vc["dates_passed"] is True
    assert vc["provenance_ok"] is True


def test_fact_checker_partial_vs_not_found_vs_conflicting():
    u1 = SourceUnit(
        id="u1",
        text="Model A achieves 88% accuracy. Model B was evaluated on dataset X.",
        doc_id="paper_a",
        page_no=1,
    ).to_dict()
    span = {
        "id": "ev_1",
        "quote": "Model A achieves 88% accuracy.",
        "source_unit_id": "u1",
        "doc_id": "paper_a",
        "page_no": 1,
        "context_hash": compute_context_hash("Model A achieves 88% accuracy.", "paper_a", 1),
    }

    # Case 1: 2 supported + 1 not_enough_info -> PARTIAL (NOT CONFLICTING_EVIDENCE)
    statuses_partial = iter(["support", "support", "not_enough_info"])
    fc_partial = _make_fc_with_judge(
        lambda c, ev, ctx: {"status": next(statuses_partial), "confidence": 0.85, "evidence": "", "reasoning": "ok"}
    )
    out_partial = fc_partial.run(
        {
            "query": "Compare A and B",
            "answer": "Model A achieves 88% accuracy. Model B uses dataset X. Model C is fastest.",
            "claims": [
                {"id": "c1", "text": "Model A achieves 88% accuracy."},
                {"id": "c2", "text": "Model B was evaluated on dataset X."},
                {"id": "c3", "text": "Model C is fastest."},
            ],
            "evidence_spans": [span],
            "source_units": [u1],
        }
    )
    assert out_partial["evidence_status"] == "PARTIAL"
    assert out_partial["fact_check_label"] == "not_enough_info"

    # Case 2: 2 supported + 1 minor contradict -> PARTIAL (1 claim alone does not force full CONFLICTING_EVIDENCE)
    statuses_mixed = iter(["support", "support", "contradict"])
    fc_mixed = _make_fc_with_judge(
        lambda c, ev, ctx: {"status": next(statuses_mixed), "confidence": 0.85, "evidence": "", "reasoning": "ok"}
    )
    out_mixed = fc_mixed.run(
        {
            "query": "Compare A and B",
            "answer": "Three claims",
            "claims": [
                {"id": "c1", "text": "Model A achieves 88% accuracy."},
                {"id": "c2", "text": "Model B was evaluated on dataset X."},
                {"id": "c3", "text": "Model B fails on dataset X."},
            ],
            "evidence_spans": [span],
            "source_units": [u1],
        }
    )
    assert out_mixed["evidence_status"] == "PARTIAL"

    # Case 3: Missing evidence only (0 support, all not_enough_info) -> NOT_FOUND (never CONFLICTING_EVIDENCE)
    fc_nf = _make_fc_with_judge(
        lambda c, ev, ctx: {"status": "not_enough_info", "confidence": 0.4, "evidence": "", "reasoning": "missing"}
    )
    out_nf = fc_nf.run(
        {
            "query": "What about Mars?",
            "answer": "Mars has water.",
            "claims": [{"id": "c1", "text": "Mars has water."}],
            "evidence_spans": [],
            "source_units": [u1],
        }
    )
    assert out_nf["evidence_status"] == "NOT_FOUND"
    assert out_nf["fact_check_label"] == "not_enough_info"

    # Case 4: Real contradiction (contradict >= support) -> CONFLICTING_EVIDENCE
    fc_conflict = _make_fc_with_judge(
        lambda c, ev, ctx: {"status": "contradict", "confidence": 0.9, "evidence": "", "reasoning": "contradicts source"}
    )
    out_conflict = fc_conflict.run(
        {
            "query": "What is Model A accuracy?",
            "answer": "Model A achieves 50% accuracy.",
            "claims": [{"id": "c1", "text": "Model A achieves 50% accuracy."}],
            "evidence_spans": [span],
            "source_units": [u1],
        }
    )
    assert out_conflict["evidence_status"] == "CONFLICTING_EVIDENCE"
    assert out_conflict["fact_check_label"] == "contradict"


def test_fact_checker_provenance_check_with_merged_multi_pdf_and_bad_hash():
    shared_text = "LoRA reduces trainable parameters by 10000x."
    su1 = SourceUnit(id="u_pdf1", text=shared_text, doc_id="doc_1", page_no=4)
    su2 = SourceUnit(id="u_pdf2", text=shared_text, doc_id="doc_2", page_no=7)
    su1.merge_provenance(su2)
    merged_unit = su1.to_dict()

    # Span citing doc_2 page 7 (secondary merged provenance) must PASS provenance check!
    valid_span_doc2 = {
        "id": "ev_2",
        "quote": shared_text,
        "source_unit_id": "u_pdf2",
        "doc_id": "doc_2",
        "page_no": 7,
        "context_hash": compute_context_hash(shared_text, "doc_2", 7),
    }
    fc = _make_fc_with_judge(
        lambda c, ev, ctx: {"status": "support", "confidence": 0.95, "evidence": shared_text, "reasoning": "ok"}
    )
    out_ok = fc.run(
        {
            "query": "LoRA reduction?",
            "answer": shared_text,
            "claims": [{"id": "c1", "text": shared_text, "evidence_ids": ["ev_2"]}],
            "evidence_spans": [valid_span_doc2],
            "source_units": [merged_unit],
        }
    )
    assert out_ok["evidence_status"] == "FOUND"
    assert out_ok["verified_claims"][0]["provenance_ok"] is True

    # Span with tampered context_hash must FAIL provenance_ok and downgrade support -> not_enough_info
    tampered_span = dict(valid_span_doc2, context_hash="0000deadbeef0000")
    out_bad = fc.run(
        {
            "query": "LoRA reduction?",
            "answer": shared_text,
            "claims": [{"id": "c1", "text": shared_text, "evidence_ids": ["ev_2"]}],
            "evidence_spans": [tampered_span],
            "source_units": [merged_unit],
        }
    )
    assert out_bad["evidence_status"] == "NOT_FOUND"
    assert out_bad["rejected_claims"][0]["provenance_ok"] is False
