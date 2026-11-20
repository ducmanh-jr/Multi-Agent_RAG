"""
Cross-Mode Evidence Consistency Suite (Đợt 6 — Criterion E).

Verifies that all 6 official execution modes:
  ["cpu_min", "cpu_max", "gpu_finetune", "gpu_max", "hybrid", "gemini"]
produce strictly consistent EvidenceStatus and FactCheckLabel decisions
when provided with the exact same inputs and source units.
"""

from __future__ import annotations

import pytest
from core.model_gateway import OFFICIAL_EXECUTION_MODES
from core.state import SourceUnit, compute_context_hash
from modes.registry import get_agent_class

EXECUTION_MODES = ["cpu_min", "cpu_max", "gpu_finetune", "gpu_max", "hybrid", "gemini"]
assert set(EXECUTION_MODES) == OFFICIAL_EXECUTION_MODES


def test_cross_mode_supported_claim_identical_verdict():
    """All 6 modes must evaluate a fully grounded claim to evidence_status='FOUND' and label='support'."""
    quote = "Mô hình DeBERTa v3 cải thiện đáng kể độ chính xác phân loại ngữ nghĩa so với RoBERTa."
    doc_id = "deberta_paper"
    page_no = 4
    c_hash = compute_context_hash(quote, doc_id, page_no)

    u = SourceUnit(id="u1", doc_id=doc_id, page_no=page_no, text=quote).to_dict()
    span = {
        "id": "ev_1",
        "quote": quote,
        "source_unit_id": "u1",
        "doc_id": doc_id,
        "page_no": page_no,
        "context_hash": c_hash,
    }

    state = {
        "query": "DeBERTa v3 có cải thiện độ chính xác không?",
        "answer": quote,
        "claims": [{"id": "c1", "text": quote, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
    }

    results: dict[str, dict] = {}
    for mode in EXECUTION_MODES:
        fc = get_agent_class("factchecker", mode)()
        results[mode] = fc.run(dict(state))

    # Assert invariant across all 6 modes
    for mode, out in results.items():
        assert out["evidence_status"] == "FOUND", (
            f"Mode {mode} deviated: expected evidence_status='FOUND', got '{out.get('evidence_status')}'"
        )
        assert out["fact_check_label"] == "support", (
            f"Mode {mode} deviated: expected fact_check_label='support', got '{out.get('fact_check_label')}'"
        )
        assert len(out.get("verified_claims", [])) == 1, (
            f"Mode {mode} must verify exactly 1 claim"
        )
        assert len(out.get("rejected_claims", [])) == 0, (
            f"Mode {mode} must have 0 rejected claims"
        )


def test_cross_mode_contradicting_claim_identical_verdict():
    """All 6 modes must evaluate a contradictory claim to CONFLICTING_EVIDENCE and 'contradict'."""
    source_text = "Mô hình Transformer hoàn toàn có thể áp dụng cho phân loại văn bản tiếng Việt."
    doc_id = "vn_nlp_paper"
    page_no = 2
    c_hash = compute_context_hash(source_text, doc_id, page_no)

    u = SourceUnit(id="u1", doc_id=doc_id, page_no=page_no, text=source_text).to_dict()
    span = {
        "id": "ev_1",
        "quote": source_text,
        "source_unit_id": "u1",
        "doc_id": doc_id,
        "page_no": page_no,
        "context_hash": c_hash,
    }

    contra_claim = "Mô hình Transformer không thể áp dụng cho phân loại văn bản tiếng Việt."

    state = {
        "query": "Transformer có dùng được cho tiếng Việt?",
        "answer": contra_claim,
        "claims": [{"id": "c1", "text": contra_claim, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
    }

    results: dict[str, dict] = {}
    for mode in EXECUTION_MODES:
        fc = get_agent_class("factchecker", mode)()
        results[mode] = fc.run(dict(state))

    for mode, out in results.items():
        assert out["evidence_status"] == "CONFLICTING_EVIDENCE", (
            f"Mode {mode} deviated: expected evidence_status='CONFLICTING_EVIDENCE', got '{out.get('evidence_status')}'"
        )
        assert out["fact_check_label"] == "contradict", (
            f"Mode {mode} deviated: expected fact_check_label='contradict', got '{out.get('fact_check_label')}'"
        )
        assert len(out.get("rejected_claims", [])) >= 1, (
            f"Mode {mode} must reject contradictory claim"
        )
        assert len(out.get("verified_claims", [])) == 0, (
            f"Mode {mode} must not verify contradictory claim"
        )


def test_cross_mode_missing_evidence_identical_rejection():
    """All 6 modes must refuse to verify claims that lack supporting evidence."""
    unrelated_text = "Khí quyển Trái Đất chủ yếu gồm nitơ và oxy với một lượng nhỏ agon."
    doc_id = "earth_doc"
    page_no = 1
    c_hash = compute_context_hash(unrelated_text, doc_id, page_no)

    u = SourceUnit(id="u1", doc_id=doc_id, page_no=page_no, text=unrelated_text).to_dict()
    span = {
        "id": "ev_1",
        "quote": unrelated_text,
        "source_unit_id": "u1",
        "doc_id": doc_id,
        "page_no": page_no,
        "context_hash": c_hash,
    }

    unsupported_claim = "Thuật toán FlashAttention-2 giảm 50% thời gian tính toán attention matrix."

    state = {
        "query": "FlashAttention-2 giảm bao nhiêu thời gian?",
        "answer": unsupported_claim,
        "claims": [{"id": "c1", "text": unsupported_claim, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
    }

    results: dict[str, dict] = {}
    for mode in EXECUTION_MODES:
        fc = get_agent_class("factchecker", mode)()
        results[mode] = fc.run(dict(state))

    for mode, out in results.items():
        assert out["evidence_status"] in ("NOT_FOUND", "PARTIAL"), (
            f"Mode {mode} deviated: expected NOT_FOUND or PARTIAL, got '{out.get('evidence_status')}'"
        )
        assert out["fact_check_label"] in ("not_enough_info", "contradict"), (
            f"Mode {mode} must not return support for missing evidence, got '{out.get('fact_check_label')}'"
        )
        assert len(out.get("verified_claims", [])) == 0, (
            f"Mode {mode} must not verify unsupported claim"
        )


def test_cross_mode_truncated_context_preserves_found_across_all_modes():
    """Truncated context coverage must NOT cause any mode to falsely downgrade from FOUND to PARTIAL."""
    quote = "Phương pháp LoRA chỉ huấn luyện 0.1% tổng số tham số của mô hình gốc."
    doc_id = "lora_doc"
    page_no = 3
    c_hash = compute_context_hash(quote, doc_id, page_no)

    u = SourceUnit(id="u1", doc_id=doc_id, page_no=page_no, text=quote).to_dict()
    span = {
        "id": "ev_1",
        "quote": quote,
        "source_unit_id": "u1",
        "doc_id": doc_id,
        "page_no": page_no,
        "context_hash": c_hash,
    }

    coverage = {
        "total_units": 100,
        "included_units": 10,
        "omitted_units": 90,
        "coverage_pct": 10.0,
        "truncated": True,
    }

    state = {
        "query": "LoRA huấn luyện bao nhiêu phần trăm tham số?",
        "answer": quote,
        "claims": [{"id": "c1", "text": quote, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
        "context_coverage": coverage,
    }

    results: dict[str, dict] = {}
    for mode in EXECUTION_MODES:
        fc = get_agent_class("factchecker", mode)()
        results[mode] = fc.run(dict(state))

    for mode, out in results.items():
        assert out["evidence_status"] == "FOUND", (
            f"Mode {mode} falsely downgraded evidence_status to '{out.get('evidence_status')}' under truncated context"
        )
        assert out["fact_check_label"] == "support"
        assert out.get("context_coverage", {}).get("truncated") is True
