"""
Parameterized Multi-Mode Regression Suite (Đợt 6).

Covers all 6 official execution modes:
  EXECUTION_MODES = ["cpu_min", "cpu_max", "gpu_finetune", "gpu_max", "hybrid", "gemini"]

Verifies:
  1. Critic fail-closed across all 6 modes (exception, malformed JSON, invalid schema).
  2. Fact-Checker behavior across support, contradict, and not_enough_info across all 6 modes.
  3. Numerical and unit-aware verification (85% vs 85, km vs m, 2025 vs 2024) across all 6 modes.
  4. Citation provenance integrity (tampered context_hash, wrong page, wrong doc) across all 6 modes.
  5. Context packing coverage preservation across all 6 modes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from core.model_gateway import OFFICIAL_EXECUTION_MODES
from core.state import SourceUnit, compute_context_hash
from modes.registry import get_agent_class
from utils.llm_client import UnifiedLLMClient

EXECUTION_MODES = ["cpu_min", "cpu_max", "gpu_finetune", "gpu_max", "hybrid", "gemini"]
assert set(EXECUTION_MODES) == OFFICIAL_EXECUTION_MODES


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_critic_exception_fails_closed_across_all_modes(mode: str):
    """When Critic encounters an exception, every mode's SummarizerAgent must fail-closed."""
    summarizer = get_agent_class("summarizer", mode)()

    u = SourceUnit(
        id="u1",
        doc_id="doc1",
        page_no=1,
        text="Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
    ).to_dict()

    state = {
        "query": "Độ chính xác của mô hình là bao nhiêu?",
        "source_units": [u],
        "retrieved_docs": [u],
        "generator_model_family": mode,
    }

    with patch.object(
        UnifiedLLMClient,
        "generate",
        return_value="Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
    ), patch.object(
        UnifiedLLMClient,
        "call_kaggle_specialist",
        return_value={
            "answer": "Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
            "claims": [{"id": "c1", "text": "Mô hình đạt độ chính xác 94.2%"}],
            "critic": {"error": "LLM critic exception: timeout"},
        },
    ), patch.object(
        UnifiedLLMClient,
        "call_gpu_max_32b",
        return_value={
            "parsed": {
                "answer": "Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
                "claims": [{"id": "c1", "text": "Mô hình đạt độ chính xác 94.2%"}],
                "error": "LLM critic exception: timeout",
            }
        },
    ), patch.object(
        UnifiedLLMClient,
        "_call_gemini",
        return_value='{"answer": "Mô hình đạt 94.2%", "error": "timeout"}',
    ), patch.object(
        UnifiedLLMClient,
        "generate_json",
        side_effect=RuntimeError("LLM critic exception: timeout"),
    ), patch(
        "core.verifier.evaluate_inner_critic_fast",
        side_effect=RuntimeError("Fast critic timeout"),
    ):
        out = summarizer.run(state)

        assert out["processing_status"] == "VERIFICATION_FAILED", (
            f"Mode {mode} must return VERIFICATION_FAILED on critic exception, got {out.get('processing_status')}"
        )
        assert out.get("critic_action") == "verification_failed", (
            f"Mode {mode} critic_action must be 'verification_failed', got {out.get('critic_action')}"
        )
        assert out.get("confidence") == "low", (
            f"Mode {mode} confidence must be 'low', got {out.get('confidence')}"
        )
        # In fail-closed state, no claims are approved
        assert out.get("claims") == [], (
            f"Mode {mode} must clear approved claims on verification failure, got {out.get('claims')}"
        )


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_critic_malformed_json_fails_closed_across_all_modes(mode: str):
    """When Critic returns malformed/schema-violating JSON, every mode must fail-closed."""
    summarizer = get_agent_class("summarizer", mode)()

    u = SourceUnit(
        id="u1",
        doc_id="doc1",
        page_no=1,
        text="Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
    ).to_dict()

    state = {
        "query": "Độ chính xác của mô hình?",
        "source_units": [u],
        "retrieved_docs": [u],
    }

    with patch.object(
        UnifiedLLMClient,
        "generate",
        return_value="Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
    ), patch.object(
        UnifiedLLMClient,
        "call_kaggle_specialist",
        return_value={
            "answer": "Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
            "claims": [{"id": "c1", "text": "Mô hình đạt độ chính xác 94.2%"}],
            "critic": {"invalid_key": True},
        },
    ), patch.object(
        UnifiedLLMClient,
        "call_gpu_max_32b",
        return_value={
            "parsed": {
                "answer": "Mô hình đạt độ chính xác 94.2% trên tập kiểm thử năm 2024.",
                "claims": [{"id": "c1", "text": "Mô hình đạt độ chính xác 94.2%"}],
                "error": "malformed_json",
            }
        },
    ), patch.object(
        UnifiedLLMClient,
        "_call_gemini",
        return_value='{"corrupt": true}',
    ), patch.object(
        UnifiedLLMClient,
        "generate_json",
        return_value={"error": "Failed to parse JSON response", "raw_text": "not json"},
    ), patch(
        "core.verifier.evaluate_inner_critic_fast",
        return_value=MagicMock(action="invalid_action_unknown"),
    ):
        out = summarizer.run(state)
        assert out["processing_status"] == "VERIFICATION_FAILED"
        assert out.get("critic_action") == "verification_failed"
        assert out.get("confidence") == "low"
        assert out.get("claims") == []


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_fact_checker_supported_claim_across_all_modes(mode: str):
    """When a claim is fully supported by source units and valid citations, all 6 modes agree."""
    fc = get_agent_class("factchecker", mode)()

    quote = "Mô hình RoBERTa đạt độ chính xác 92.5% trên tập dữ liệu kiểm thử năm 2024."
    doc_id = "paper_2024"
    page_no = 2
    c_hash = compute_context_hash(quote, doc_id, page_no)

    u = SourceUnit(
        id="u1",
        doc_id=doc_id,
        page_no=page_no,
        text=quote,
    ).to_dict()

    span = {
        "id": "ev_1",
        "quote": quote,
        "source_unit_id": "u1",
        "doc_id": doc_id,
        "page_no": page_no,
        "context_hash": c_hash,
    }

    state = {
        "query": "Độ chính xác của RoBERTa năm 2024?",
        "answer": quote,
        "claims": [{"id": "c1", "text": quote, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
    }

    out = fc.run(state)

    assert out["evidence_status"] == "FOUND", (
        f"Mode {mode} must return evidence_status='FOUND', got {out.get('evidence_status')}"
    )
    assert out["fact_check_label"] == "support", (
        f"Mode {mode} must return fact_check_label='support', got {out.get('fact_check_label')}"
    )
    assert len(out.get("verified_claims", [])) == 1, (
        f"Mode {mode} verified_claims must have 1 claim, got {len(out.get('verified_claims', []))}"
    )
    assert len(out.get("rejected_claims", [])) == 0, (
        f"Mode {mode} rejected_claims must be empty, got {out.get('rejected_claims')}"
    )

    vc = out["verified_claims"][0]
    assert vc.get("numerical_passed") is True, f"Mode {mode} numerical_passed must be True"
    assert vc.get("dates_passed") is True, f"Mode {mode} dates_passed must be True"
    assert vc.get("provenance_ok") is True, f"Mode {mode} provenance_ok must be True"


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_fact_checker_contradicting_claim_across_all_modes(mode: str):
    """When a claim contradicts source text (direct negation/polarity clash), all 6 modes classify conflict."""
    fc = get_agent_class("factchecker", mode)()

    source_text = "Mô hình Transformer hoàn toàn có thể áp dụng cho phân loại văn bản tiếng Việt."
    doc_id = "paper_2024"
    page_no = 1
    c_hash = compute_context_hash(source_text, doc_id, page_no)

    u = SourceUnit(
        id="u1",
        doc_id=doc_id,
        page_no=page_no,
        text=source_text,
    ).to_dict()

    span = {
        "id": "ev_1",
        "quote": source_text,
        "source_unit_id": "u1",
        "doc_id": doc_id,
        "page_no": page_no,
        "context_hash": c_hash,
    }

    # Direct polarity contradiction: claims "không thể áp dụng" while source says "hoàn toàn có thể áp dụng"
    contra_claim = "Mô hình Transformer không thể áp dụng cho phân loại văn bản tiếng Việt."

    state = {
        "query": "Transformer có áp dụng được cho tiếng Việt không?",
        "answer": contra_claim,
        "claims": [{"id": "c1", "text": contra_claim, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
    }

    out = fc.run(state)

    assert out["evidence_status"] == "CONFLICTING_EVIDENCE", (
        f"Mode {mode} must return evidence_status='CONFLICTING_EVIDENCE', got {out.get('evidence_status')}"
    )
    assert out["fact_check_label"] == "contradict", (
        f"Mode {mode} must return fact_check_label='contradict', got {out.get('fact_check_label')}"
    )
    assert len(out.get("rejected_claims", [])) >= 1, (
        f"Mode {mode} must reject contradictory claim"
    )
    assert len(out.get("verified_claims", [])) == 0, (
        f"Mode {mode} must not verify contradictory claim"
    )


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_fact_checker_missing_evidence_across_all_modes(mode: str):
    """When a claim has zero supporting evidence in source units, no mode auto-approves."""
    fc = get_agent_class("factchecker", mode)()

    unrelated_text = "Nghiên cứu về quang hợp ở thực vật bậc cao và cơ chế hấp thụ photon ánh sáng."
    doc_id = "biology_doc"
    page_no = 1
    c_hash = compute_context_hash(unrelated_text, doc_id, page_no)

    u = SourceUnit(
        id="u1",
        doc_id=doc_id,
        page_no=page_no,
        text=unrelated_text,
    ).to_dict()

    span = {
        "id": "ev_1",
        "quote": unrelated_text,
        "source_unit_id": "u1",
        "doc_id": doc_id,
        "page_no": page_no,
        "context_hash": c_hash,
    }

    unsupported_claim = "Thuật toán AdamW giúp giảm 35% dung lượng VRAM khi huấn luyện LLM."

    state = {
        "query": "AdamW giảm bao nhiêu phần trăm VRAM?",
        "answer": unsupported_claim,
        "claims": [{"id": "c1", "text": unsupported_claim, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
    }

    out = fc.run(state)

    assert out["evidence_status"] in ("NOT_FOUND", "PARTIAL"), (
        f"Mode {mode} must classify unsupported claim as NOT_FOUND or PARTIAL, got {out.get('evidence_status')}"
    )
    assert out["fact_check_label"] in ("not_enough_info", "contradict"), (
        f"Mode {mode} fact_check_label must be not_enough_info or contradict, got {out.get('fact_check_label')}"
    )
    assert len(out.get("verified_claims", [])) == 0, (
        f"Mode {mode} must have 0 verified claims for unsupported claim"
    )


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_numerical_unit_sensitivity_across_all_modes(mode: str):
    """Numerical verifier must reject unit mismatches (85% vs 85, km vs m, 2025 vs 2024) across all modes."""
    fc = get_agent_class("factchecker", mode)()

    # Case 1: 85% in claim vs 85 bare count in document
    u_pct = SourceUnit(
        id="u1",
        doc_id="d1",
        page_no=1,
        text="Hệ thống ghi nhận 85 mẫu thử nghiệm trong năm 2024.",
    ).to_dict()
    span_pct = {
        "id": "ev_1",
        "quote": "Hệ thống ghi nhận 85 mẫu thử nghiệm trong năm 2024.",
        "source_unit_id": "u1",
        "doc_id": "d1",
        "page_no": 1,
        "context_hash": compute_context_hash("Hệ thống ghi nhận 85 mẫu thử nghiệm trong năm 2024.", "d1", 1),
    }

    out_pct = fc.run({
        "query": "Tỷ lệ là bao nhiêu?",
        "answer": "Hệ thống đạt tỷ lệ 85% mẫu thử nghiệm trong năm 2024.",
        "claims": [{"id": "c1", "text": "Hệ thống đạt tỷ lệ 85% mẫu thử nghiệm trong năm 2024.", "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span_pct],
        "source_units": [u_pct],
    })

    if out_pct.get("verified_claims"):
        assert out_pct["verified_claims"][0].get("numerical_passed") is False or out_pct["evidence_status"] != "FOUND"
    else:
        assert len(out_pct.get("rejected_claims", [])) >= 1

    # Case 2: 85 km vs 85 m
    u_dist = SourceUnit(
        id="u2",
        doc_id="d2",
        page_no=1,
        text="Khoảng cách truyền dữ liệu tối đa là 85 m trong phòng thí nghiệm.",
    ).to_dict()
    span_dist = {
        "id": "ev_2",
        "quote": "Khoảng cách truyền dữ liệu tối đa là 85 m trong phòng thí nghiệm.",
        "source_unit_id": "u2",
        "doc_id": "d2",
        "page_no": 1,
        "context_hash": compute_context_hash("Khoảng cách truyền dữ liệu tối đa là 85 m trong phòng thí nghiệm.", "d2", 1),
    }

    out_dist = fc.run({
        "query": "Khoảng cách truyền là bao nhiêu?",
        "answer": "Khoảng cách truyền dữ liệu tối đa là 85 km trong phòng thí nghiệm.",
        "claims": [{"id": "c2", "text": "Khoảng cách truyền dữ liệu tối đa là 85 km trong phòng thí nghiệm.", "evidence_ids": ["ev_2"]}],
        "evidence_spans": [span_dist],
        "source_units": [u_dist],
    })
    if out_dist.get("verified_claims"):
        assert out_dist["verified_claims"][0].get("numerical_passed") is False or out_dist["evidence_status"] != "FOUND"
    else:
        assert len(out_dist.get("rejected_claims", [])) >= 1

    # Case 3: 2025 in claim vs 2024 in document
    u_yr = SourceUnit(
        id="u3",
        doc_id="d3",
        page_no=1,
        text="Hội thảo được tổ chức thành công vào năm 2024 tại Hà Nội.",
    ).to_dict()
    span_yr = {
        "id": "ev_3",
        "quote": "Hội thảo được tổ chức thành công vào năm 2024 tại Hà Nội.",
        "source_unit_id": "u3",
        "doc_id": "d3",
        "page_no": 1,
        "context_hash": compute_context_hash("Hội thảo được tổ chức thành công vào năm 2024 tại Hà Nội.", "d3", 1),
    }

    out_yr = fc.run({
        "query": "Hội thảo tổ chức năm nào?",
        "answer": "Hội thảo được tổ chức thành công vào năm 2025 tại Hà Nội.",
        "claims": [{"id": "c3", "text": "Hội thảo được tổ chức thành công vào năm 2025 tại Hà Nội.", "evidence_ids": ["ev_3"]}],
        "evidence_spans": [span_yr],
        "source_units": [u_yr],
    })
    if out_yr.get("verified_claims"):
        assert out_yr["verified_claims"][0].get("dates_passed") is False or out_yr["evidence_status"] != "FOUND"
    else:
        assert len(out_yr.get("rejected_claims", [])) >= 1


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_citation_provenance_tampered_and_wrong_page_across_all_modes(mode: str):
    """Citation provenance verifier detects tampered hashes and wrong page numbers across all modes."""
    fc = get_agent_class("factchecker", mode)()

    doc_text = "Hệ thống AI đạt điểm benchmark 89.1 vào tháng 5 năm 2024."
    u = SourceUnit(id="u1", doc_id="paper_1", page_no=3, text=doc_text).to_dict()

    # 1. Tampered context hash
    tampered_span = {
        "id": "ev_1",
        "quote": doc_text,
        "source_unit_id": "u1",
        "doc_id": "paper_1",
        "page_no": 3,
        "context_hash": "tampered_hash_000000000000",
    }

    out_tampered = fc.run({
        "query": "Điểm benchmark là bao nhiêu?",
        "answer": doc_text,
        "claims": [{"id": "c1", "text": doc_text, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [tampered_span],
        "source_units": [u],
    })

    if out_tampered.get("verified_claims"):
        assert out_tampered["verified_claims"][0].get("provenance_ok") is False
    else:
        assert len(out_tampered.get("rejected_claims", [])) >= 1

    # 2. Wrong page number (evidence span claims page 99, but unit is at page 3)
    wrong_page_span = {
        "id": "ev_2",
        "quote": doc_text,
        "source_unit_id": "u1",
        "doc_id": "paper_1",
        "page_no": 99,  # Mismatch with unit page 3
        "context_hash": compute_context_hash(doc_text, "paper_1", 99),
    }

    out_wrong_page = fc.run({
        "query": "Điểm benchmark là bao nhiêu?",
        "answer": doc_text,
        "claims": [{"id": "c2", "text": doc_text, "evidence_ids": ["ev_2"]}],
        "evidence_spans": [wrong_page_span],
        "source_units": [u],
    })

    if out_wrong_page.get("verified_claims"):
        assert out_wrong_page["verified_claims"][0].get("provenance_ok") is False
    else:
        assert len(out_wrong_page.get("rejected_claims", [])) >= 1


@pytest.mark.parametrize("mode", EXECUTION_MODES)
def test_context_packing_coverage_preservation_across_all_modes(mode: str):
    """FactChecker must preserve context_coverage and not downgrade to PARTIAL when all claims are supported."""
    fc = get_agent_class("factchecker", mode)()

    quote = "Mô hình BERT đạt F1-score 91.2% trên bài toán SQuAD năm 2024."
    doc_id = "bert_paper"
    page_no = 5
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

    coverage_input = {
        "total_units": 60,
        "included_units": 12,
        "omitted_units": 48,
        "coverage_pct": 20.0,
        "truncated": True,
    }

    state = {
        "query": "F1-score của BERT?",
        "answer": quote,
        "claims": [{"id": "c1", "text": quote, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
        "context_coverage": coverage_input,
    }

    out = fc.run(state)

    # Coverage metadata must be preserved
    assert "context_coverage" in out
    cov = out["context_coverage"]
    assert cov.get("total_units") == 60
    assert cov.get("included_units") == 12
    assert cov.get("truncated") is True

    # Truncation alone must not downgrade FOUND to PARTIAL when all claims are supported
    assert out["evidence_status"] == "FOUND", (
        f"Mode {mode} should not downgrade to PARTIAL on truncation when all claims are supported; got {out['evidence_status']}"
    )
    assert out["fact_check_label"] == "support"
