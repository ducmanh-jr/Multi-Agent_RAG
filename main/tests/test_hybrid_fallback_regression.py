"""
Hybrid Mode Cascading & Strict Zero-Gemini Regression Suite (Đợt 6 — Criterion G).

Verifies:
  1. Priority cascading order: GPU Finetune -> GPU Max -> Local CPU.
  2. Fail-Closed Preservation: When primary backend yields VERIFICATION_FAILED,
     HybridSummarizerAgent STOPS immediately and does NOT cascade to paper over the failure.
  3. Absolute Gemini Exclusion: Under total cascading failure across all backends,
     Gemini is NEVER called (0 calls, zero fallback).
  4. Hybrid Fact Checker cascading across backends to deterministic local verification.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
import pytest

from core.model_gateway import model_gateway, ProviderPolicyError
from core.state import SourceUnit, compute_context_hash
from modes.hybrid.summarizer import HybridSummarizerAgent
from modes.hybrid.fact_checker import HybridFactCheckerAgent
from modes.hybrid.retrieval import HybridRetrievalAgent
from modes.hybrid.writer import HybridWriterAgent
from utils.llm_client import UnifiedLLMClient


def test_hybrid_backend_selection_priority_order():
    """Probe status dictates backend cascading order: gpu_finetune -> gpu_max -> cpu_max."""
    # 1. Both online -> priority 1 is gpu_finetune
    caps_both_online = {
        "gpu_finetune": {"online": True},
        "kaggle_max": {"online": True, "has_vllm_32b": True},
    }
    chain_both = model_gateway.get_hybrid_backend_chain(caps_both_online, stage="summarizer")
    assert chain_both == ["gpu_finetune", "gpu_max", "cpu_max"]

    # 2. gpu_finetune offline, gpu_max online -> priority is gpu_max
    caps_gpu_max_only = {
        "gpu_finetune": {"online": False},
        "kaggle_max": {"online": True, "has_vllm_32b": True},
    }
    chain_max = model_gateway.get_hybrid_backend_chain(caps_gpu_max_only, stage="summarizer")
    assert chain_max == ["gpu_max", "cpu_max"]

    # 3. Both GPUs offline -> priority is cpu_max
    caps_gpus_offline = {
        "gpu_finetune": {"online": False},
        "kaggle_max": {"online": False, "has_vllm_32b": False},
    }
    chain_cpu = model_gateway.get_hybrid_backend_chain(caps_gpus_offline, stage="summarizer")
    assert chain_cpu == ["cpu_max"]

    # Invariant: Gemini is NEVER in any hybrid chain
    for chain in (chain_both, chain_max, chain_cpu):
        assert "gemini" not in chain


def test_hybrid_summarizer_preserves_verification_failed_and_never_overrides():
    """When primary backend fails Critic verification, Hybrid must NOT cascade to paper over it."""
    agent = HybridSummarizerAgent()

    # Mock probe: both GPUs are reported online
    caps_online = {
        "gpu_finetune": {"online": True},
        "kaggle_max": {"online": True, "has_vllm_32b": True},
    }

    fail_state = {
        "answer": "Bản thảo chưa được duyệt",
        "claims": [],
        "critic_action": "verification_failed",
        "processing_status": "VERIFICATION_FAILED",
        "confidence": "low",
        "reasoning_trace": [{"node": "GPUFinetuneSummarizer", "action": "critic_failed"}],
    }

    # gpu_finetune returns VERIFICATION_FAILED
    agent._gpu_finetune.run = MagicMock(return_value=fail_state)
    agent._gpu_max.run = MagicMock(return_value={"answer": "Override answer", "processing_status": "SUCCESS"})
    agent._cpu_max.run = MagicMock(return_value={"answer": "CPU answer", "processing_status": "SUCCESS"})

    u = SourceUnit(id="u1", doc_id="d1", page_no=1, text="Sample text.").to_dict()
    state = {"query": "Test query", "source_units": [u], "complexity": "MEDIUM"}

    with patch("modes.hybrid.summarizer.probe_all_capabilities", return_value=caps_online):
        out = agent.run(state)

    # Must preserve VERIFICATION_FAILED
    assert out["processing_status"] == "VERIFICATION_FAILED"
    assert out["critic_action"] == "verification_failed"
    assert out.get("confidence") == "low"
    assert out.get("claims") == []

    # Must NOT have called gpu_max or cpu_max to bypass verification failure
    agent._gpu_max.run.assert_not_called()
    agent._cpu_max.run.assert_not_called()


def test_hybrid_agents_zero_gemini_under_total_backend_failure():
    """Under catastrophic failure across all backends, Hybrid NEVER invokes Gemini API."""
    caps_offline = {
        "gpu_finetune": {"online": False},
        "kaggle_max": {"online": False},
    }

    u = SourceUnit(id="u1", doc_id="d1", page_no=1, text="Text").to_dict()
    state = {
        "query": "Total failure test query",
        "source_units": [u],
        "retrieved_docs": [u],
        "complexity": "HIGH",
    }

    summarizer = HybridSummarizerAgent()
    summarizer._gpu_finetune.run = MagicMock(side_effect=RuntimeError("GPU FT offline"))
    summarizer._gpu_max.run = MagicMock(side_effect=RuntimeError("GPU Max offline"))
    # Even if cpu_max summarizer returns an empty answer
    summarizer._cpu_max.run = MagicMock(return_value={"answer": "", "claims": []})

    fact_checker = HybridFactCheckerAgent()
    fact_checker._gpu_finetune_fc.run = MagicMock(side_effect=RuntimeError("GPU FT offline"))
    fact_checker._gpu_max_fc.run = MagicMock(side_effect=RuntimeError("GPU Max offline"))

    retrieval = HybridRetrievalAgent()
    writer = HybridWriterAgent()

    with patch("modes.hybrid.summarizer.probe_all_capabilities", return_value=caps_offline), \
         patch("modes.hybrid.fact_checker.probe_all_capabilities", return_value=caps_offline), \
         patch("modes.hybrid.retrieval.probe_all_capabilities", return_value=caps_offline), \
         patch("modes.hybrid.writer.probe_all_capabilities", return_value=caps_offline), \
         patch.object(UnifiedLLMClient, "_call_gemini") as mock_gemini:

        # 1. Summarizer run
        summarizer.run(state)
        # 2. FactChecker run
        fc_state = dict(state)
        fc_state.update({"answer": "Answer", "claims": [{"id": "c1", "text": "Claim"}]})
        fact_checker.run(fc_state)
        # 3. Retrieval run
        retrieval.run(state)
        # 4. Writer run
        writer_state = dict(state)
        writer_state.update({
            "answer": "Answer",
            "verified_claims": [{"claim": "Claim", "text": "Claim"}],
            "evidence_status": "FOUND",
        })
        writer.run(writer_state)

        # Gemini API call count must remain strictly ZERO
        assert mock_gemini.call_count == 0, (
            f"Expected ZERO Gemini calls in hybrid mode, got {mock_gemini.call_count}"
        )


def test_hybrid_fact_checker_cascading_to_local_verifier():
    """HybridFactChecker cascades cleanly when remote verifiers raise, returning valid contract."""
    caps_online = {
        "gpu_finetune": {"online": True},
        "kaggle_max": {"online": True, "has_vllm_32b": True},
    }

    fc = HybridFactCheckerAgent()

    # Remote backends raise exceptions
    fc._gpu_finetune_fc.run = MagicMock(side_effect=RuntimeError("Kaggle connection reset"))
    fc._gpu_max_fc.run = MagicMock(side_effect=RuntimeError("GPU Max 32B timeout"))

    quote = "Mô hình BERT hoàn toàn có thể áp dụng cho phân loại văn bản tiếng Việt."
    c_hash = compute_context_hash(quote, "paper_1", 1)
    u = SourceUnit(id="u1", doc_id="paper_1", page_no=1, text=quote).to_dict()
    span = {
        "id": "ev_1",
        "quote": quote,
        "source_unit_id": "u1",
        "doc_id": "paper_1",
        "page_no": 1,
        "context_hash": c_hash,
    }

    state = {
        "query": "BERT có dùng được cho tiếng Việt?",
        "answer": quote,
        "claims": [{"id": "c1", "text": quote, "evidence_ids": ["ev_1"]}],
        "evidence_spans": [span],
        "source_units": [u],
    }

    with patch("modes.hybrid.fact_checker.probe_all_capabilities", return_value=caps_online):
        out = fc.run(state)

    # Local fallback succeeded with valid contract
    assert out["evidence_status"] == "FOUND"
    assert out["fact_check_label"] == "support"
    assert len(out["verified_claims"]) == 1
    assert any("Đối soát sự thật" in t.get("action", "") for t in out.get("reasoning_trace", []))
