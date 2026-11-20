"""
Automated Provider Policy & Model Gateway Verification Suite (Đợt 1).

Verifies AGENT.md Section 16 & Section 5 (3-Layer Compliance):
  1. Official execution modes are strictly: cpu_min, cpu_max, gpu_finetune, gpu_max, hybrid, gemini.
  2. Mode 'gpu_finetune' works cleanly; legacy mode 'gpu_ft' is explicitly rejected everywhere.
  3. GEMINI_ALLOWED_MODES is strictly frozenset({"gemini"}).
  4. All 5 non-Gemini official modes (cpu_min, cpu_max, gpu_finetune, gpu_max, hybrid)
     and legacy aliases (api, local_api) are blocked from calling Gemini.
  5. Mode 'gemini' without GEMINI_API_KEY raises ProviderConfigurationError clearly.
  6. Hybrid mode follows GPU Finetune (gpu_finetune) -> GPU Max -> Local CPU and never
     instantiates or falls back to Gemini*Agent.
  7. Agent 0 Scout uses OpenAlex (with backward-compatible alias) and handles
     uppercase REJECT/CLARIFY decisions properly.
"""

from __future__ import annotations

import inspect
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from core.model_gateway import (
    GEMINI_ALLOWED_MODES,
    OFFICIAL_EXECUTION_MODES,
    ProviderConfigurationError,
    ProviderPolicyError,
    model_gateway,
)
from utils.llm_client import UnifiedLLMClient


FORBIDDEN_MODES = [
    "cpu_min",
    "cpu_max",
    "gpu_finetune",
    "gpu_max",
    "hybrid",
]


def test_official_execution_modes_and_gpu_ft_rejected():
    from fastapi import HTTPException
    from modes.registry import get_agent_class, get_mode_agents, normalize_mode

    assert OFFICIAL_EXECUTION_MODES == frozenset(
        {"cpu_min", "cpu_max", "gpu_finetune", "gpu_max", "hybrid", "gemini"}
    )
    assert "gpu_ft" not in OFFICIAL_EXECUTION_MODES

    # gpu_finetune must be accepted and instantiate all 5 official GPUFinetune agents
    assert model_gateway.validate_execution_mode("gpu_finetune") == "gpu_finetune"
    assert normalize_mode("gpu_finetune") == "gpu_finetune"
    agents = get_mode_agents("gpu_finetune")
    assert set(agents.keys()) == {"retrieval", "reader", "summarizer", "fact_checker", "writer"}
    for ag in agents.values():
        assert "GPUFinetune" in type(ag).__name__

    # Legacy removed mode 'gpu_ft' must be rejected everywhere
    for removed in ("gpu_ft", "gpuft", "gpu_ft_new"):
        with pytest.raises(ValueError):
            model_gateway.validate_execution_mode(removed)
        with pytest.raises(ValueError):
            normalize_mode(removed)
        with pytest.raises(ValueError):
            get_agent_class("retrieval", removed)

    # API config endpoint must reject 'gpu_ft' with HTTP 400
    from api.app import SetModeRequest, set_execution_mode

    with pytest.raises(HTTPException) as exc_info:
        set_execution_mode(SetModeRequest(mode="gpu_ft"))
    assert exc_info.value.status_code == 400


def test_gemini_allowed_modes_constant_is_strict():
    assert GEMINI_ALLOWED_MODES == frozenset({"gemini"})
    assert model_gateway.is_gemini_allowed("gemini") is True
    assert model_gateway.is_gemini_allowed("GEMINI") is True

    for forbidden in FORBIDDEN_MODES + ["api", "local_api", "", None]:
        assert model_gateway.is_gemini_allowed(forbidden) is False
        with pytest.raises(ProviderPolicyError):
            model_gateway.assert_provider_allowed("gemini", forbidden, caller="test")


def test_guard_agent_instantiation_blocks_gemini_outside_gemini_mode():
    model_gateway.guard_agent_instantiation("GeminiSummarizerAgent", "gemini", "gemini")

    for forbidden in FORBIDDEN_MODES + ["api", "local_api"]:
        with pytest.raises(ProviderPolicyError):
            model_gateway.guard_agent_instantiation("GeminiSummarizerAgent", "gemini", forbidden)


def test_gemini_mode_missing_api_key_raises_configuration_error(monkeypatch):
    client = UnifiedLLMClient()
    monkeypatch.setattr(client.settings, "EXECUTION_MODE", "gemini")
    monkeypatch.setattr(client.settings, "GEMINI_API_KEY", "")

    with patch("utils.llm_client.get_cached_llm", return_value=None):
        with pytest.raises(ProviderConfigurationError):
            client.generate("Explain attention mechanism")

    with pytest.raises(ProviderConfigurationError):
        client._call_gemini("Explain attention mechanism", mode="gemini")


@pytest.mark.parametrize("mode", FORBIDDEN_MODES + ["api", "local_api"])
def test_llm_client_call_gemini_blocked_in_all_non_gemini_modes(monkeypatch, mode):
    client = UnifiedLLMClient()
    monkeypatch.setattr(client.settings, "EXECUTION_MODE", mode)
    monkeypatch.setattr(client.settings, "GEMINI_API_KEY", "test-dummy-key")

    with pytest.raises(ProviderPolicyError):
        client._call_gemini("Hello", mode=mode)


def test_llm_client_generate_gpu_max_routes_to_remote_not_gemini(monkeypatch):
    client = UnifiedLLMClient()
    monkeypatch.setattr(client.settings, "EXECUTION_MODE", "gpu_max")
    monkeypatch.setattr(client.settings, "GEMINI_API_KEY", "test-dummy-key")
    monkeypatch.setattr(client.settings, "REMOTE_INFERENCE_URL", "")

    with patch("utils.llm_client.get_cached_llm", return_value=None), patch.object(
        client, "_call_gemini", side_effect=AssertionError("Gemini must NEVER be called in gpu_max")
    ):
        with pytest.raises(RuntimeError, match="REMOTE_INFERENCE_URL"):
            client.generate("Test gpu_max routing")


def test_llm_client_generate_hybrid_never_calls_gemini(monkeypatch):
    client = UnifiedLLMClient()
    monkeypatch.setattr(client.settings, "EXECUTION_MODE", "hybrid")
    monkeypatch.setattr(client.settings, "GEMINI_API_KEY", "test-dummy-key")
    monkeypatch.setattr(client.settings, "REMOTE_INFERENCE_URL", "")

    with patch("utils.llm_client.get_cached_llm", return_value=None), patch(
        "utils.llm_client.set_cached_llm", return_value=None
    ), patch.object(
        client, "_call_local_cpu", side_effect=ConnectionError("Ollama offline")
    ), patch.object(
        client, "_call_gemini", side_effect=AssertionError("Gemini must NEVER be called in hybrid")
    ):
        res = client.generate("Test hybrid offline fallback", json_mode=True)
        assert isinstance(res, str) and len(res) > 0


def test_hybrid_backend_chain_priority_and_no_gemini():
    all_online_caps = {
        "gemini": {"online": True},
        "gpu_finetune": {"online": True},
        "kaggle_max": {"online": True, "has_vllm_32b": True},
        "ollama": {"online": True},
    }
    chain_sum = model_gateway.get_hybrid_backend_chain(all_online_caps, stage="summarizer")
    assert chain_sum == ["gpu_finetune", "gpu_max", "cpu_max"]
    assert "gemini" not in chain_sum

    chain_fc = model_gateway.get_hybrid_backend_chain(all_online_caps, stage="fact_checker")
    assert chain_fc == ["gpu_finetune", "gpu_max", "cpu_max"]
    assert "gemini" not in chain_fc

    gpu_finetune_offline_caps = {
        "gemini": {"online": True},
        "gpu_finetune": {"online": False},
        "kaggle_max": {"online": True, "has_vllm_32b": True},
    }
    assert model_gateway.select_hybrid_backend(gpu_finetune_offline_caps, stage="summarizer") == "gpu_max"

    all_gpu_offline_caps = {"gemini": {"online": True}}
    assert model_gateway.select_hybrid_backend(all_gpu_offline_caps, stage="summarizer") == "cpu_max"
    assert model_gateway.select_hybrid_backend(all_gpu_offline_caps, stage="retrieval") == "cpu_max"
    assert model_gateway.select_hybrid_backend(all_gpu_offline_caps, stage="reader") == "cpu_max"
    assert model_gateway.select_hybrid_backend(all_gpu_offline_caps, stage="writer") == "cpu_max"


def test_hybrid_agents_never_instantiate_or_import_gemini_agents():
    import modes.hybrid.fact_checker as h_fc
    import modes.hybrid.reader as h_rd
    import modes.hybrid.retrieval as h_ret
    import modes.hybrid.summarizer as h_sum
    import modes.hybrid.writer as h_wr

    modules = [h_ret, h_rd, h_sum, h_fc, h_wr]
    for mod in modules:
        for attr_name in dir(mod):
            assert not attr_name.startswith("Gemini"), (
                f"Module {mod.__name__} must not import Gemini symbol '{attr_name}'"
            )

    agents = [
        h_ret.HybridRetrievalAgent(),
        h_rd.HybridReaderAgent(),
        h_sum.HybridSummarizerAgent(),
        h_fc.HybridFactCheckerAgent(),
        h_wr.HybridWriterAgent(),
    ]
    for agent in agents:
        for attr_name, attr_val in vars(agent).items():
            cls_name = type(attr_val).__name__
            assert "Gemini" not in cls_name, (
                f"{agent.name}.{attr_name} instantiated forbidden Gemini class '{cls_name}'"
            )


def test_hybrid_agents_runtime_never_fall_back_to_gemini():
    from modes.hybrid.fact_checker import HybridFactCheckerAgent
    from modes.hybrid.reader import HybridReaderAgent
    from modes.hybrid.retrieval import HybridRetrievalAgent
    from modes.hybrid.summarizer import HybridSummarizerAgent
    from modes.hybrid.writer import HybridWriterAgent

    caps_with_gemini_only = {
        "gemini": {"online": True},
        "gpu_finetune": {"online": False},
        "kaggle_max": {"online": False, "has_vllm_32b": False},
    }
    sample_state: dict[str, Any] = {
        "query": "What is attention?",
        "retrieved_docs": [{"id": "d1", "doc_id": "d1", "page_no": 1, "text": "Attention maps queries and keys."}],
        "source_units": [{"id": "u1", "doc_id": "d1", "page_no": 1, "text": "Attention maps queries and keys."}],
        "answer": "Attention maps queries and keys [1].",
        "claims": [{"id": "c1", "text": "Attention maps queries and keys."}],
        "evidence_spans": [{"id": "ev1", "quote": "Attention maps queries and keys.", "doc_id": "d1", "page_no": 1}],
        "verified_claims": [{"claim": "Attention maps queries and keys.", "status": "SUPPORT"}],
        "rejected_claims": [],
    }

    with patch("utils.llm_client.llm_client._call_gemini", side_effect=AssertionError("Gemini called in Hybrid!")), \
         patch("modes.hybrid.retrieval.probe_all_capabilities", return_value=caps_with_gemini_only), \
         patch("modes.hybrid.reader.probe_all_capabilities", return_value=caps_with_gemini_only), \
         patch("modes.hybrid.summarizer.probe_all_capabilities", return_value=caps_with_gemini_only), \
         patch("modes.hybrid.fact_checker.probe_all_capabilities", return_value=caps_with_gemini_only), \
         patch("modes.hybrid.writer.probe_all_capabilities", return_value=caps_with_gemini_only):

        ret_out = HybridRetrievalAgent().run(sample_state)
        assert "retrieved_docs" in ret_out

        rd_out = HybridReaderAgent().run(sample_state)
        assert "source_units" in rd_out

        sum_agent = HybridSummarizerAgent()
        sum_agent._cpu_max.run = MagicMock(
            return_value={
                "answer": "Local CPU answer",
                "claims": [],
                "evidence_spans": [],
                "critic_action": "approve",
                "reasoning_trace": [],
            }
        )
        sum_out = sum_agent.run(sample_state)
        assert sum_out["answer"] == "Local CPU answer"

        fc_agent = HybridFactCheckerAgent()
        fc_agent._cpu_max_fc.run = MagicMock(
            return_value={
                "fact_check_label": "support",
                "verified_claims": [],
                "rejected_claims": [],
                "reasoning_trace": [],
            }
        )
        fc_out = fc_agent.run(sample_state)
        assert fc_out["fact_check_label"] == "support"

        wr_out = HybridWriterAgent().run(sample_state)
        assert wr_out["final_report"]


def test_openalex_scout_alias_and_case_insensitive_decision():
    from agents.agent0_scout.gemini_scout import search_gemini_academic_papers
    from agents.agent0_scout.node import ScoutAgent
    from agents.agent0_scout.openalex_scout import search_openalex_academic_papers

    # Backward-compatible alias points to OpenAlex implementation (zero Gemini)
    assert search_gemini_academic_papers is search_openalex_academic_papers
    src = inspect.getsource(search_openalex_academic_papers)
    assert "generativelanguage.googleapis.com" not in src

    scout = ScoutAgent()

    # Out-of-domain query returns uppercase "REJECT" in query_expander -> must trigger reject
    reject_res = scout.run({"query": "cách nấu phở bò"})
    assert reject_res.get("scout_decision") == "reject"

    # Broad query returns uppercase "CLARIFY" in query_expander -> must trigger clarify
    clarify_res = scout.run({"query": "AI"})
    assert clarify_res.get("scout_decision") == "clarify"
