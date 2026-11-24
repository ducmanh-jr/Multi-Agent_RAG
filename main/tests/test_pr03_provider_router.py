"""PR-03 Hybrid Provider Router verification:
1. Strict separation of ExecutionMode vs ModelProvider (single modes cannot silently call Gemini).
2. Role-based routing under HYBRID_PROVIDER_POLICY (gemini_first & auto complexity-aware).
3. Grounding invariants: fact_checker (Local BGE NLI), embedding, reranker, and citation_audit
   remain local/Kaggle primary even under gemini_first.
4. HybridSummarizerAgent falls back to Kaggle Qwen 7B on Gemini 429/failure.
5. llm_client._call_gemini uses config-driven DEFAULT_LLM_MODEL (gemini-3.8-flash) with 429 fallback.
"""

from __future__ import annotations

import httpx
import pytest

from core.config import settings
from core.provider_router import (
    ProviderPolicyError,
    assert_provider_allowed,
    load_hybrid_provider_policy,
    provider_router,
)
from modes.hybrid.summarizer import HybridSummarizerAgent
from utils.llm_client import UnifiedLLMClient


def test_single_execution_modes_enforce_provider_isolation():
    for isolated_mode in ("cpu_min", "cpu_max", "gpu_finetune", "gpu_max"):
        with pytest.raises(ProviderPolicyError):
            assert_provider_allowed("gemini", isolated_mode, role="generator")

    # Hybrid and Gemini explicitly allow Gemini; local is always allowed
    assert_provider_allowed("gemini", "hybrid", role="generator")
    assert_provider_allowed("kaggle", "hybrid", role="fact_checker")
    assert_provider_allowed("local", "cpu_min", role="citation_audit")


def test_hybrid_provider_policy_routing_and_grounding_invariants(monkeypatch):
    caps = {
        "gemini": {"online": True},
        "gpu_finetune": {"online": True},
        "kaggle_max": {"online": False, "has_vllm_32b": False},
    }

    # 1. gemini_first profile: Gemini primary for planner/generator/writer, Qwen 7B fallback;
    #    fact_checker & embedding/reranker & citation_audit stay local/Kaggle primary!
    monkeypatch.setattr(settings, "HYBRID_PROVIDER_PROFILE", "gemini_first")
    assert provider_router.resolve_chain("generator", caps=caps) == ["gemini", "gpu_finetune", "cpu_max"]
    assert provider_router.resolve_chain("writer", caps=caps) == ["gemini", "gpu_finetune", "cpu_max"]
    assert provider_router.resolve_chain("planner", caps=caps) == ["gemini", "gpu_finetune", "local"]
    assert provider_router.resolve_chain("fact_checker", caps=caps)[0] == "gpu_finetune"
    assert provider_router.resolve_chain("embedding", caps=caps) == ["gpu_finetune"]
    assert provider_router.resolve_chain("reranker", caps=caps) == ["gpu_finetune"]
    assert provider_router.resolve_chain("citation_audit", caps=caps) == ["local"]

    # 2. auto profile: HIGH complexity / multi-doc -> Gemini primary; simple single-doc -> Qwen fast-path
    monkeypatch.setattr(settings, "HYBRID_PROVIDER_PROFILE", "auto")
    assert provider_router.resolve_chain("generator", state={"complexity": "HIGH"}, caps=caps)[0] == "gemini"
    assert provider_router.resolve_chain("generator", state={"pdf_paths": ["a.pdf", "b.pdf"]}, caps=caps)[0] == "gemini"
    assert provider_router.resolve_chain("generator", state={"complexity": "LOW"}, caps=caps)[0] == "gpu_finetune"

    # 3. Policy override cannot override citation_audit away from local
    pol = load_hybrid_provider_policy({"citation_audit": "gemini", "generator": "kaggle"})
    assert pol["citation_audit"] == "local"
    assert pol["generator"] == "kaggle"


def test_hybrid_summarizer_gemini_first_falls_back_to_qwen_on_rate_limit(monkeypatch):
    monkeypatch.setattr(settings, "HYBRID_PROVIDER_PROFILE", "gemini_first")
    monkeypatch.setattr("modes.hybrid.summarizer.probe_all_capabilities", lambda: {
        "gemini": {"online": True},
        "gpu_finetune": {"online": True},
    })

    summarizer = HybridSummarizerAgent()
    order = []

    def fail_gemini(state):
        order.append("gemini")
        raise RuntimeError("Gemini HTTP 429 rate limit")

    def ok_qwen(state):
        order.append("gpu_finetune")
        return {"answer": "Qwen 7B local fallback answer [1].", "critic_action": "approve"}

    monkeypatch.setattr(summarizer._gemini, "run", fail_gemini)
    monkeypatch.setattr(summarizer._gpu_finetune, "run", ok_qwen)

    out = summarizer.run({"query": "Compare models", "complexity": "HIGH"})
    assert order == ["gemini", "gpu_finetune"]
    assert out["answer"] == "Qwen 7B local fallback answer [1]."
    assert out["reasoning_trace"][-1]["details"]["selected_generator_backend"] == "gpu_finetune_dual_qlora"


def test_llm_client_uses_config_gemini_model_and_cascades_on_429(monkeypatch):
    monkeypatch.setattr(settings, "DEFAULT_LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "gemini-3.1-flash-lite")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")

    requested_urls = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_urls.append(str(request.url))
        if "gemini-3.8-flash" in str(request.url):
            return httpx.Response(429, headers={"Retry-After": "0"}, json={"error": "rate_limited"})
        return httpx.Response(200, json={
            "candidates": [{"content": {"parts": [{"text": "Fallback Gemini OK"}]}}]
        })

    client = UnifiedLLMClient()
    monkeypatch.setattr(client, "http_client", httpx.Client(transport=httpx.MockTransport(handler)))
    try:
        text = client._call_gemini("Hello")
        assert text == "Fallback Gemini OK"
        assert any("gemini-3.8-flash:generateContent" in u for u in requested_urls)
        assert any("gemini-3.1-flash-lite:generateContent" in u for u in requested_urls)
    finally:
        client.http_client.close()
