"""M0 tests for SQLite cache isolation, fallback non-caching, and provider observations."""

from __future__ import annotations

import json
from pathlib import Path

import fitz
import httpx
import pytest

from core.config import RunConfig, settings, use_run_config
from core.worker_observations import capture_worker_observations, drain_worker_observations
from eval.run_benchmark import run_mode_evaluation
from utils import llm_client as llm_module


def test_heuristic_fallback_is_never_persisted_into_llm_cache(monkeypatch):
    """
    Critical M0 invariant: heuristic _offline_extractive_fallback responses (when Ollama or
    Cloud APIs are unreachable) MUST NEVER poison llm_cache under the requested model's key.
    """
    cache_store: dict[str, str] = {}
    monkeypatch.setattr(llm_module, "get_cached_llm", cache_store.get)
    monkeypatch.setattr(llm_module, "set_cached_llm", lambda k, p, r: cache_store.__setitem__(k, r))

    client = llm_module.UnifiedLLMClient()

    # 1. CPU mode when local Ollama is offline -> falls back to extractive heuristic, must NOT cache
    cpu_cfg = RunConfig.capture(agent_modes={"summarizer": "cpu_max"}, llm_cache_enabled=True)
    monkeypatch.setattr(client, "_call_local_cpu", lambda **kw: (_ for _ in ()).throw(ConnectionError("offline")))
    with use_run_config(cpu_cfg), capture_worker_observations():
        out_cpu = client.generate(
            "Câu hỏi: Thí nghiệm kéo dài bao nhiêu ngày?\n[Unit ID: u1 | Doc: p | Trang: 1]: Thí nghiệm kéo dài 21 ngày.",
            agent="summarizer",
        )
        cpu_obs = drain_worker_observations()

    assert out_cpu
    assert cache_store == {}, "Fallback heuristic response must NOT be written to llm_cache!"
    assert len(cpu_obs) == 1
    assert cpu_obs[0]["provider"] == "heuristic_fallback"
    assert cpu_obs[0]["status"] == "fallback"
    assert "local_cpu_offline" in cpu_obs[0]["fallback_reason"]

    # 2. Cloud mode when Gemini fails -> falls back to extractive heuristic, must NOT cache
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "DEEPSEEK_API_KEY", "")
    monkeypatch.setattr(settings, "OPENAI_API_KEY", "")
    cloud_cfg = RunConfig.capture(agent_modes={"writer": "api"}, llm_cache_enabled=True)
    monkeypatch.setattr(client, "_call_gemini", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("quota_429")))
    with use_run_config(cloud_cfg), capture_worker_observations():
        out_cloud = client.generate("Câu hỏi: Tóm tắt?\n[Unit ID: u1 | Doc: p | Trang: 1]: Nội dung thử nghiệm dài hơn hai mươi lăm ký tự.", agent="writer")
        cloud_obs = drain_worker_observations()

    assert out_cloud
    assert cache_store == {}, "Cloud fallback heuristic response must NOT be written to llm_cache!"
    assert any(ev.get("provider") == "heuristic_fallback" and ev.get("status") == "fallback" for ev in cloud_obs)


def test_gemini_observation_records_requested_attempted_and_actual_model_version(monkeypatch):
    """
    Verify _call_gemini records requested_model, attempted_models cascade, served_model,
    actual modelVersion from REST response JSON, and fallback_reason.
    """
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(settings, "DEFAULT_LLM_MODEL", "gemini-4-carbon")
    monkeypatch.setattr(settings, "FALLBACK_LLM_MODEL", "gemini-3.8-flash")
    monkeypatch.setattr(llm_module, "log_api_usage", lambda *a, **kw: None)
    cfg = RunConfig.capture()

    def transport(request: httpx.Request) -> httpx.Response:
        if "gemini-4-carbon" in request.url.path:
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(
            200,
            json={
                "modelVersion": "gemini-3.8-flash-002",
                "candidates": [{"content": {"parts": [{"text": '{"answer": "ok"}'}]}}],
                "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5, "totalTokenCount": 15},
            },
        )

    client = llm_module.UnifiedLLMClient()
    with httpx.Client(transport=httpx.MockTransport(transport)) as http:
        client.http_client = http
        with use_run_config(cfg), capture_worker_observations():
            text = client._call_gemini("Test prompt", json_mode=True)
            events = drain_worker_observations()

    assert text == '{"answer": "ok"}'
    assert len(events) == 1
    ev = events[0]
    assert ev["provider"] == "gemini"
    assert ev["status"] == "fallback"
    assert ev["requested_model"] == "gemini-4-carbon"
    assert ev["attempted_models"] == ["gemini-4-carbon", "gemini-3.8-flash"]
    assert ev["served_model"] == "gemini-3.8-flash"
    assert ev["model_version"] == "gemini-3.8-flash-002"
    assert "gemini-4-carbon:http_404" in ev["fallback_reason"]


def test_benchmark_cache_regimes_and_mode_db_isolation(monkeypatch, tmp_path: Path):
    """
    Verify run_mode_evaluation honours cold / warm_no_cache / warm_cached regimes
    and collects execution observations per question.
    """
    pdf_path = tmp_path / "sample.pdf"
    with fitz.open() as doc:
        p = doc.new_page()
        p.insert_text((60, 75), "The trial lasted 21 days.")
        doc.save(pdf_path, no_new_id=True)

    questions = [
        {
            "id": "q-1",
            "category": "single_hop",
            "language": "en",
            "documents": ["sample.pdf"],
            "question": "How many days did the trial last?",
            "gold_numbers": ["21"],
            "gold_sources": [{"document": "sample.pdf", "page": 1, "passage": "The trial lasted 21 days."}],
        }
    ]

    summary_cold = run_mode_evaluation("cpu_min", questions, tmp_path, cache_regime="cold")
    summary_warm = run_mode_evaluation("cpu_min", questions, tmp_path, cache_regime="warm_no_cache")

    assert summary_cold["cache_regime"] == "cold"
    assert summary_warm["cache_regime"] == "warm_no_cache"
    assert summary_cold["cache_hit_total"] == 0
    assert summary_warm["cache_hit_total"] == 0
    assert "by_execution_path" in summary_warm
    assert "execution_observation" in summary_warm["records"][0]
