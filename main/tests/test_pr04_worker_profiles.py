"""PR-04 Kaggle Worker Hardening verification:
1. kaggle_finetune_server.py /health exposes truthful ready & mandatory_ready fields
   for both 'gpu_finetune' (4 Qwen adapters + factcheck) and 'hybrid_specialist' (factcheck only).
2. WorkerManager._poll_worker_ready() honors profile-aware readiness (probing /api/factcheck
   under hybrid_specialist and enforcing optional REQUIRE_EMBEDDING_READY / REQUIRE_RERANKER_READY,
   while keeping gpu_finetune strictly fail-closed on missing Qwen adapters).
3. /health never leaks API keys or admin tokens; load_kaggle_secret_into_env works safely.
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

from kaggle.run_kaggle import load_kaggle_secret_into_env
from kaggle.worker_manager import WorkerManager


def _load_finetune_server():
    path = Path(__file__).resolve().parents[2] / "kaggle" / "kaggle_finetune_server.py"
    spec = importlib.util.spec_from_file_location("pr04_kaggle_finetune_server", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_finetune_server_health_truthful_contract_across_profiles(monkeypatch):
    srv = _load_finetune_server()
    client = TestClient(srv.app)

    monkeypatch.setenv("GEMINI_API_KEY", "super-secret-gemini-key")
    monkeypatch.setenv("SUPERVISOR_API_TOKEN", "super-secret-tunnel-token")

    # 1. gpu_finetune profile: missing Qwen adapters -> degraded / not ready
    monkeypatch.setenv("KAGGLE_WORKER_PROFILE", "gpu_finetune")
    monkeypatch.setattr(srv, "_models", {"factcheck": object()})
    monkeypatch.setattr(srv, "_loaded_adapters", ["router"])
    resp_degraded = client.get("/health").json()
    assert resp_degraded["ready"] is False
    assert resp_degraded["mandatory_ready"] is False
    assert resp_degraded["status"] == "degraded"
    assert "super-secret" not in str(resp_degraded)

    # 2. gpu_finetune profile: full 4 adapters + qwen_cluster + factcheck -> online / ready
    monkeypatch.setattr(srv, "_models", {"qwen_cluster": object(), "factcheck": object(), "embedding": object()})
    monkeypatch.setattr(srv, "_loaded_adapters", ["router", "generator", "critic", "writer"])
    resp_full = client.get("/health").json()
    assert resp_full["ready"] is True
    assert resp_full["mandatory_ready"] is True
    assert resp_full["status"] == "online"
    assert resp_full["capabilities"]["embedding"] is True

    # 3. hybrid_specialist profile: ONLY factcheck loaded (no Qwen!) -> ready=True
    monkeypatch.setenv("KAGGLE_WORKER_PROFILE", "hybrid_specialist")
    monkeypatch.setattr(srv, "_models", {"factcheck": object(), "embedding": object(), "reranker": object()})
    monkeypatch.setattr(srv, "_loaded_adapters", [])
    resp_hybrid = client.get("/health").json()
    assert resp_hybrid["worker_profile"] == "hybrid_specialist"
    assert resp_hybrid["ready"] is True
    assert resp_hybrid["mandatory_ready"] is True
    assert resp_hybrid["mandatory_adapters"] == []
    assert resp_hybrid["capabilities"]["qwen_cluster"] is False
    assert resp_hybrid["capabilities"]["factcheck"] is True
    assert "super-secret" not in str(resp_hybrid)


def test_worker_manager_profile_aware_readiness_polling(monkeypatch):
    wm = WorkerManager(project_dir=os.path.abspath("."))
    seen_probes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_probes.append(request.url.path)
        if request.url.path == "/health":
            return httpx.Response(200, json={
                "status": "online",
                "ready": True,
                "mandatory_ready": True,
                "worker_profile": "hybrid_specialist",
                "loaded_adapters": [],
                "active_models": ["factcheck", "embedding", "reranker"],
            })
        if request.url.path == "/api/factcheck":
            return httpx.Response(200, json={"evaluations": []})
        return httpx.Response(404)

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def patched_async_client(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    # hybrid_specialist profile passes readiness using /api/factcheck without Qwen adapters
    monkeypatch.setenv("KAGGLE_WORKER_PROFILE", "hybrid_specialist")
    monkeypatch.setenv("REQUIRE_EMBEDDING_READY", "true")
    monkeypatch.setenv("REQUIRE_RERANKER_READY", "true")
    with patch("httpx.AsyncClient", side_effect=patched_async_client):
        ready = asyncio.run(wm._poll_worker_ready("gpu_finetune", "http://127.0.0.1:8001", timeout_seconds=2))
    assert ready is True
    assert "/api/factcheck" in seen_probes
    assert "/api/router" not in seen_probes

    # Under strict gpu_finetune profile, the exact same Qwen-less payload MUST NOT pass readiness
    seen_probes.clear()
    monkeypatch.setenv("KAGGLE_WORKER_PROFILE", "gpu_finetune")
    with patch("httpx.AsyncClient", side_effect=patched_async_client):
        ready_strict = asyncio.run(wm._poll_worker_ready("gpu_finetune", "http://127.0.0.1:8001", timeout_seconds=1))
    assert ready_strict is False


def test_load_kaggle_secret_into_env_safe_behavior(monkeypatch):
    monkeypatch.delenv("TEST_KAGGLE_SECRET_VAR", raising=False)

    fake_mod = types.ModuleType("kaggle_secrets")

    class FakeClient:
        def get_secret(self, name):
            return "secret_value_123" if name == "TEST_KAGGLE_SECRET_VAR" else ""

    fake_mod.UserSecretsClient = FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "kaggle_secrets", fake_mod)

    assert load_kaggle_secret_into_env("TEST_KAGGLE_SECRET_VAR") is True
    assert os.environ["TEST_KAGGLE_SECRET_VAR"] == "secret_value_123"

    # Existing env value is preserved (never overwritten)
    monkeypatch.setenv("TEST_KAGGLE_SECRET_VAR", "already_set")
    assert load_kaggle_secret_into_env("TEST_KAGGLE_SECRET_VAR") is True
    assert os.environ["TEST_KAGGLE_SECRET_VAR"] == "already_set"
