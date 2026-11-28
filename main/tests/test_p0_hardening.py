import importlib
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient

from core.config import agent_inference_mode, settings
from core.figure_reader import _resolve_vision_provider
from core.provider_router import is_gemini_permitted, provider_router


def test_env_example_contains_no_hardcoded_credentials_or_tunnel_url():
    env_example = (Path(__file__).resolve().parents[2] / ".env.example").read_text(encoding="utf-8")
    assert "GEMINI_API_KEY=\n" in env_example or "GEMINI_API_KEY=\r\n" in env_example
    assert "trycloudflare.com" not in env_example
    assert "WQ.Ab8RN6" not in env_example


def test_single_provider_gate_blocks_gemini_in_local_only_and_single_modes(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "test-fake-key-123456")
    monkeypatch.setattr(settings, "HYBRID_PROVIDER_PROFILE", "local_only")
    monkeypatch.setattr(settings, "LOCAL_CPU_URL", "")
    monkeypatch.delenv("FIGURE_VISION_MODEL", raising=False)

    assert is_gemini_permitted("hybrid") is False
    assert is_gemini_permitted("cpu_min") is False
    assert is_gemini_permitted("gpu_finetune") is False
    assert is_gemini_permitted("gemini") is True

    # Vision resolver obeys local_only zero-Gemini invariant
    assert _resolve_vision_provider("auto", "hybrid") == ""
    assert _resolve_vision_provider("gemini", "hybrid") == ""

    # Planner chain obeys local_only zero-Gemini invariant
    chain = provider_router.resolve_chain(
        "planner",
        caps={"gemini": {"online": True}, "gpu_finetune": {"online": True}},
    )
    assert "gemini" not in chain
    assert chain == ["gpu_finetune", "local"]

    # llm_client.generate() blocks _call_gemini under hybrid + local_only
    llm_module = importlib.import_module("utils.llm_client")
    called = []
    monkeypatch.setattr(llm_module.llm_client, "_call_gemini", lambda *a, **kw: called.append(True) or "BAD")
    monkeypatch.setattr(llm_module, "get_cached_llm", lambda *_: None)
    monkeypatch.setattr(llm_module, "set_cached_llm", lambda *a, **kw: None)
    with agent_inference_mode("hybrid"):
        llm_module.llm_client.generate("Test zero gemini gate", agent="summarizer")
    assert called == []


def test_gateway_cloudflare_aware_rate_limiter_enforces_429(monkeypatch):
    import kaggle.kaggle_supervisor_gateway as gw

    monkeypatch.setattr(gw, "ADMIN_TOKEN", None)
    monkeypatch.setattr(gw, "INFERENCE_TOKEN", None)
    monkeypatch.setattr(gw, "PUBLIC_TUNNEL", False)
    monkeypatch.delenv("SUPERVISOR_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("SUPERVISOR_API_TOKEN", raising=False)
    monkeypatch.delenv("REMOTE_INFERENCE_TOKEN", raising=False)
    monkeypatch.setattr(gw, "RATE_LIMIT_REQUESTS", 2)
    monkeypatch.setattr(gw, "RATE_LIMIT_WINDOW", 60.0)
    gw._client_request_history.clear()

    gw.worker_manager.transition_state = "ready"
    gw.worker_manager.active_worker = "gpu_finetune"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b'{"ok": true}'
    mock_resp.headers = {"content-type": "application/json"}

    client = TestClient(gw.app)
    with patch.object(gw._http_client, "request", new_callable=AsyncMock, return_value=mock_resp):
        headers_a = {"CF-Connecting-IP": "203.0.113.10"}
        headers_b = {"CF-Connecting-IP": "203.0.113.20"}
        assert client.post("/api/embed", json={"texts": ["a"]}, headers=headers_a).status_code == 200
        assert client.post("/api/embed", json={"texts": ["b"]}, headers=headers_a).status_code == 200
        assert client.post("/api/embed", json={"texts": ["c"]}, headers=headers_a).status_code == 429
        # Different Cloudflare client IP has its own bucket
        assert client.post("/api/embed", json={"texts": ["d"]}, headers=headers_b).status_code == 200


def test_finetune_server_bounds_summarizer_and_writer_max_tokens():
    import kaggle.kaggle_finetune_server as kfs

    s_req = kfs.SummarizeRequest(query="Q", context="C")
    w_req = kfs.WriterRequest(query="Q")
    assert s_req.max_tokens == 768
    assert w_req.max_tokens == 512
