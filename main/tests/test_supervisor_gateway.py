"""
Unit & Integration Tests for Kaggle Supervisor Gateway (v2.2.0).

Verifies:
1. Complete integration with WorkerManager.
2. Status contract and /health payload schema conformance.
3. Bearer Token Authentication guard on /api/admin/* endpoints.
4. Dynamic Mode Switching (POST /api/admin/switch_mode) returning 202 Accepted (async) or 200 OK (wait=True).
5. Request Lease Guarding: Proxy endpoints acquire lease and reject with HTTP 503 during transitions.
6. vLLM payload adaptation when Worker B (gpu_max) is active.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from kaggle import kaggle_supervisor_gateway as gw
from kaggle.worker_manager import WorkerSwitchingError


@pytest.fixture
def client():
    return TestClient(gw.app, raise_server_exceptions=False)


def test_gateway_health_contract_schema(client):
    """Verify /health and / return truthful status contracts matching WorkerManager."""
    with patch.object(gw.worker_manager, "get_status_payload", return_value={
        "runtime_status": "running",
        "selected_mode": "gpu_finetune",
        "active_worker": "gpu_finetune",
        "target_worker": None,
        "transition_state": "ready",
        "transition_id": "init_001",
        "transition_message": "Sẵn sàng phục vụ.",
        "active_requests": 0,
        "queue_depth": 0,
        "is_switching": False,
        "vram_allocation": {"gpu_0": {"used_gb": 7.5, "total_gb": 16.0, "free_gb": 8.5}},
        "vram_probe": {"source": "nvidia-smi", "is_valid": True, "error": None},
        "worker_profiles_available": {"gpu_finetune": True, "gpu_max": True},
        "active_worker_capabilities": {"router": True, "generator": True},
    }):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()

        # Contract fields
        assert data["runtime_status"] == "running"
        assert data["selected_mode"] == "gpu_finetune"
        assert data["active_worker"] == "gpu_finetune"
        assert data["transition_state"] == "ready"
        assert "vram_allocation" in data
        assert "worker_profiles_available" in data

        # Backward compatibility fields
        assert data["status"] == "online"
        assert data["active_mode"] == "gpu_finetune"
        assert data["worker_health"]["status"] == "online"
        assert data["worker_health"]["worker_status"] == "online"


def test_gateway_admin_auth_guard(client):
    """Verify admin endpoints enforce Bearer Token authentication when token is set."""
    with patch.dict(os.environ, {"SUPERVISOR_ADMIN_TOKEN": "secret_token_123"}):
        with patch.object(gw, "ADMIN_TOKEN", "secret_token_123"):
            # 1. Missing Authorization header -> 401
            resp = client.get("/api/admin/status")
            assert resp.status_code == 401
            assert "header Authorization" in resp.text

            # 2. Invalid Token -> 403
            resp = client.get("/api/admin/status", headers={"Authorization": "Bearer wrong_token"})
            assert resp.status_code == 403
            assert "Forbidden" in resp.text

            # 3. Valid Token -> 200
            with patch.object(gw.worker_manager, "get_status_payload", return_value={"runtime_status": "running"}):
                resp = client.get("/api/admin/status", headers={"Authorization": "Bearer secret_token_123"})
                assert resp.status_code == 200
                assert resp.json()["runtime_status"] == "running"


def test_gateway_switch_mode_async_accepted(client):
    """Verify POST /api/admin/switch_mode returns 202 Accepted for background transition."""
    with patch.object(gw.worker_manager, "request_switch_mode", new_callable=AsyncMock) as mock_switch:
        resp = client.post("/api/admin/switch_mode", json={"target_mode": "gpu_max"})
        assert resp.status_code == 202
        data = resp.json()
        assert data["status"] == "accepted"
        assert data["target_mode"] == "gpu_max"
        assert data["transition_state"] == "switching"


def test_gateway_switch_mode_synchronous_wait(client):
    """Verify POST /api/admin/switch_mode with wait=True returns 200 OK after completion."""
    with patch.object(gw.worker_manager, "request_switch_mode", new_callable=AsyncMock, return_value={
        "status": "success",
        "transition_id": "trans_123",
        "selected_mode": "gpu_max",
        "active_worker": "gpu_max",
        "message": "Worker GPU_MAX đã trực tuyến và sẵn sàng phục vụ."
    }):
        resp = client.post("/api/admin/switch_mode", json={"target_mode": "gpu_max", "wait": True})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "success"
        assert data["active_worker"] == "gpu_max"


def test_gateway_switch_mode_rejects_invalid_mode(client):
    """Verify POST /api/admin/switch_mode rejects invalid mode names with HTTP 400."""
    resp = client.post("/api/admin/switch_mode", json={"target_mode": "unsupported_mode"})
    assert resp.status_code == 400
    assert "Chế độ không hợp lệ" in resp.text


def test_gateway_proxy_rejects_with_503_during_transition(client):
    """Verify reverse proxy returns 503 while worker is in transition."""
    for state in ["stopping", "reclaiming_vram", "spawning", "draining"]:
        gw.worker_manager.transition_state = state
        gw.worker_manager.transition_id = "test_trans_id"

        resp = client.post("/api/router", json={"query": "test"})
        assert resp.status_code == 503
        data = resp.json()
        assert "GPU đang trong quá trình chuyển chế độ" in data["error"]
        assert data["transition_state"] == state

    gw.worker_manager.transition_state = "ready"


def test_gateway_proxy_acquires_lease_and_catches_switching_error(client):
    """Verify proxy catches WorkerSwitchingError from acquire_worker_lease and returns 503."""
    gw.worker_manager.transition_state = "ready"
    gw.worker_manager.active_worker = "gpu_finetune"

    with patch.object(gw.worker_manager, "acquire_worker_lease", side_effect=WorkerSwitchingError("Lease rejected: switching")):
        resp = client.post("/api/router", json={"query": "test"})
        assert resp.status_code == 503
        assert "Lease rejected" in resp.json()["error"]


def test_gateway_proxy_adapts_payload_for_gpu_max_vllm(client):
    """
    Verify that when active_worker is gpu_max, academic endpoints like /api/router
    are transformed into vLLM /v1/chat/completions format.
    """
    gw.worker_manager.transition_state = "ready"
    gw.worker_manager.active_worker = "gpu_max"

    mock_vllm_response = MagicMock()
    mock_vllm_response.status_code = 200
    mock_vllm_response.json.return_value = {
        "choices": [{"message": {"content": '{"intent": "academic_research", "sub_queries": []}'}}]
    }

    with patch.object(gw._http_client, "post", new_callable=AsyncMock, return_value=mock_vllm_response) as mock_post:
        resp = client.post("/api/router", json={"query": "Explain attention mechanism"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "Qwen2.5-32B-Instruct-AWQ"
        assert '{"intent": "academic_research"' in data["text"]

        # Verify format sent to vLLM
        mock_post.assert_called_once()
        args, kwargs = mock_post.call_args
        assert "/v1/chat/completions" in args[0]
        payload = kwargs["json"]
        assert payload["model"] == "Qwen/Qwen2.5-32B-Instruct-AWQ"
        assert payload["messages"][1]["content"] == "Explain attention mechanism"
