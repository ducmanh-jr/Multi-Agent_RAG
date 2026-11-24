"""PR-00 Critical P0 Fixes verification:
1. find_best_supporting_unit() never defaults to source_units[0]; filters VI/EN stopwords and requires meaningful overlap.
2. Unmatched claims retain evidence_ids=[] and produce NOT_ENOUGH_INFO + EvidenceGap in review_claims().
3. Kaggle Supervisor Gateway enforces Bearer token auth on inference proxy endpoints, fails closed on
   PUBLIC_TUNNEL=true without token, strips auth header before internal forwarding, and forbids wildcard CORS.
4. Backend clients (llm_client, remote_dense, remote_rerank) send Bearer REMOTE_INFERENCE_TOKEN when configured.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from agents.agent1_retrieval.tools.remote_dense import RemoteDenseRetriever
from agents.agent1_retrieval.tools.remote_rerank import rerank_passages
from core.config import settings
from core.verifier import find_best_supporting_unit, review_claims, split_answer_into_atomic_claims
from kaggle import kaggle_supervisor_gateway as gw
from utils.llm_client import llm_client


def test_find_best_supporting_unit_never_defaults_to_first_unit():
    units = [
        {"id": "u1", "doc_id": "paper", "page_no": 1, "text": "Orchard irrigation trial ran for 21 days."},
        {"id": "u2", "doc_id": "paper", "page_no": 2, "text": "Greenhouse humidity remained steady at 65%."},
    ]
    # Completely unrelated claim -> must return None (NOT u1!)
    assert find_best_supporting_unit("Quantum superconductor reached zero kelvin.", units) is None

    # Only common English stopwords shared ("the", "for", "at") -> must return None
    assert find_best_supporting_unit("The weather for the festival at night was pleasant.", units) is None

    # Only common Vietnamese stopwords shared -> must return None
    vi_units = [
        {"id": "v1", "doc_id": "bao", "page_no": 1, "text": "Thuật toán nén lượng tử hoạt động ở 4 bit."},
    ]
    assert find_best_supporting_unit("Các nhân vật trong bộ phim này đều có kết cục rất đẹp.", vi_units) is None

    # Genuine lexical/numeric overlap -> matches exact supporting unit
    matched = find_best_supporting_unit("The orchard irrigation trial lasted 21 days.", units)
    assert matched is not None and matched["id"] == "u1"


def test_unmatched_claim_keeps_empty_evidence_ids_and_creates_evidence_gap():
    units = [
        {"id": "u1", "doc_id": "paper", "page_no": 1, "text": "Orchard irrigation trial ran for 21 days."},
    ]
    fabricated = "Quantum superconductor reached zero kelvin in space."

    claims = split_answer_into_atomic_claims(
        fabricated,
        units,
        existing_claims=[{"id": "c1", "text": fabricated}],
    )
    assert len(claims) == 1
    assert claims[0]["evidence_ids"] == []
    assert claims[0]["unit_id"] is None

    reviewer_called = []
    def strict_reviewer(pairs):
        reviewer_called.append(pairs)
        return []

    result = review_claims(
        {"query": "What happened?", "answer": fabricated, "claims": claims, "source_units": units},
        reviewer=strict_reviewer,
    )
    assert reviewer_called == []
    assert not result["verified_claims"]
    assert result["fact_check_label"] == "not_enough_info"
    assert len(result["rejected_claims"]) == 1
    assert result["rejected_claims"][0]["claim_id"] == "c1"
    assert len(result["evidence_gaps"]) == 1


def test_gateway_cors_has_no_wildcard_and_inference_proxy_enforces_bearer_auth():
    assert "*" not in gw.ALLOWED_ORIGINS
    client = TestClient(gw.app, raise_server_exceptions=False)

    # 1. PUBLIC_TUNNEL=true without any token -> fail-closed 503 on inference proxy
    with patch.object(gw, "ADMIN_TOKEN", None), \
         patch.object(gw, "INFERENCE_TOKEN", None), \
         patch.object(gw, "PUBLIC_TUNNEL", True):
        resp = client.post("/api/embed", json={"texts": ["hello"]})
        assert resp.status_code == 503

    # 2. Token configured -> 401 missing, 403 invalid, 200 valid (and strips Authorization downstream)
    forwarded_headers = {}

    class DummyResp:
        status_code = 200
        content = b'{"ok": true}'
        headers = {"content-type": "application/json"}

    async def fake_request(method, url, headers=None, content=None, params=None):
        forwarded_headers.update(headers or {})
        return DummyResp()

    with patch.object(gw, "INFERENCE_TOKEN", "infer-secret-999"), \
         patch.object(gw, "PUBLIC_TUNNEL", True), \
         patch.object(gw.worker_manager, "transition_state", "ready"), \
         patch.object(gw.worker_manager, "active_worker", "gpu_finetune"), \
         patch.object(gw.worker_manager, "get_active_worker_url", return_value="http://127.0.0.1:8001"), \
         patch.object(gw._http_client, "request", new=fake_request):
        assert client.post("/api/embed", json={"texts": ["a"]}).status_code == 401
        assert client.post(
            "/api/embed",
            json={"texts": ["a"]},
            headers={"Authorization": "Bearer wrong-token"},
        ).status_code == 403

        ok = client.post(
            "/api/embed",
            json={"texts": ["a"]},
            headers={"Authorization": "Bearer infer-secret-999"},
        )
        assert ok.status_code == 200
        assert "authorization" not in {k.lower() for k in forwarded_headers}

        # /health remains reachable for liveness probes
        with patch.object(gw.worker_manager, "get_status_payload", return_value={
            "runtime_status": "running",
            "selected_mode": "gpu_finetune",
            "active_worker": "gpu_finetune",
            "transition_state": "ready",
            "transition_message": "Sẵn sàng phục vụ.",
            "active_worker_capabilities": {},
        }):
            assert client.get("/health").status_code == 200


def test_backend_clients_forward_remote_inference_bearer_token(monkeypatch):
    seen_auth: list[tuple[str, str | None]] = []
    fake_rev = "a" * 40

    def handler(request: httpx.Request) -> httpx.Response:
        seen_auth.append((request.url.path, request.headers.get("authorization")))
        if request.url.path == "/api/embed":
            texts = json.loads(request.content.decode("utf-8"))["texts"]
            return httpx.Response(200, json={
                "model": settings.EMBEDDING_MODEL,
                "revision": fake_rev,
                "embeddings": [[1.0] + [0.0] * 1023 for _ in texts],
            })
        if request.url.path == "/api/rerank":
            cands = json.loads(request.content.decode("utf-8"))["candidates"]
            return httpx.Response(200, json={
                "model": settings.RERANKER_MODEL,
                "revision": fake_rev,
                "scores": [float(i) for i in range(len(cands))],
                "ranked_indices": list(reversed(range(len(cands)))),
            })
        return httpx.Response(200, json={"text": "ok"})

    monkeypatch.setattr(settings, "REMOTE_INFERENCE_TOKEN", "my-kaggle-secret")
    mock_client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(llm_client, "http_client", mock_client)

    llm_client._post_worker("https://kaggle.example.invalid", "generate", {"prompt": "hi"}, 5.0, "generator")
    RemoteDenseRetriever().rank(
        "query",
        [{"id": "c1", "text": "passage text", "doc_id": "d", "page_no": 1}],
        top_k=1,
        client=mock_client,
        url="https://kaggle.example.invalid",
        model=settings.EMBEDDING_MODEL,
        timeout=5.0,
    )
    rerank_passages(
        "query",
        [{"id": "c1", "text": "first"}, {"id": "c2", "text": "second"}],
        client=mock_client,
        url="https://kaggle.example.invalid",
        model=settings.RERANKER_MODEL,
        timeout=5.0,
    )

    assert ("/generate", "Bearer my-kaggle-secret") in seen_auth
    assert ("/api/embed", "Bearer my-kaggle-secret") in seen_auth
    assert ("/api/rerank", "Bearer my-kaggle-secret") in seen_auth
