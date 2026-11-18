"""
Unit tests for FastAPI endpoints:
  - Health check & diagnostics
  - Auth status & config
  - Document catalog
  - Run execution & polling
"""

from fastapi.testclient import TestClient
from api.app import app

client = TestClient(app)


def test_health_and_config():
    r_health = client.get("/api/diagnostics")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_config = client.get("/api/config")
    assert r_config.status_code == 200
    assert "default_model" in r_config.json()


def test_auth_status():
    r = client.get("/api/auth/status")
    assert r.status_code == 200
    assert r.json()["authenticated"] is True


def test_agents_spec():
    r = client.get("/api/agents")
    assert r.status_code == 200
    data = r.json()
    agents = data.get("agents", data)
    assert len(agents) == 5
    assert any(a["role"] == "RetrievalAgent" for a in agents)
    assert any(a["role"] == "FactCheckerAgent" for a in agents)


def test_direct_research_api():
    r = client.post("/api/research", json={"query": "Kiến trúc Transformer trong NLP"})
    assert r.status_code == 200
    data = r.json()
    assert "final_report" in data
    assert "verified_claims" in data


def test_runs_api_and_ui_compatibility():
    # 1. Create run
    r_create = client.post("/api/runs", json={"question": "Kiểm tra cơ chế Attention"})
    assert r_create.status_code == 200
    run_id = r_create.json()["id"]

    # 2. Get run status and check Web UI compatibility structure
    r_get = client.get(f"/api/runs/{run_id}")
    assert r_get.status_code == 200
    data = r_get.json()
    assert "events" in data
    assert "request" in data
    assert data["request"]["question"] == "Kiểm tra cơ chế Attention"


def test_research_stream_api():
    r = client.post("/api/research/stream", json={"query": "Khái niệm LoRA"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    text = r.text
    assert "data:" in text
    assert "started" in text
    assert "finished" in text


def test_web_static_assets():
    r_css = client.get("/assets/styles.css")
    assert r_css.status_code == 200
    assert "text/css" in r_css.headers["content-type"]

    r_js = client.get("/assets/app.js")
    assert r_js.status_code == 200
