import pytest
from fastapi.testclient import TestClient

from api import auth
from api.app import app
from core import db

pytestmark = pytest.mark.integration


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(auth, "AUTH_DEV_MODE", False)
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "workspace-test-password")
    monkeypatch.setattr(auth, "_sessions", {})
    return TestClient(app, base_url="https://workspace.test")


@pytest.mark.parametrize(("method", "path"), [
    ("GET", "/api/runs"),
    ("POST", "/api/runs"),
    ("POST", "/api/documents/upload"),
    ("GET", "/api/documents/doc/file"),
    ("GET", "/api/documents/doc/figures/figure"),
    ("GET", "/api/runs/run/export"),
    ("POST", "/api/config/mode"),
    ("POST", "/api/research"),
    ("POST", "/api/research/stream"),
    ("POST", "/api/scout"),
])
def test_api_rejects_unauthorized_requests_before_route(client, method, path):
    response = client.request(method, path)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"


def test_cookie_and_bearer_access_then_logout(client, monkeypatch, tmp_path):
    pdf = tmp_path / "article.pdf"
    pdf.write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(db, "get_document", lambda doc_id: {"path": str(pdf), "name": pdf.name})
    assert client.get("/health").status_code == client.get("/api/health").status_code == 200
    assert client.get("/api/auth/status").json()["authenticated"] is False
    assert client.post("/api/auth/login", json={"password": "wrong"}).status_code == 401

    login = client.post("/api/auth/login", json={"password": "workspace-test-password"})
    assert login.status_code == 200
    cookie = login.headers["set-cookie"].lower()
    assert all(option in cookie for option in ("httponly", "secure", "samesite=strict", "path=/api"))
    assert client.get("/api/auth/status").json()["authenticated"] is True
    assert client.get("/api/config").json()["auth_required"] is True
    pdf_response = client.get("/api/documents/doc/file")
    assert pdf_response.content == pdf.read_bytes()
    assert pdf_response.headers["cache-control"] == "private, no-store"

    bearer = {"Authorization": f"Bearer {login.json()['token']}"}
    api = TestClient(app, base_url="https://workspace.test")
    assert api.get("/api/documents/doc/file", headers=bearer).status_code == 200
    assert client.post("/api/config/mode", headers={"Origin": "https://another.test"}).status_code == 403
    assert client.post("/api/auth/logout").status_code == 200
    assert client.get("/api/documents/doc/file").status_code == 401
    assert api.get("/api/documents/doc/file", headers=bearer).status_code == 401


def test_expired_session_is_rejected(client):
    login = client.post("/api/auth/login", json={"password": "workspace-test-password"})
    auth._sessions[login.json()["token"]] = -1
    assert client.get("/api/runs").status_code == 401


def test_passwordless_dev_access_is_explicit_and_loopback_only(client, monkeypatch):
    monkeypatch.setattr(auth, "AUTH_SECRET_PASSWORD", "")
    assert client.post("/api/auth/login").status_code == 503
    assert client.get("/api/runs").status_code == 401

    monkeypatch.setattr(auth, "AUTH_DEV_MODE", True)
    assert client.get("/api/runs").status_code == 401
    spoofed_host = TestClient(app, base_url="http://localhost", client=("198.51.100.1", 1234))
    assert spoofed_host.get("/api/runs").status_code == 401
    local = TestClient(app, base_url="http://localhost", client=("127.0.0.1", 1234))
    assert local.get("/api/auth/status").json()["authenticated"] is True
    assert local.post("/api/auth/login").status_code == 503
