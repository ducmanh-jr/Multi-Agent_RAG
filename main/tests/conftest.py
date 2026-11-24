import sys
import pytest


@pytest.fixture(autouse=True)
def _isolate_local_env_tokens(monkeypatch):
    """Prevent local developer .env tokens from leaking into gateway auth unit tests."""
    for env_key in (
        "SUPERVISOR_ADMIN_TOKEN",
        "SUPERVISOR_API_TOKEN",
        "REMOTE_INFERENCE_TOKEN",
        "ADMIN_API_KEY",
        "PUBLIC_TUNNEL",
    ):
        monkeypatch.delenv(env_key, raising=False)
    for mod_name in ("kaggle.kaggle_supervisor_gateway", "kaggle_supervisor_gateway"):
        gw = sys.modules.get(mod_name)
        if gw is not None:
            monkeypatch.setattr(gw, "ADMIN_TOKEN", None, raising=False)
            monkeypatch.setattr(gw, "INFERENCE_TOKEN", None, raising=False)
            monkeypatch.setattr(gw, "PUBLIC_TUNNEL", False, raising=False)


@pytest.fixture
def research_db(monkeypatch, tmp_path):
    from core import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "research.db"))
    monkeypatch.setattr(db, "_db_initialized", False)
    db.init_db()
    return db
