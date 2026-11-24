import pytest


@pytest.fixture
def research_db(monkeypatch, tmp_path):
    from core import db

    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "research.db"))
    monkeypatch.setattr(db, "_db_initialized", False)
    db.init_db()
    return db
