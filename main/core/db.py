"""

Persistent SQLite Database Engine for Multi-Agent Research RAG.

Stores documents, research runs, and LangGraph traces persistently in data/workspace.db.

Survives server restarts and worker reloads with zero external dependencies.

"""



from __future__ import annotations



import datetime

import json

import os

import sqlite3

import uuid

from typing import Any, Optional



from core.config import settings

from utils.logger import logger



DB_PATH = os.path.join(settings.BASE_DIR, "data", "workspace.db")


def _migrate_orphan_runs(cursor: sqlite3.Cursor):
    """Groups existing orphan runs that lack a session_id into sessions."""
    try:
        cursor.execute("SELECT id, question, created_at, document_id FROM research_runs WHERE session_id IS NULL ORDER BY created_at ASC")
        orphans = cursor.fetchall()
        if not orphans:
            return
        doc_session_map = {}
        for r in orphans:
            r_id = r["id"] if isinstance(r, sqlite3.Row) else r[0]
            question = r["question"] if isinstance(r, sqlite3.Row) else r[1]
            created_at = r["created_at"] if isinstance(r, sqlite3.Row) else r[2]
            doc_id = r["document_id"] if isinstance(r, sqlite3.Row) else r[3]
            s_key = doc_id or f"single_{r_id}"
            if s_key not in doc_session_map:
                sess_id = f"session_{uuid.uuid4().hex[:10]}"
                clean_title = (question or "Cuộc trò chuyện").strip()
                if clean_title.startswith("/"):
                    clean_title = clean_title.lstrip("/").replace("-", " ").capitalize()
                if len(clean_title) > 60:
                    clean_title = clean_title[:57] + "..."
                cursor.execute("""
                    INSERT OR IGNORE INTO chat_sessions (id, title, created_at, updated_at, document_id)
                    VALUES (?, ?, ?, ?, ?)
                """, (sess_id, clean_title or "Đoạn nghiên cứu", created_at, created_at, doc_id))
                doc_session_map[s_key] = sess_id
            target_sess_id = doc_session_map[s_key]
            cursor.execute("UPDATE research_runs SET session_id = ? WHERE id = ?", (target_sess_id, r_id))
    except Exception as e:
        logger.warning(f"Session migration notice: {e}")


def _create_tables(conn: sqlite3.Connection):

    """Creates tables if they do not exist."""

    cursor = conn.cursor()



    # 1. Documents Table

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS documents (

            id TEXT PRIMARY KEY,

            name TEXT NOT NULL,

            path TEXT NOT NULL,

            size INTEGER DEFAULT 0,

            doc_type TEXT DEFAULT 'pdf',

            uploaded_at TEXT NOT NULL,

            metadata TEXT DEFAULT '{}'

        );

    """)



    # 2. Research Runs Table

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS research_runs (

            id TEXT PRIMARY KEY,

            question TEXT NOT NULL,

            status TEXT DEFAULT 'processing',

            created_at TEXT NOT NULL,

            final_report TEXT,

            verified_claims TEXT DEFAULT '[]',

            rejected_claims TEXT DEFAULT '[]',

            evidence_spans TEXT DEFAULT '[]',

            reasoning_trace TEXT DEFAULT '[]',

            error TEXT,

            document_id TEXT,

            session_id TEXT

        );
    """)
    try:
        cursor.execute("ALTER TABLE research_runs ADD COLUMN document_id TEXT")
    except Exception:
        pass
    try:
        cursor.execute("ALTER TABLE research_runs ADD COLUMN session_id TEXT")
    except Exception:
        pass

    # 2b. Chat Sessions Table (Conversations / Đoạn chat)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            document_id TEXT
        );
    """)

    _migrate_orphan_runs(cursor)




    # 3. LLM Semantic & Prompt Cache Table

    cursor.execute("""

        CREATE TABLE IF NOT EXISTS llm_cache (

            prompt_hash TEXT PRIMARY KEY,

            prompt TEXT NOT NULL,

            response TEXT NOT NULL,

            created_at TEXT NOT NULL

        );

    """)

    # 4. API Usage Logs Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_usage_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            provider TEXT NOT NULL,
            model TEXT,
            endpoint TEXT,
            timestamp TEXT NOT NULL
        );
    """)

    conn.commit()





def get_db_connection() -> sqlite3.Connection:

    """Creates a thread-safe connection with dictionary-like row factory and ensures tables exist."""

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    conn.row_factory = sqlite3.Row

    _create_tables(conn)

    return conn





def init_db():

    """Initializes tables for documents, research runs, and system settings."""

    conn = get_db_connection()

    conn.close()

    logger.info("Initialized persistent SQLite workspace database at: " + DB_PATH)





# ==================== DOCUMENT OPERATIONS ====================



def save_document(doc_id: str, name: str, path: str, size: int, doc_type: str = "pdf", metadata: dict = None) -> dict[str, Any]:

    conn = get_db_connection()

    cursor = conn.cursor()

    now = datetime.datetime.now().isoformat()

    meta_json = json.dumps(metadata or {}, ensure_ascii=False)



    cursor.execute("""

        INSERT OR REPLACE INTO documents (id, name, path, size, doc_type, uploaded_at, metadata)

        VALUES (?, ?, ?, ?, ?, ?, ?)

    """, (doc_id, name, path, size, doc_type, now, meta_json))



    conn.commit()

    conn.close()



    return {

        "id": doc_id,

        "name": name,

        "path": path,

        "size": size,

        "type": doc_type,

        "uploaded_at": now,

    }





def get_document(doc_id: str) -> Optional[dict[str, Any]]:

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))

    row = cursor.fetchone()

    if not row:

        # Resilient fallback: support prefix, name match, or path match

        cursor.execute(

            "SELECT * FROM documents WHERE ? LIKE id || '%' OR id LIKE ? || '%' OR name LIKE ? OR path LIKE ? LIMIT 1",

            (doc_id, doc_id, f"%{doc_id}%", f"%{doc_id}%")

        )

        row = cursor.fetchone()

    conn.close()

    if not row:

        return None

    return dict(row)





def list_documents() -> list[dict[str, Any]]:

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM documents ORDER BY uploaded_at DESC")

    rows = cursor.fetchall()

    conn.close()

    return [dict(r) for r in rows]





def delete_document(doc_id: str) -> bool:

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))

    deleted = cursor.rowcount > 0

    conn.commit()

    conn.close()

    return deleted





# ==================== RUN OPERATIONS ====================



def save_run(
    run_id: str,
    question: str,
    status: str = "processing",
    final_report: Optional[str] = None,
    verified_claims: list = None,
    rejected_claims: list = None,
    evidence_spans: list = None,
    reasoning_trace: list = None,
    error: Optional[str] = None,
    document_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()

    if not session_id:
        session_id = f"session_{uuid.uuid4().hex[:10]}"

    cursor.execute("SELECT id FROM chat_sessions WHERE id = ?", (session_id,))
    if not cursor.fetchone():
        clean_title = (question or "Cuộc trò chuyện").strip()
        if clean_title.startswith("/"):
            clean_title = clean_title.lstrip("/").replace("-", " ").capitalize()
        if len(clean_title) > 60:
            clean_title = clean_title[:57] + "..."
        cursor.execute("""
            INSERT INTO chat_sessions (id, title, created_at, updated_at, document_id)
            VALUES (?, ?, ?, ?, ?)
        """, (session_id, clean_title or "Đoạn nghiên cứu", now, now, document_id))
    else:
        cursor.execute("""
            UPDATE chat_sessions SET updated_at = ?, document_id = COALESCE(?, document_id) WHERE id = ?
        """, (now, document_id, session_id))

    cursor.execute("""
        INSERT OR REPLACE INTO research_runs (
            id, question, status, created_at, final_report,
            verified_claims, rejected_claims, evidence_spans, reasoning_trace, error, document_id, session_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        run_id,
        question,
        status,
        now,
        final_report,
        json.dumps(verified_claims or [], ensure_ascii=False),
        json.dumps(rejected_claims or [], ensure_ascii=False),
        json.dumps(evidence_spans or [], ensure_ascii=False),
        json.dumps(reasoning_trace or [], ensure_ascii=False),
        error,
        document_id,
        session_id,
    ))

    conn.commit()
    conn.close()

    return get_run(run_id)





def update_run_result(

    run_id: str,

    status: str,

    final_report: str = "",

    verified_claims: list = None,

    rejected_claims: list = None,

    evidence_spans: list = None,

    reasoning_trace: list = None,

    error: Optional[str] = None,

):

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("""

        UPDATE research_runs SET

            status = ?,

            final_report = ?,

            verified_claims = ?,

            rejected_claims = ?,

            evidence_spans = ?,

            reasoning_trace = ?,

            error = ?

        WHERE id = ?

    """, (

        status,

        final_report,

        json.dumps(verified_claims or [], ensure_ascii=False),

        json.dumps(rejected_claims or [], ensure_ascii=False),

        json.dumps(evidence_spans or [], ensure_ascii=False),

        json.dumps(reasoning_trace or [], ensure_ascii=False),

        error,

        run_id,

    ))

    conn.commit()

    conn.close()





def get_run(run_id: str) -> Optional[dict[str, Any]]:

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM research_runs WHERE id = ?", (run_id,))

    row = cursor.fetchone()

    conn.close()

    if not row:

        return None

    d = dict(row)

    d["verified_claims"] = json.loads(d.get("verified_claims") or "[]")

    d["rejected_claims"] = json.loads(d.get("rejected_claims") or "[]")

    d["evidence_spans"] = json.loads(d.get("evidence_spans") or "[]")

    d["reasoning_trace"] = json.loads(d.get("reasoning_trace") or "[]")

    return d





def list_runs() -> list[dict[str, Any]]:

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM research_runs ORDER BY created_at DESC, rowid DESC")

    rows = cursor.fetchall()

    conn.close()

    result = []

    for row in rows:

        d = dict(row)

        d["verified_claims"] = json.loads(d.get("verified_claims") or "[]")

        d["rejected_claims"] = json.loads(d.get("rejected_claims") or "[]")

        d["evidence_spans"] = json.loads(d.get("evidence_spans") or "[]")

        d["reasoning_trace"] = json.loads(d.get("reasoning_trace") or "[]")

        result.append(d)

    return result





def delete_run(run_id: str) -> bool:

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("DELETE FROM research_runs WHERE id = ?", (run_id,))

    deleted = cursor.rowcount > 0

    conn.commit()

    conn.close()

    return deleted


def clear_all_runs() -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM research_runs")
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    return deleted







# ==================== CHAT SESSION OPERATIONS ====================

def create_session(session_id: str, title: str, document_id: Optional[str] = None) -> dict[str, Any]:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()
    cursor.execute("""
        INSERT OR REPLACE INTO chat_sessions (id, title, created_at, updated_at, document_id)
        VALUES (?, ?, ?, ?, ?)
    """, (session_id, title, now, now, document_id))
    conn.commit()
    conn.close()
    return get_session(session_id)


def get_session(session_id: str) -> Optional[dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
    row = cursor.fetchone()
    conn.close()
    if not row:
        return None
    return dict(row)


def list_sessions() -> list[dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT s.id, s.title, s.created_at, s.updated_at, s.document_id,
               COUNT(r.id) as message_count
        FROM chat_sessions s
        LEFT JOIN research_runs r ON s.id = r.session_id
        GROUP BY s.id
        ORDER BY s.updated_at DESC, s.created_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]


def rename_session(session_id: str, new_title: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    now = datetime.datetime.now().isoformat()
    cursor.execute("""
        UPDATE chat_sessions SET title = ?, updated_at = ? WHERE id = ?
    """, (new_title.strip(), now, session_id))
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def delete_session(session_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
    cursor.execute("DELETE FROM research_runs WHERE session_id = ?", (session_id,))
    deleted = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return deleted


def clear_all_sessions() -> int:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM chat_sessions")
    cursor.execute("DELETE FROM research_runs")
    count = cursor.rowcount
    conn.commit()
    conn.close()
    return count


def get_session_runs(session_id: str) -> list[dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM research_runs WHERE session_id = ? ORDER BY created_at ASC", (session_id,))
    rows = cursor.fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["verified_claims"] = json.loads(d.get("verified_claims") or "[]")
        d["rejected_claims"] = json.loads(d.get("rejected_claims") or "[]")
        d["evidence_spans"] = json.loads(d.get("evidence_spans") or "[]")
        d["reasoning_trace"] = json.loads(d.get("reasoning_trace") or "[]")
        result.append(d)
    return result


# ==================== LLM CACHE OPERATIONS ====================



def get_cached_llm(prompt_hash: str) -> Optional[str]:

    """Retrieves cached LLM completion by prompt hash."""

    conn = get_db_connection()

    cursor = conn.cursor()

    cursor.execute("SELECT response FROM llm_cache WHERE prompt_hash = ?", (prompt_hash,))

    row = cursor.fetchone()

    conn.close()

    return row["response"] if row else None





def set_cached_llm(prompt_hash: str, prompt: str, response: str):

    """Caches LLM response to eliminate redundant API calls."""

    conn = get_db_connection()

    cursor = conn.cursor()

    now = datetime.datetime.now().isoformat()

    cursor.execute("""

        INSERT OR REPLACE INTO llm_cache (prompt_hash, prompt, response, created_at)

        VALUES (?, ?, ?, ?)

    """, (prompt_hash, prompt, response, now))

    conn.commit()

    conn.close()







def log_api_usage(provider: str, model: str, endpoint: str = "generateContent", timestamp: Optional[str] = None):
    """Logs an external API request (e.g. Gemini API) into api_usage_logs for daily quota tracking."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        ts = timestamp or datetime.datetime.now().isoformat()
        cursor.execute(
            "INSERT INTO api_usage_logs (provider, model, endpoint, timestamp) VALUES (?, ?, ?, ?)",
            (provider, model, endpoint, ts),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"Failed to log API usage for {provider}/{model}: {e}")


# Initialize on import

init_db()

