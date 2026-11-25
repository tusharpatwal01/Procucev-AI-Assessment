import sqlite3
from datetime import datetime, timezone
from typing import List, Optional

DB_PATH = "transcriptions.db"


def _get_conn():
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with _get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                language TEXT,
                created_at TEXT NOT NULL
            )
            """)


def create_transcription(text: str, language: str = "auto") -> int:
    with _get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO transcriptions (text, language, created_at) VALUES (?, ?, ?)",
            (
                text,
                language,
                datetime.now(timezone.utc).isoformat().replace("+00:00", ""),
            ),
        )
        return cur.lastrowid


def get_all_transcriptions() -> List[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM transcriptions ORDER BY created_at DESC"
        ).fetchall()


def get_transcription(tid: int) -> Optional[sqlite3.Row]:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT * FROM transcriptions WHERE id = ?", (tid,)
        ).fetchone()


def update_transcription(tid: int, text: str, language: str = None):
    with _get_conn() as conn:
        if language:
            conn.execute(
                "UPDATE transcriptions SET text = ?, language = ? WHERE id = ?",
                (text, language, tid),
            )
        else:
            conn.execute("UPDATE transcriptions SET text = ? WHERE id = ?", (text, tid))


def delete_transcription(tid: int):
    with _get_conn() as conn:
        conn.execute("DELETE FROM transcriptions WHERE id = ?", (tid,))
