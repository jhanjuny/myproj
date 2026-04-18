"""
SQLite database layer.
Schema:
  subjects(id, name, color, created_at)
  chapters(id, subject_id, title, order_idx, created_at)
  notes(id, chapter_id, title, markdown, highlights_json, source_files_json,
        created_at, updated_at)
"""
import sqlite3
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from src.config import DB_PATH


# ── Connection helper ─────────────────────────────────────────────────────────

@contextmanager
def get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation ────────────────────────────────────────────────────────────

def init_db():
    with get_conn() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS subjects (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT    NOT NULL UNIQUE,
            color       TEXT    NOT NULL DEFAULT '#667eea',
            created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
        );

        CREATE TABLE IF NOT EXISTS chapters (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_id  INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
            title       TEXT    NOT NULL,
            order_idx   INTEGER NOT NULL DEFAULT 0,
            created_at  REAL    NOT NULL DEFAULT (unixepoch('now'))
        );

        CREATE TABLE IF NOT EXISTS notes (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_id        INTEGER NOT NULL REFERENCES chapters(id) ON DELETE CASCADE,
            title             TEXT    NOT NULL,
            markdown          TEXT    NOT NULL DEFAULT '',
            highlights_json   TEXT    NOT NULL DEFAULT '[]',
            source_files_json TEXT    NOT NULL DEFAULT '[]',
            created_at        REAL    NOT NULL DEFAULT (unixepoch('now')),
            updated_at        REAL    NOT NULL DEFAULT (unixepoch('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_chapters_subject ON chapters(subject_id);
        CREATE INDEX IF NOT EXISTS idx_notes_chapter   ON notes(chapter_id);
        """)


# ── Subject CRUD ──────────────────────────────────────────────────────────────

def create_subject(name: str, color: str = "#667eea") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT OR IGNORE INTO subjects (name, color) VALUES (?, ?)",
            (name, color),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute(
            "SELECT id FROM subjects WHERE name = ?", (name,)
        ).fetchone()
        return row["id"]


def get_subjects() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM subjects ORDER BY name"
        ).fetchall()
        return [dict(r) for r in rows]


def rename_subject(subject_id: int, new_name: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE subjects SET name = ? WHERE id = ?",
            (new_name, subject_id),
        )


def delete_subject(subject_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM subjects WHERE id = ?", (subject_id,))


# ── Chapter CRUD ──────────────────────────────────────────────────────────────

def create_chapter(subject_id: int, title: str, order_idx: int = 0) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            "INSERT INTO chapters (subject_id, title, order_idx) VALUES (?, ?, ?)",
            (subject_id, title, order_idx),
        )
        return cur.lastrowid


def get_chapters(subject_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM chapters WHERE subject_id = ? ORDER BY order_idx, id",
            (subject_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def rename_chapter(chapter_id: int, new_title: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE chapters SET title = ? WHERE id = ?",
            (new_title, chapter_id),
        )


def reorder_chapter(chapter_id: int, order_idx: int):
    with get_conn() as conn:
        conn.execute(
            "UPDATE chapters SET order_idx = ? WHERE id = ?",
            (order_idx, chapter_id),
        )


def delete_chapter(chapter_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM chapters WHERE id = ?", (chapter_id,))


# ── Note CRUD ─────────────────────────────────────────────────────────────────

def create_note(
    chapter_id: int,
    title: str,
    markdown: str,
    source_files: list[str] | None = None,
) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO notes (chapter_id, title, markdown, source_files_json)
               VALUES (?, ?, ?, ?)""",
            (
                chapter_id,
                title,
                markdown,
                json.dumps(source_files or []),
            ),
        )
        return cur.lastrowid


def get_note(note_id: int) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM notes WHERE id = ?", (note_id,)
        ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["highlights"] = json.loads(d["highlights_json"])
        d["source_files"] = json.loads(d["source_files_json"])
        return d


def get_notes(chapter_id: int) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, created_at, updated_at FROM notes "
            "WHERE chapter_id = ? ORDER BY created_at",
            (chapter_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def update_note_markdown(note_id: int, markdown: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE notes SET markdown = ?, updated_at = ? WHERE id = ?",
            (markdown, time.time(), note_id),
        )


def update_note_highlights(note_id: int, highlights: list[dict]):
    with get_conn() as conn:
        conn.execute(
            "UPDATE notes SET highlights_json = ?, updated_at = ? WHERE id = ?",
            (json.dumps(highlights), time.time(), note_id),
        )


def rename_note(note_id: int, new_title: str):
    with get_conn() as conn:
        conn.execute(
            "UPDATE notes SET title = ?, updated_at = ? WHERE id = ?",
            (new_title, time.time(), note_id),
        )


def delete_note(note_id: int):
    with get_conn() as conn:
        conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))


def search_notes(query: str) -> list[dict]:
    """Full-text search across note titles and markdown."""
    with get_conn() as conn:
        pattern = f"%{query}%"
        rows = conn.execute(
            """SELECT n.id, n.title, n.chapter_id, c.title AS chapter_title,
                      s.id AS subject_id, s.name AS subject_name
               FROM notes n
               JOIN chapters c ON n.chapter_id = c.id
               JOIN subjects s ON c.subject_id = s.id
               WHERE n.title LIKE ? OR n.markdown LIKE ?
               ORDER BY n.updated_at DESC
               LIMIT 100""",
            (pattern, pattern),
        ).fetchall()
        return [dict(r) for r in rows]
