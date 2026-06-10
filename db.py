"""
db.py
─────
SQLite layer for the Skin Lesion Classifier app.

Responsibilities:
  • Locate / create the database file and the user-uploads directory.
  • Provide a context-managed connection that auto-commits on success
    and auto-rollbacks on error.
  • Enforce foreign keys (off by default in SQLite — must be turned on
    per connection).
  • Define the schema for `users` and `predictions`.

Nothing in this module touches Streamlit. Keeping the data layer
framework-agnostic means it can be reused by tests, scripts, or a
future FastAPI/Flask version of the app.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# ── Paths ────────────────────────────────────────────────────────────────
# We anchor everything to this file's directory so the app works no matter
# where it's launched from (a common foot-gun: relative paths break the
# moment the user runs `streamlit run` from a different cwd).
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
UPLOADS_DIR  = DATA_DIR / "uploads"
DB_PATH      = DATA_DIR / "app.db"


def _ensure_dirs() -> None:
    """Make sure `data/` and `data/uploads/` exist before we touch the DB."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """
    Context-managed SQLite connection.

    Usage:
        with get_connection() as conn:
            conn.execute("INSERT INTO ...")

    Guarantees:
      • Foreign keys are enforced (`PRAGMA foreign_keys = ON`).
      • Rows behave like dicts (`row["username"]`) via sqlite3.Row.
      • Commit on clean exit, rollback on exception, always close.
    """
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Create tables if they don't exist. Idempotent — safe to call on every
    app startup.

    Schema notes:
      • `users.username` is UNIQUE + COLLATE NOCASE so "Alice" and "alice"
        cannot both register. We don't want two accounts that look
        identical to a human eye.
      • `users.password_hash` is the bcrypt output (a string that
        embeds the algorithm + cost + salt + digest). NEVER store
        plaintext passwords here.
      • `predictions.user_id` has ON DELETE CASCADE so removing a user
        also removes their history rows — no dangling foreign keys.
      • The composite index on (user_id, created_at DESC) makes the
        "show me my history newest-first" query O(log N) instead of a
        full table scan.
    """
    _ensure_dirs()
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id            INTEGER  PRIMARY KEY AUTOINCREMENT,
                username      TEXT     NOT NULL UNIQUE COLLATE NOCASE,
                password_hash TEXT     NOT NULL,
                created_at    TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS predictions (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id            INTEGER NOT NULL,
                image_path         TEXT    NOT NULL,
                original_filename  TEXT    NOT NULL,
                predicted_class    TEXT    NOT NULL,
                confidence         REAL    NOT NULL,
                probabilities_json TEXT    NOT NULL,
                created_at         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_predictions_user_created
                ON predictions(user_id, created_at DESC);
            """
        )


# ── Query helpers ────────────────────────────────────────────────────────
# Kept here (not in auth.py / classifier.py) so the SQL lives in one place.
# Every query uses parameter substitution (`?`) — never f-string
# interpolation — to prevent SQL injection.

def insert_prediction(
    user_id: int,
    image_path: str,
    original_filename: str,
    predicted_class: str,
    confidence: float,
    probabilities_json: str,
) -> int:
    """Store one prediction row. Returns the new row id."""
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO predictions
                (user_id, image_path, original_filename,
                 predicted_class, confidence, probabilities_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (user_id, image_path, original_filename,
             predicted_class, confidence, probabilities_json),
        )
        return int(cur.lastrowid)


def list_predictions_for_user(user_id: int, limit: int = 100) -> list[sqlite3.Row]:
    """Return the user's predictions, newest first."""
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, image_path, original_filename,
                   predicted_class, confidence,
                   probabilities_json, created_at
            FROM predictions
            WHERE user_id = ?
            ORDER BY created_at DESC, id DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()


def get_prediction(user_id: int, prediction_id: int) -> sqlite3.Row | None:
    """
    Look up a single prediction, but ONLY if it belongs to `user_id`.
    The WHERE user_id check is mandatory — without it, a logged-in user
    could view another user's history by guessing IDs (IDOR vulnerability).
    """
    with get_connection() as conn:
        return conn.execute(
            """
            SELECT id, image_path, original_filename,
                   predicted_class, confidence,
                   probabilities_json, created_at
            FROM predictions
            WHERE user_id = ? AND id = ?
            """,
            (user_id, prediction_id),
        ).fetchone()


def delete_prediction(user_id: int, prediction_id: int) -> bool:
    """
    Delete one of the user's own prediction rows. Returns True if a row
    was deleted. The (user_id, id) filter again prevents cross-user
    deletion.
    """
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM predictions WHERE user_id = ? AND id = ?",
            (user_id, prediction_id),
        )
        return cur.rowcount > 0
