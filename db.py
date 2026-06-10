"""
db.py
SQLite data layer.
Handles connections, schema, and queries.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

# Paths
BASE_DIR     = Path(__file__).resolve().parent
DATA_DIR     = BASE_DIR / "data"
UPLOADS_DIR  = DATA_DIR / "uploads"
DB_PATH      = DATA_DIR / "app.db"


def _ensure_dirs() -> None:
    """Ensure data directories exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """Provide a context-managed SQLite connection."""
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
    """Initialize database tables."""
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


# Query helpers

def insert_prediction(
    user_id: int,
    image_path: str,
    original_filename: str,
    predicted_class: str,
    confidence: float,
    probabilities_json: str,
) -> int:
    """Insert a prediction record."""
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
    """Get user's predictions."""
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
    """Get a specific prediction for a user."""
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
    """Delete a prediction record."""
    with get_connection() as conn:
        cur = conn.execute(
            "DELETE FROM predictions WHERE user_id = ? AND id = ?",
            (user_id, prediction_id),
        )
        return cur.rowcount > 0
