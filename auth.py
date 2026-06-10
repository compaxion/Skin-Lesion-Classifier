"""
auth.py
Authentication, registration, and bcrypt hashing.
"""

from __future__ import annotations

import re
import sqlite3

import bcrypt

from db import get_connection

# Validation rules
USERNAME_PATTERN  = re.compile(r"^[A-Za-z0-9_]{3,32}$")
MIN_PASSWORD_LEN  = 8
MAX_PASSWORD_LEN  = 128
BCRYPT_COST       = 12

# Dummy hash to prevent timing attacks
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt(rounds=BCRYPT_COST))


class AuthError(Exception):
    """Authentication error."""


# Internal helpers
def _validate_username(username: str) -> None:
    if not username or not username.strip():
        raise AuthError("Username cannot be empty.")
    if not USERNAME_PATTERN.match(username):
        raise AuthError(
            "Username must be 3–32 characters and contain only "
            "letters, digits, or underscores."
        )


def _validate_password(password: str) -> None:
    if not password:
        raise AuthError("Password cannot be empty.")
    if len(password) < MIN_PASSWORD_LEN:
        raise AuthError(
            f"Password must be at least {MIN_PASSWORD_LEN} characters long."
        )
    if len(password) > MAX_PASSWORD_LEN:
        raise AuthError(
            f"Password must be at most {MAX_PASSWORD_LEN} characters long."
        )


def _hash_password(password: str) -> str:
    """Hash password with bcrypt."""
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=BCRYPT_COST),
    ).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    """Verify password against hash."""
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


# Public API
def register_user(username: str, password: str, password_confirm: str) -> int:
    """Register a new user."""
    username = (username or "").strip()
    _validate_username(username)
    _validate_password(password)

    if password != password_confirm:
        raise AuthError("Passwords do not match.")

    pw_hash = _hash_password(password)

    try:
        with get_connection() as conn:
            cur = conn.execute(
                "INSERT INTO users (username, password_hash) VALUES (?, ?)",
                (username, pw_hash),
            )
            return int(cur.lastrowid)
    except sqlite3.IntegrityError:
        # Username already exists
        raise AuthError("This username is already taken.")


def authenticate_user(username: str, password: str) -> dict:
    """Verify login credentials."""
    if not username or not password:
        raise AuthError("Username and password are required.")

    username = username.strip()

    with get_connection() as conn:
        row = conn.execute(
            "SELECT id, username, password_hash FROM users "
            "WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()

    if row is None:
        # Prevent timing attacks for invalid usernames
        bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
        raise AuthError("Invalid username or password.")

    if not _verify_password(password, row["password_hash"]):
        raise AuthError("Invalid username or password.")

    return {"id": int(row["id"]), "username": row["username"]}
