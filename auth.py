"""
auth.py
───────
Authentication primitives: registration, login, password hashing.

Design rules:
  • Passwords are NEVER stored, logged, or compared in plaintext.
  • Hashing uses bcrypt (algorithm + per-password random salt + cost
    factor baked into the output string). Cost 12 ≈ ~250 ms per hash
    on a modern CPU — slow enough to make offline brute force painful,
    fast enough that login feels instant.
  • All SQL goes through parameterized queries in db.py. We never
    build SQL strings with f-strings or `+`.
  • On a wrong username we still perform a bcrypt comparison against a
    dummy hash, so an attacker cannot distinguish "user does not
    exist" from "wrong password" via response timing.
  • Public functions raise `AuthError` on failure with a user-safe
    message; the Streamlit layer catches that and shows it via
    `st.error`.
"""

from __future__ import annotations

import re
import sqlite3

import bcrypt

from db import get_connection

# ── Validation rules ─────────────────────────────────────────────────────
USERNAME_PATTERN  = re.compile(r"^[A-Za-z0-9_]{3,32}$")
MIN_PASSWORD_LEN  = 8
MAX_PASSWORD_LEN  = 128  # bcrypt itself truncates at 72 bytes; we cap higher
BCRYPT_COST       = 12

# A pre-computed bcrypt hash of a random string. We compare against this
# when the username doesn't exist so that response time is similar to a
# real failed login. See `authenticate_user`.
_DUMMY_HASH = bcrypt.hashpw(b"not-a-real-password", bcrypt.gensalt(rounds=BCRYPT_COST))


class AuthError(Exception):
    """Raised for any user-facing authentication failure."""


# ── Internal helpers ─────────────────────────────────────────────────────
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
    """Return the bcrypt hash as a UTF-8 string ready to store."""
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt(rounds=BCRYPT_COST),
    ).decode("utf-8")


def _verify_password(password: str, password_hash: str) -> bool:
    """
    Constant-time-ish bcrypt comparison. Returns False on any malformed
    input rather than raising — we don't want a bad row in the DB to
    crash the login screen.
    """
    try:
        return bcrypt.checkpw(
            password.encode("utf-8"),
            password_hash.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


# ── Public API ───────────────────────────────────────────────────────────
def register_user(username: str, password: str, password_confirm: str) -> int:
    """
    Create a new user. Returns the new user's id.

    Raises AuthError if:
      • Username or password fail validation.
      • The two password fields do not match.
      • The username is already taken (UNIQUE constraint violation).
    """
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
        # Triggered by the UNIQUE constraint on username (case-insensitive).
        raise AuthError("This username is already taken.")


def authenticate_user(username: str, password: str) -> dict:
    """
    Verify a username + password. Returns {"id": ..., "username": ...}
    on success, or raises AuthError on any failure.

    The same generic error message is used for "unknown user" and "wrong
    password" so an attacker cannot enumerate valid usernames.
    """
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
        # Burn ~one bcrypt comparison's worth of time so the response
        # time for "no such user" matches "wrong password".
        bcrypt.checkpw(password.encode("utf-8"), _DUMMY_HASH)
        raise AuthError("Invalid username or password.")

    if not _verify_password(password, row["password_hash"]):
        raise AuthError("Invalid username or password.")

    return {"id": int(row["id"]), "username": row["username"]}
