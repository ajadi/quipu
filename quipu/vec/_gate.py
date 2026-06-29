"""Availability gate for sqlite-vec extension.

try_load(conn): attempt to enable and load sqlite-vec. Returns True on success.
is_loaded(conn): check if vec_version() is callable (extension is live).

On ANY failure logs ONE warning (deduped via _warned module-level flag) and
returns False. No crash, no user prompt.
"""

from __future__ import annotations

import logging
import sqlite3

_logger = logging.getLogger(__name__)
_warned: bool = False


def try_load(conn: sqlite3.Connection) -> bool:
    """Enable sqlite extension loading and load sqlite-vec into *conn*.

    Catches (ImportError, AttributeError, sqlite3.OperationalError,
    sqlite3.NotSupportedError). On any failure logs ONE module-level
    warning and returns False.

    Returns True only when sqlite-vec is successfully loaded.
    """
    global _warned
    try:
        import sqlite_vec  # noqa: F401 — needed for sqlite_vec.load

        conn.enable_load_extension(True)
        try:
            sqlite_vec.load(conn)
        finally:
            # Always disable extension loading after attempt for security.
            try:
                conn.enable_load_extension(False)
            except Exception:
                pass
        return True
    except (ImportError, AttributeError, sqlite3.OperationalError,
            sqlite3.NotSupportedError) as exc:
        if not _warned:
            _warned = True
            _logger.warning(
                "sqlite-vec unavailable (%s: %s); staying on pure-Python cosine.",
                type(exc).__name__,
                exc,
            )
        return False


def is_loaded(conn: sqlite3.Connection) -> bool:
    """Return True if vec_version() is callable on *conn* (extension live)."""
    try:
        conn.execute("SELECT vec_version()").fetchone()
        return True
    except (sqlite3.OperationalError, sqlite3.DatabaseError):
        return False


def reset_warned() -> None:
    """Reset the one-time warning flag (for tests only)."""
    global _warned
    _warned = False
