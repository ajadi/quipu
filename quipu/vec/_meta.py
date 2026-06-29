"""ember_vec_meta DDL and helper accessors.

Table: ember_vec_meta(key TEXT PK, status TEXT, dim INTEGER, model TEXT, built_at TEXT)
Stores vec build lifecycle state — NOT PRAGMA user_version (which is schema-owned).

Status values: 'building' | 'complete'
"""

from __future__ import annotations

import sqlite3

_DDL = """
-- wire-value: on-disk table name kept as "ember_vec_meta" — renaming makes existing vec indexes unreadable
CREATE TABLE IF NOT EXISTS ember_vec_meta (
    key       TEXT NOT NULL,
    status    TEXT NOT NULL,
    dim       INTEGER,
    model     TEXT,
    built_at  TEXT,
    CONSTRAINT ember_vec_meta_pk PRIMARY KEY (key)
);
"""


def ensure_meta_table(conn: sqlite3.Connection) -> None:
    """Create ember_vec_meta table if it does not exist."""
    conn.execute(_DDL)
    conn.commit()


def get_build_status(conn: sqlite3.Connection) -> str | None:
    """Return the 'build' row status, or None if no row exists."""
    try:
        row = conn.execute(
            "SELECT status FROM ember_vec_meta WHERE key = 'build'"
        ).fetchone()
        return row[0] if row else None
    except sqlite3.OperationalError:
        # Table doesn't exist yet.
        return None


def set_build_status(
    conn: sqlite3.Connection,
    status: str,
    *,
    dim: int | None = None,
    model: str | None = None,
) -> None:
    """Upsert the 'build' row with the given status."""
    conn.execute(
        """
        INSERT INTO ember_vec_meta (key, status, dim, model, built_at)
        VALUES ('build', ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        ON CONFLICT(key) DO UPDATE SET
            status   = excluded.status,
            dim      = COALESCE(excluded.dim,   ember_vec_meta.dim),
            model    = COALESCE(excluded.model, ember_vec_meta.model),
            built_at = excluded.built_at
        """,
        (status, dim, model),
    )


def is_build_complete(conn: sqlite3.Connection) -> bool:
    """Return True iff ember_vec_meta has key='build' with status='complete'."""
    return get_build_status(conn) == "complete"
