"""Build and lifecycle helpers for the atoms_vec virtual table.

atoms_vec is a sqlite-vec vec0 virtual table whose rowid matches atoms.rowid.
It is a derived cache — BLOB embeddings in atoms remain the source of truth.

Public API:
    atom_count(conn) -> int
    crossed(conn, threshold) -> bool
    build(conn) -> None          # idempotent; WAL-safe single txn
    install_triggers(conn) -> None
    drop_index(conn) -> None
"""

from __future__ import annotations

import logging
import sqlite3

from quipu.vec._meta import (
    ensure_meta_table,
    get_build_status,
    is_build_complete,
    set_build_status,
)

_logger = logging.getLogger(__name__)

_VEC_TABLE_DDL = (
    "CREATE VIRTUAL TABLE IF NOT EXISTS atoms_vec "
    "USING vec0(embedding float[384])"
)

# Triggers keep atoms_vec in sync with atoms after the initial build.
_TRIGGER_INSERT = """
CREATE TRIGGER IF NOT EXISTS atoms_vec_insert
AFTER INSERT ON atoms
FOR EACH ROW
WHEN NEW.embedding IS NOT NULL
BEGIN
    INSERT OR REPLACE INTO atoms_vec(rowid, embedding)
    VALUES (NEW.rowid, NEW.embedding);
END;
"""

_TRIGGER_UPDATE = """
CREATE TRIGGER IF NOT EXISTS atoms_vec_update
AFTER UPDATE OF embedding ON atoms
FOR EACH ROW
BEGIN
    DELETE FROM atoms_vec WHERE rowid = OLD.rowid;
    INSERT OR REPLACE INTO atoms_vec(rowid, embedding)
    SELECT NEW.rowid, NEW.embedding WHERE NEW.embedding IS NOT NULL;
END;
"""

_TRIGGER_DELETE = """
CREATE TRIGGER IF NOT EXISTS atoms_vec_delete
AFTER DELETE ON atoms
FOR EACH ROW
BEGIN
    DELETE FROM atoms_vec WHERE rowid = OLD.rowid;
END;
"""


def atom_count(conn: sqlite3.Connection) -> int:
    """Return the total number of rows in atoms (O(1) via B-tree count)."""
    row = conn.execute("SELECT count(*) FROM atoms").fetchone()
    return row[0] if row else 0


def crossed(conn: sqlite3.Connection, threshold: int) -> bool:
    """Return True if atom_count >= threshold."""
    return atom_count(conn) >= threshold


def build(conn: sqlite3.Connection) -> None:
    """Create atoms_vec and bulk-insert embeddings from atoms.

    Idempotent: safe to call multiple times.
    - If status == 'complete', returns immediately (no-op).
    - Otherwise: creates table, marks 'building', bulk-inserts
      (INSERT OR REPLACE), installs triggers, marks 'complete'.
    - Single WAL transaction: readers not blocked; crash before commit
      leaves status='building' so next call re-runs idempotently.
    """
    ensure_meta_table(conn)

    if is_build_complete(conn):
        return

    _logger.info("Building atoms_vec index…")

    # Single transaction: WAL allows concurrent readers.
    with conn:
        # Create virtual table (idempotent).
        conn.execute(_VEC_TABLE_DDL)

        # Mark building (idempotent upsert).
        set_build_status(conn, "building", dim=384)

        # Bulk-insert all existing embeddings — INSERT OR REPLACE handles
        # re-runs after a crash (idempotent).
        conn.execute(
            """
            INSERT OR REPLACE INTO atoms_vec(rowid, embedding)
            SELECT rowid, embedding
            FROM   atoms
            WHERE  embedding IS NOT NULL
            """
        )

        # Install triggers for ongoing sync.
        conn.execute(_TRIGGER_INSERT)
        conn.execute(_TRIGGER_UPDATE)
        conn.execute(_TRIGGER_DELETE)

        # Mark complete — only reachable on success.
        set_build_status(conn, "complete", dim=384)

    _logger.info("atoms_vec index build complete.")


def install_triggers(conn: sqlite3.Connection) -> None:
    """Install (or re-install) all three sync triggers.

    Idempotent — uses IF NOT EXISTS. Exposed separately so tests can
    verify trigger DDL independently of the full build.
    """
    conn.execute(_TRIGGER_INSERT)
    conn.execute(_TRIGGER_UPDATE)
    conn.execute(_TRIGGER_DELETE)
    conn.commit()


def drop_index(conn: sqlite3.Connection) -> None:
    """Drop atoms_vec virtual table, triggers, and meta row.

    After this call, ensure_index will rebuild from scratch on next call.
    BLOBs in atoms are never touched.

    Safe to call on a fresh DB where build() was never run — ensure_meta_table
    is called first so the DELETE does not raise OperationalError.
    """
    ensure_meta_table(conn)
    conn.execute("DROP TRIGGER IF EXISTS atoms_vec_insert")
    conn.execute("DROP TRIGGER IF EXISTS atoms_vec_update")
    conn.execute("DROP TRIGGER IF EXISTS atoms_vec_delete")
    conn.execute("DROP TABLE IF EXISTS atoms_vec")
    conn.execute("DELETE FROM ember_vec_meta WHERE key = 'build'")  # wire-value: ember_vec_meta is the kept on-disk table name (see vec/_meta.py)
    conn.commit()
