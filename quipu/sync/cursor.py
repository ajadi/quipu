"""quipu.sync.cursor — schema-coupled read/upsert against sync_cursors.

Real columns from migration 0002:
    sync_cursors(blinded_project_id, direction, peer_id, last_seq,
                 last_cursor, updated_at)
UNIQUE(blinded_project_id, direction, peer_id) anchors the upsert.
Entirely local-only — never transmitted.
"""

from __future__ import annotations

import sqlite3


def read_cursor(
    conn: sqlite3.Connection,
    blinded_project_id: str,
    direction: str,
    peer_id: str,
) -> tuple[int, str | None]:
    """Return (last_seq, last_cursor) for the row, or (0, None) if absent."""
    row = conn.execute(
        """
        SELECT last_seq, last_cursor FROM sync_cursors
        WHERE blinded_project_id = ? AND direction = ? AND peer_id = ?
        """,
        (blinded_project_id, direction, peer_id),
    ).fetchone()
    if row is None:
        return 0, None
    return (row["last_seq"] or 0), row["last_cursor"]


def read_cursor_meta(
    conn: sqlite3.Connection,
    blinded_project_id: str,
    direction: str,
    peer_id: str,
) -> tuple[str | None, str | None]:
    """Return (last_cursor, updated_at) for the row, or (None, None) if absent."""
    row = conn.execute(
        """
        SELECT last_cursor, updated_at FROM sync_cursors
        WHERE blinded_project_id = ? AND direction = ? AND peer_id = ?
        """,
        (blinded_project_id, direction, peer_id),
    ).fetchone()
    if row is None:
        return None, None
    return row["last_cursor"], row["updated_at"]


def upsert_cursor(
    conn: sqlite3.Connection,
    blinded_project_id: str,
    direction: str,
    peer_id: str,
    last_seq: int,
    last_cursor: str | None,
) -> None:
    """Insert or update the cursor row for (project, direction, peer)."""
    conn.execute(
        """
        INSERT INTO sync_cursors
            (blinded_project_id, direction, peer_id, last_seq, last_cursor,
             updated_at)
        VALUES (?, ?, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
        ON CONFLICT (blinded_project_id, direction, peer_id)
        DO UPDATE SET
            last_seq = excluded.last_seq,
            last_cursor = excluded.last_cursor,
            updated_at = excluded.updated_at
        """,
        (blinded_project_id, direction, peer_id, last_seq, last_cursor),
    )
    conn.commit()
