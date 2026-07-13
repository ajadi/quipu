"""hub.store — SQLite WAL adapter. All SQL lives here; no SQL in route handlers.

Schema per ## database-architect: hub_oplog with AUTOINCREMENT ingest_seq,
UNIQUE(blinded_project_id, entry_id) dedup, WAL+NORMAL, idx_hub_oplog_pull.
"""

from __future__ import annotations

import base64
import sqlite3


# ---------------------------------------------------------------------------
# Connection + schema init
# ---------------------------------------------------------------------------

def get_connection(db_path: str) -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection. Call once at startup."""
    # check_same_thread=False is safe here ONLY because all DB access runs on the
    # single asyncio event-loop thread with NO await between connection acquisition
    # and query completion (sqlite3 calls are synchronous inside async handlers).
    # If a future contributor introduces an await between getting the connection
    # and finishing the SQL, this must be revisited — add a lock or use a
    # per-request connection instead.
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Idempotent schema init. CREATE TABLE/INDEX IF NOT EXISTS."""
    conn.executescript("""
        PRAGMA journal_mode=WAL;
        PRAGMA synchronous=NORMAL;
        CREATE TABLE IF NOT EXISTS hub_oplog (
            ingest_seq         INTEGER PRIMARY KEY AUTOINCREMENT,
            blinded_project_id TEXT    NOT NULL,
            entry_id           TEXT    NOT NULL,
            client_id          TEXT    NOT NULL,
            sequence_no        INTEGER NOT NULL,
            op                 TEXT    NOT NULL CHECK (op IN ('upsert', 'invalidate')),
            record_id          TEXT    NOT NULL,
            ts                 TEXT    NOT NULL,
            payload            BLOB    NOT NULL,
            CONSTRAINT hub_oplog_dedup UNIQUE (blinded_project_id, entry_id)
        );
        CREATE INDEX IF NOT EXISTS idx_hub_oplog_pull
            ON hub_oplog (blinded_project_id, ingest_seq);
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Adapter functions
# ---------------------------------------------------------------------------

def append(
    conn: sqlite3.Connection,
    blinded_project_id: str,
    entries: list[dict],
) -> None:
    """Ingest a batch of hub-visible entry dicts.

    payload arrives as a base64 string; decoded to bytes before INSERT.
    ingest_seq is server-assigned (AUTOINCREMENT).
    INSERT OR IGNORE: re-pushed entries with existing (blinded_project_id, entry_id)
    are silently skipped (idempotent).
    """
    sql = """
        INSERT OR IGNORE INTO hub_oplog
            (blinded_project_id, entry_id, client_id, sequence_no,
             op, record_id, ts, payload)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with conn:
        for entry in entries:
            payload_bytes = base64.b64decode(entry["payload"])
            conn.execute(sql, (
                blinded_project_id,
                entry["entry_id"],
                entry["client_id"],
                entry["sequence_no"],
                entry["op"],
                entry["record_id"],
                entry["ts"],
                payload_bytes,
            ))


def get_cursor(conn: sqlite3.Connection, blinded_project_id: str) -> str:
    """Return the current cursor for a project without fetching any payloads.

    Cursor contract: "0" when no rows exist for the project; str(MAX(ingest_seq))
    otherwise.  Equivalent to what read_since(None) would return as next_cursor,
    but O(1) instead of O(n).
    """
    row = conn.execute(
        "SELECT MAX(ingest_seq) FROM hub_oplog WHERE blinded_project_id = ?",
        (blinded_project_id,),
    ).fetchone()
    # MAX returns NULL when no rows match; sqlite3.Row index 0 gives None.
    max_seq = row[0]
    return str(max_seq) if max_seq is not None else "0"


def read_since(
    conn: sqlite3.Connection,
    blinded_project_id: str,
    cursor: str | None,
    limit: int,
) -> tuple[list[dict], str, bool]:
    """Return a page of entries after cursor, the next cursor, and has_more.

    offset = int(cursor) if cursor else 0.
    Fetches up to `limit + 1` rows; if that yields more than `limit`, the page
    is trimmed to `limit` rows and has_more=True (no second COUNT query needed).
    next_cursor = str(max ingest_seq in the RETURNED page) if non-empty,
    else (cursor or "0") — unchanged from prior behavior.
    payload re-encoded as base64 string for JSON wire.
    ingest_seq is never exposed in the returned entry dicts.
    """
    offset = int(cursor) if cursor else 0
    sql = """
        SELECT ingest_seq, entry_id, client_id, sequence_no,
               op, record_id, ts, payload
        FROM hub_oplog
        WHERE blinded_project_id = ? AND ingest_seq > ?
        ORDER BY ingest_seq
        LIMIT ?
    """
    rows = conn.execute(sql, (blinded_project_id, offset, limit + 1)).fetchall()

    has_more = len(rows) > limit
    if has_more:
        rows = rows[:limit]

    entries: list[dict] = []
    last_seq: int | None = None
    for row in rows:
        last_seq = row["ingest_seq"]
        entries.append({
            "entry_id": row["entry_id"],
            "client_id": row["client_id"],
            "sequence_no": row["sequence_no"],
            "op": row["op"],
            "record_id": row["record_id"],
            "blinded_project_id": blinded_project_id,
            "ts": row["ts"],
            "payload": base64.b64encode(row["payload"]).decode("ascii"),
        })

    next_cursor = str(last_seq) if last_seq is not None else (cursor or "0")
    return entries, next_cursor, has_more
