"""quipu.sync.oplog_store — the ONLY schema-coupled module for oplog_entries.

Speaks the REAL columns from migration 0002:
    oplog_entries(entry_id, client_id, sequence_no, op, record_id,
                  blinded_project_id, ts, payload, source, pushed)

Apply path is INSERT OR IGNORE on UNIQUE(client_id, sequence_no) (+ UNIQUE(entry_id)):
replays / re-pulls / local-vs-pulled collisions collapse to one row.

The hub-visible dict (to_transport_dict / from_transport_dict) carries ONLY
hub-visible fields — never source/pushed, never plaintext, never the real
project_id or the key. The payload blob is base64-encoded so the dict is JSON-safe.
"""

from __future__ import annotations

import base64
import sqlite3

from quipu.oplog.entry import OplogEntry

# Hub-visible fields, in the order they appear on the wire.
_TRANSPORT_FIELDS = (
    "entry_id",
    "client_id",
    "sequence_no",
    "op",
    "record_id",
    "blinded_project_id",
    "ts",
    "payload",  # base64 string on the wire
)


def to_transport_dict(e: OplogEntry) -> dict:
    """Serialize an OplogEntry to a hub-visible, JSON-safe dict.

    NEVER includes source/pushed. payload is base64-encoded.
    """
    return {
        "entry_id": e.entry_id,
        "client_id": e.client_id,
        "sequence_no": e.sequence_no,
        "op": e.op,
        "record_id": e.record_id,
        "blinded_project_id": e.blinded_project_id,
        "ts": e.ts,
        "payload": base64.b64encode(e.payload).decode("ascii"),
    }


def from_transport_dict(d: dict) -> OplogEntry:
    """Reconstruct an OplogEntry from a hub-visible dict (source='remote')."""
    return OplogEntry(
        entry_id=d["entry_id"],
        client_id=d["client_id"],
        sequence_no=int(d["sequence_no"]),
        op=d["op"],
        record_id=d["record_id"],
        blinded_project_id=d["blinded_project_id"],
        ts=d["ts"],
        payload=base64.b64decode(d["payload"]),
        source="remote",
        pushed=False,
    )


def _row_to_entry(row: sqlite3.Row) -> OplogEntry:
    return OplogEntry(
        entry_id=row["entry_id"],
        client_id=row["client_id"],
        sequence_no=row["sequence_no"],
        op=row["op"],
        record_id=row["record_id"],
        blinded_project_id=row["blinded_project_id"],
        ts=row["ts"],
        payload=row["payload"],
        source=row["source"],
        pushed=bool(row["pushed"]),
    )


class OplogStore:
    """CRUD over oplog_entries. Holds a raw sqlite3.Connection (Store._conn)."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def append_local(self, e: OplogEntry) -> bool:
        """INSERT OR IGNORE a locally-produced entry (source='local', pushed=0).

        Returns True if a new row was inserted, False if it already existed.
        """
        cur = self._conn.execute(
            """
            INSERT OR IGNORE INTO oplog_entries
                (entry_id, client_id, sequence_no, op, record_id,
                 blinded_project_id, ts, payload, source, pushed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'local', 0)
            """,
            (e.entry_id, e.client_id, e.sequence_no, e.op, e.record_id,
             e.blinded_project_id, e.ts, e.payload),
        )
        self._conn.commit()
        return cur.rowcount > 0

    def next_sequence_no(self, client_id: str) -> int:
        """Next monotonic sequence_no for *client_id* (max+1, starts at 1)."""
        row = self._conn.execute(
            "SELECT MAX(sequence_no) AS m FROM oplog_entries WHERE client_id = ?",
            (client_id,),
        ).fetchone()
        return (row["m"] or 0) + 1

    def unpushed(self, blinded_project_id: str, client_id: str) -> list[OplogEntry]:
        """Local, not-yet-pushed entries for this project+client, ordered by seq."""
        rows = self._conn.execute(
            """
            SELECT * FROM oplog_entries
            WHERE blinded_project_id = ? AND client_id = ?
              AND source = 'local' AND pushed = 0
            ORDER BY sequence_no
            """,
            (blinded_project_id, client_id),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def mark_pushed(self, entry_ids: list[str]) -> None:
        """Flag the given entries as confirmed-pushed (pushed=1)."""
        if not entry_ids:
            return
        self._conn.executemany(
            "UPDATE oplog_entries SET pushed = 1 WHERE entry_id = ?",
            [(eid,) for eid in entry_ids],
        )
        self._conn.commit()

    def apply_remote(self, entries: list[OplogEntry]) -> list[str]:
        """INSERT OR IGNORE pulled entries (source='remote'). Return newly-inserted record_ids."""
        new_records: list[str] = []
        for e in entries:
            cur = self._conn.execute(
                """
                INSERT OR IGNORE INTO oplog_entries
                    (entry_id, client_id, sequence_no, op, record_id,
                     blinded_project_id, ts, payload, source, pushed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'remote', 0)
                """,
                (e.entry_id, e.client_id, e.sequence_no, e.op, e.record_id,
                 e.blinded_project_id, e.ts, e.payload),
            )
            if cur.rowcount > 0:
                new_records.append(e.record_id)
        self._conn.commit()
        return new_records

    def entries_for_record(
        self, blinded_project_id: str, record_id: str
    ) -> list[OplogEntry]:
        """All entries for a record, ordered ts ASC then entry_id ASC (deterministic)."""
        rows = self._conn.execute(
            """
            SELECT * FROM oplog_entries
            WHERE blinded_project_id = ? AND record_id = ?
            ORDER BY ts ASC, entry_id ASC
            """,
            (blinded_project_id, record_id),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def list_entries_by_project(
        self, blinded_project_id: str, *, op: str | None = None, limit: int | None = None
    ) -> list[OplogEntry]:
        """List entries for a project, newest-first, with optional op filter and limit."""
        sql = "SELECT * FROM oplog_entries WHERE blinded_project_id = ?"
        params: list = [blinded_project_id]
        if op is not None:
            sql += " AND op = ?"
            params.append(op)
        sql += " ORDER BY ts DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_entry(r) for r in rows]
