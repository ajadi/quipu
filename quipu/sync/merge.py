"""quipu.sync.merge — CRDT resolution of a record's oplog history into the replica.

Policy (per record_id), a pure function of the rows currently present:
  1. invalidate-wins LATCH: any op='invalidate' => resolved state is invalidated
     and STAYS invalidated, independent of arrival order (monotonic).
  2. LWW among upserts: highest ts wins; equal ts => higher entry_id
     (lexicographic) tie-break.
  3. Resolved state is written to atoms via Store: upsert winner ->
     insert(id=record_id, created_at=ts, ...) (or invalidate-toggle if it
     already exists); invalidated -> update_invalidated(record_id, True).

Conflict logging: two+ upserts on the same record from DIFFERENT client_ids with
divergent content resolved only by the entry_id tie-break (equal ts) are appended
to memory/sync-conflicts.md. A clean invalidate-over-upsert is policy, NOT a
conflict, and is not logged.

Idempotency: re-running with no new rows reproduces the same resolved state.
"""

from __future__ import annotations

import os
from pathlib import Path

from quipu.oplog.codec import decode_entry
from quipu.oplog.entry import OplogEntry
from quipu.storage.store import Store
from quipu.sync.oplog_store import OplogStore


def _resolve_conflicts_path(
    conflicts_path: Path | None, store: Store | None = None
) -> Path:
    """Resolve the sync-conflicts.md location (packaging-safe).

    Order: explicit arg -> QUIPU_CONFLICTS_PATH env -> the DB file's base
    directory -> cwd/memory. Never anchored to __file__ (breaks in a pip install).
    """
    if conflicts_path is not None:
        return Path(conflicts_path)
    env = os.environ.get("QUIPU_CONFLICTS_PATH")
    if env:
        return Path(env)
    db_path = _db_base_dir(store)
    base = db_path if db_path is not None else Path.cwd()
    return base / "memory" / "sync-conflicts.md"


def _db_base_dir(store: Store | None) -> Path | None:
    """Best-effort base directory of the DB file backing *store*, or None."""
    if store is None:
        return None
    try:
        row = store._conn.execute("PRAGMA database_list").fetchone()
        db_file = row["file"] if row is not None else None
    except Exception:
        return None
    if not db_file:
        return None
    return Path(db_file).resolve().parent


def _safe(value: object) -> str:
    """Neutralize hub-supplied text for a single markdown line.

    Strips CR/LF (so a hub field cannot break out of its line) and prefixes a
    space when the value starts with '#' (so it cannot inject a markdown heading).
    """
    s = str(value).replace("\r", " ").replace("\n", " ")
    if s.startswith("#"):
        s = " " + s
    return s


def _winning_upsert(upserts: list[OplogEntry]) -> OplogEntry:
    """LWW: highest ts, entry_id lexicographic tie-break on equal ts."""
    return max(upserts, key=lambda e: (e.ts, e.entry_id))


def resolve_record(
    oplog: OplogStore,
    store: Store,
    blinded_project_id: str,
    record_id: str,
    key: bytes,
    *,
    conflicts_path: Path | None = None,
) -> None:
    """Resolve and write the current state of *record_id* into the replica."""
    entries = oplog.entries_for_record(blinded_project_id, record_id)
    if not entries:
        return

    invalidated = any(e.op == "invalidate" for e in entries)
    upserts = [e for e in entries if e.op == "upsert"]

    if upserts:
        winner = _winning_upsert(upserts)
        fields = decode_entry(winner.payload, key, blinded_project_id)
        existing = store.get(record_id)
        if existing is None:
            store.insert(
                content=fields.get("content", ""),
                project_id=fields.get("project_id"),
                type=fields.get("type", "diary"),
                scope=fields.get("scope", "project"),
                metadata=fields.get("metadata") or {},
                refs=fields.get("refs") or [],
                id=record_id,
                created_at=winner.ts,
            )
        # V1 append-only model: atom content is immutable after insert; a content
        # change is a NEW atom id (invalidate-old + insert-new). The same
        # record_id never carries differing content across clients, so a
        # pre-existing atom keeps its content and we only converge the
        # invalidated flag (handled below). In-place content LWW is N/A under the
        # append-only model.

        _maybe_log_conflict(record_id, upserts, store, conflicts_path)

    if invalidated:
        # Latch: ensure the flag is set, durably. If the atom was never seen
        # locally (invalidate arrived before any upsert for this record_id),
        # update_invalidated is a no-op — so insert a tombstone first, then set
        # the flag. This keeps the invalidate-wins latch durable: a
        # later-arriving upsert for the same record_id still sees it invalidated.
        if not store.update_invalidated(record_id, True):
            inv_entry = next(e for e in entries if e.op == "invalidate")
            inv_fields = decode_entry(inv_entry.payload, key, blinded_project_id)
            store.insert(
                content=inv_fields.get("content", ""),
                project_id=inv_fields.get("project_id"),
                type=inv_fields.get("type", "diary"),
                scope=inv_fields.get("scope", "project"),
                metadata=inv_fields.get("metadata") or {},
                refs=inv_fields.get("refs") or [],
                id=record_id,
                created_at=inv_entry.ts,
            )
            store.update_invalidated(record_id, True)


def _maybe_log_conflict(
    record_id: str,
    upserts: list[OplogEntry],
    store: Store | None = None,
    conflicts_path: Path | None = None,
) -> None:
    """Log a genuine concurrent-divergent upsert conflict (equal-ts tie-break)."""
    if len(upserts) < 2:
        return
    winner = _winning_upsert(upserts)
    # Genuine conflict = at least one OTHER upsert from a different client with
    # the SAME ts as the winner (so the outcome hinged on the entry_id tie-break).
    rivals = [
        e
        for e in upserts
        if e.entry_id != winner.entry_id
        and e.client_id != winner.client_id
        and e.ts == winner.ts
    ]
    if rivals:
        log_conflict(
            record_id,
            [winner, *rivals],
            store=store,
            conflicts_path=conflicts_path,
        )


def log_conflict(
    record_id: str,
    entries: list[OplogEntry],
    *,
    store: Store | None = None,
    conflicts_path: Path | None = None,
) -> None:
    """Append a conflict record to sync-conflicts.md for human review.

    Hub-supplied fields (record_id, entry_id, client_id, ts) are sanitized via
    _safe() before interpolation so a malicious hub cannot inject newlines or a
    spurious markdown heading into the log.
    """
    path = _resolve_conflicts_path(conflicts_path, store)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"## conflict: record_id={_safe(record_id)}",
        f"- resolved_by: LWW ts tie-break -> entry_id (winner={_safe(entries[0].entry_id)})",
    ]
    for e in entries:
        lines.append(
            f"- entry_id={_safe(e.entry_id)} client_id={_safe(e.client_id)} "
            f"seq={e.sequence_no} ts={_safe(e.ts)} op={_safe(e.op)}"
        )
    lines.append("")
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
