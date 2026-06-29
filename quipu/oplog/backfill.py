"""quipu.oplog.backfill — one-shot re-emit of pre-existing atoms into oplog_entries.

The producer captures only NEW writes going forward. Atoms created BEFORE sync
was enabled never reach oplog_entries, so `quipu push` never ships them and a
second machine cannot reconstruct them. This module converges that gap.

backfill_project(store, project_id) enumerates the project's current atoms and
emits one UPSERT per live atom / one INVALIDATE per invalidated atom through the
SAME producer codec path (shared helper producer._emit_for_atom — NO new cipher
call site). It converges CURRENT state only; it does NOT replay history.

Gate: resolved ONCE via producer._active_key (hub configured AND key available).
No key / no hub -> BackfillResult(status="inactive"); no crash, no KDF-per-atom.

Idempotency: per-record pre-check via OplogStore.entries_for_record — any atom
that already has an oplog entry (local from a prior emit/backfill, OR remote from
a pull) is skipped. Re-run = 0 new entries. This is load-bearing: entry_id is
keyed on (client_id, sequence_no), so a re-run mints fresh seq/entry_ids and the
UNIQUE backstop would NOT dedup it.

Failure isolation: each atom is its own append_local (commits per row). A per-atom
exception is caught (counted as skipped) and the run continues; a crash mid-run
leaves emitted rows intact and is safe to re-run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from quipu.oplog import producer

logger = logging.getLogger(__name__)


@dataclass
class BackfillResult:
    """Outcome of a backfill_project run.

    status: "ok" (gate active, run completed) | "inactive" (no hub/key — no-op).
    emitted: number of new oplog rows appended.
    skipped: atoms skipped (already had an entry, or raised per-atom).
    """

    status: str
    emitted: int = 0
    skipped: int = 0


def backfill_project(store: object, project_id: str) -> BackfillResult:
    """Emit oplog entries for the current state of every atom in *project_id*.

    Returns BackfillResult(status="inactive") when sync is not active (no hub
    config or no derivable key) — a clean local-only no-op, not an error.
    Otherwise emits one UPSERT per live atom / one INVALIDATE per invalidated
    atom (skipping atoms that already have an oplog entry) and returns counts.
    """
    ka = producer._active_key(project_id)
    if ka is None:
        return BackfillResult(status="inactive")

    key, bpid = ka

    from quipu.config import get_client_id
    from quipu.sync.oplog_store import OplogStore

    client_id = get_client_id(store)
    oplog = OplogStore(store._conn)  # type: ignore[union-attr]

    atoms = store.list_by_project(project_id, include_invalidated=True)  # type: ignore[union-attr]

    emitted = 0
    skipped = 0
    for atom in atoms:
        try:
            # Dedup probe: skip any record that already has a local OR remote
            # oplog entry (covers re-run AND backfill-vs-live double-emit).
            if oplog.entries_for_record(bpid, atom.id):
                skipped += 1
                continue

            op = "invalidate" if atom.invalidated else "upsert"
            inserted = producer._emit_for_atom(
                store, op=op, atom=atom, key=key, bpid=bpid, client_id=client_id
            )
            if inserted:
                emitted += 1
            else:
                skipped += 1
        except Exception as exc:
            logger.warning(
                "quipu.oplog.backfill: failed for record_id=%r (%s); skipped, continuing.",
                getattr(atom, "id", "?"),
                type(exc).__name__,
            )
            skipped += 1

    return BackfillResult(status="ok", emitted=emitted, skipped=skipped)
