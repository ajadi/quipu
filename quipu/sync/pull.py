"""quipu.sync.pull — fetch hub entries, apply to the replica, merge per record.

Entry point. blinded_project_id is computed ONCE here; the 32-byte *key* is
passed IN by the caller (derived once via get_or_derive_key). No KDF in any loop.

Idempotency comes from two layers:
  - apply_remote: INSERT OR IGNORE on UNIQUE(client_id, sequence_no) — re-pulled
    entries collapse to no-ops (sequence_no dedup).
  - merge.resolve_record: a pure function of present rows — re-running with no
    new entries reproduces the same resolved state.

Pull cursor: stored once per project keyed by the local client_id as peer_id.
The InMemoryTransport cursor is a project-wide opaque offset, so a single
consumption cursor per project is sufficient and idempotent. (See developer
handoff: deviation from the architect's per-remote-peer cursor sketch, justified
by the offset-based transport contract.)
"""

from __future__ import annotations

from quipu.sync._aad import aad_for
from quipu.sync.cursor import read_cursor, upsert_cursor
from quipu.sync.merge import resolve_record
from quipu.sync.oplog_store import OplogStore, from_transport_dict
from quipu.storage.store import Store
from quipu.sync.transport import Transport

_PULL = "pull"


def pull(project_id: str, *, store: Store, transport: Transport, key: bytes, client_id: str) -> int:
    """Pull + apply + merge entries for *project_id*. Returns count newly applied."""
    blinded_project_id = aad_for(project_id, key).decode()

    _, last_cursor = read_cursor(store._conn, blinded_project_id, _PULL, client_id)

    dicts, next_cursor = transport.pull(blinded_project_id, last_cursor)

    oplog = OplogStore(store._conn)
    entries = [from_transport_dict(d) for d in dicts]
    new_record_ids = oplog.apply_remote(entries) if entries else []

    # Resolve each touched record exactly once (dedup record_ids).
    for record_id in dict.fromkeys(new_record_ids):
        resolve_record(oplog, store, blinded_project_id, record_id, key)

    # ALWAYS advance the cursor, even on an empty page, so the opaque hub offset
    # keeps moving forward and a real paging transport never re-fetches forever.
    # last_seq=0 is intentional: pull tracks progress via the opaque last_cursor,
    # not last_seq (last_seq is a push-direction concept).
    upsert_cursor(store._conn, blinded_project_id, _PULL, client_id, 0, next_cursor)

    return len(new_record_ids)
