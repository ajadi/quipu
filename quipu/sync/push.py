"""quipu.sync.push — send unpushed local oplog entries to the hub.

Entry point. blinded_project_id is computed ONCE here (via aad/_blind through
the choke-point); the 32-byte *key* is passed IN by the caller (who derived it
once via get_or_derive_key). No KDF and no per-entry blinding happens here.
"""

from __future__ import annotations

from quipu.sync._aad import aad_for
from quipu.sync.cursor import read_cursor, upsert_cursor
from quipu.sync.oplog_store import OplogStore, to_transport_dict
from quipu.storage.store import Store
from quipu.sync.transport import Transport

_PUSH = "push"


def push(project_id: str, *, store: Store, transport: Transport, key: bytes, client_id: str) -> int:
    """Push this client's unpushed entries for *project_id*. Returns count pushed.

    Idempotent: already-pushed entries are filtered by the local pushed flag;
    the hub additionally dedups by entry_id. Re-running with nothing new pushes 0.
    """
    # AAD choke-point also yields the blinded project id (computed once).
    blinded_project_id = aad_for(project_id, key).decode()

    oplog = OplogStore(store._conn)
    pending = oplog.unpushed(blinded_project_id, client_id)
    if not pending:
        return 0

    entries = [to_transport_dict(e) for e in pending]
    transport.push(blinded_project_id, entries)

    oplog.mark_pushed([e.entry_id for e in pending])

    last_seq, _ = read_cursor(store._conn, blinded_project_id, _PUSH, client_id)
    new_last = max(last_seq, max(e.sequence_no for e in pending))
    upsert_cursor(store._conn, blinded_project_id, _PUSH, client_id, new_last, None)

    return len(pending)
