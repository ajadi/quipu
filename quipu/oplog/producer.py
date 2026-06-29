"""quipu.oplog.producer — wire local atom writes/invalidations to oplog_entries.

Called from three LOCAL seams ONLY:
  - quipu/write/pipeline.py::write()        -> emit(op='upsert')
  - quipu/invalidation/cosine.py::invalidate_superseded()  -> emit(op='invalidate')
  - quipu/mcp/tools.py::_handle_quipu_invalidate            -> emit(op='invalidate')

merge.py remote-apply NEVER calls emit (non-emitting by construction).

Activation: emit only when sync is active (hub configured AND QUIPU_KEY or
QUIPU_PASSPHRASE set AND key derivable). Cached per (project_id, process) —
KDF runs at most once per project_id per process.

Failure isolation: the entire emit() body is wrapped in one try/except Exception
so oplog production NEVER raises, NEVER blocks a local write.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Process-lifetime caches keyed by project_id.
# On success: _KEY_CACHE[pid] = bytes (32-byte key); _BPID_CACHE[pid] = str (64-hex bpid).
# On failure: _KEY_CACHE[pid] = None (short-circuit all future calls in this process).
_KEY_CACHE: dict[str, bytes | None] = {}
_BPID_CACHE: dict[str, str] = {}


def reset_cache() -> None:
    """Clear process-lifetime caches. TEST HOOK ONLY — do not call in production."""
    _KEY_CACHE.clear()
    _BPID_CACHE.clear()


def _active_key(project_id: str) -> tuple[bytes, str] | None:
    """Return (key, blinded_project_id) if sync is active, else None.

    Active = get_hub_config() is not None
             AND (QUIPU_KEY or QUIPU_PASSPHRASE set — getpass-hang guard)
             AND key derivable via get_or_derive_key(project_id).

    Result is cached per project_id for the process lifetime:
      - success -> (key, bpid) cached; subsequent calls are a dict lookup (no KDF).
      - failure -> None cached; subsequent calls short-circuit immediately.

    Never prompts (QUIPU_KEY/QUIPU_PASSPHRASE guard). Never raises.
    """
    import os

    # Fast path: already cached (success or failure).
    if project_id in _KEY_CACHE:
        cached_key = _KEY_CACHE[project_id]
        if cached_key is None:
            return None
        return cached_key, _BPID_CACHE[project_id]

    try:
        from quipu.config import get_hub_config
        from quipu.keystore._backend import get_or_derive_key
        from quipu.sync._aad import aad_for

        # Condition 1: hub configured.
        if get_hub_config() is None:
            _KEY_CACHE[project_id] = None
            return None

        # Condition 2: key source available in env (getpass-hang guard, known-issues L16).
        if not (os.environ.get("QUIPU_KEY") or os.environ.get("QUIPU_PASSPHRASE")):
            _KEY_CACHE[project_id] = None
            return None

        # Condition 3: key derivable.
        key = get_or_derive_key(project_id)
        bpid = aad_for(project_id, key).decode()

        _KEY_CACHE[project_id] = key
        _BPID_CACHE[project_id] = bpid
        return key, bpid

    except Exception as exc:
        logger.warning(
            "quipu.oplog.producer: key derivation failed for project_id=%r (%s); "
            "caching as local-only (0 oplog entries until process restart).",
            project_id,
            type(exc).__name__,
        )
        _KEY_CACHE[project_id] = None
        return None


def _emit_for_atom(
    store: object,
    *,
    op: str,
    atom: object,
    key: bytes,
    bpid: str,
    client_id: str,
) -> bool:
    """Build + encrypt + append ONE OplogEntry for *atom* via the shared codec path.

    This is the codec-bearing core extracted from emit(): it is the SINGLE site
    that calls encode_entry. Both live emit() and the one-shot backfill delegate
    here after resolving the (key, bpid, client_id) gate themselves.

    ts = atom.updated_at (event-time, NEVER wall-clock now).

    Returns True if a new row was appended, False if it already existed
    (append_local is INSERT OR IGNORE on UNIQUE(client_id, sequence_no)).
    """
    from quipu.oplog.codec import encode_entry
    from quipu.oplog.entry import OplogEntry
    from quipu.sync.oplog_store import OplogStore

    oplog = OplogStore(store._conn)  # type: ignore[union-attr]
    seq = oplog.next_sequence_no(client_id)
    entry_id = OplogEntry.compute_entry_id(client_id, seq)

    op_fields = {
        "op": op,
        "record_id": atom.id,  # type: ignore[union-attr]
        "ts": atom.updated_at,  # type: ignore[union-attr]
        "content": atom.content,  # type: ignore[union-attr]
        "type": atom.type,  # type: ignore[union-attr]
        "scope": atom.scope,  # type: ignore[union-attr]
        "metadata": atom.metadata,  # type: ignore[union-attr]
        "refs": atom.refs,  # type: ignore[union-attr]
        "project_id": atom.project_id,  # type: ignore[union-attr]
    }

    payload = encode_entry(op_fields, key, bpid)

    entry = OplogEntry(
        entry_id=entry_id,
        client_id=client_id,
        sequence_no=seq,
        op=op,
        record_id=atom.id,  # type: ignore[union-attr]
        blinded_project_id=bpid,
        ts=atom.updated_at,  # type: ignore[union-attr]  — event-time, NOT now
        payload=payload,
        source="local",
        pushed=False,
    )

    return oplog.append_local(entry)


def emit(store: object, *, op: str, atom: object, project_id: str | None) -> None:
    """Emit one OplogEntry for a local atom write or invalidation.

    No-op (silently) when:
      - project_id is None (unscoped atom)
      - sync is not active (no hub config, no key env, or key derivation failure)

    The entire body is wrapped in try/except Exception so oplog production
    NEVER raises and NEVER blocks the atom write that preceded it.

    Args:
        store:      The Storage Store whose _conn owns the oplog_entries table.
        op:         'upsert' | 'invalidate'
        atom:       Atom dataclass (id, updated_at, content, type, scope,
                    metadata, refs, project_id attributes required).
        project_id: Project scope; None -> return immediately (no-op).
    """
    try:
        if project_id is None:
            return

        ka = _active_key(project_id)
        if ka is None:
            return

        key, bpid = ka

        from quipu.config import get_client_id

        client_id = get_client_id(store)
        _emit_for_atom(
            store, op=op, atom=atom, key=key, bpid=bpid, client_id=client_id
        )

    except Exception as exc:
        logger.warning(
            "quipu.oplog.producer: emit failed (op=%r, project_id=%r, err=%s); "
            "atom write succeeded; oplog entry dropped.",
            op,
            project_id,
            type(exc).__name__,
        )
