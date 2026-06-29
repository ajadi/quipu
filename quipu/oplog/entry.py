"""quipu.oplog.entry — OplogEntry dataclass + content-addressed entry_id.

Fields mirror the oplog_entries DB columns 1:1 (minus the local rowid surrogate).
Schema coupling lives in quipu/sync/oplog_store.py — this module is schema-agnostic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class OplogEntry:
    """One append-only oplog operation.

    payload is the opaque encrypted blob produced by quipu/oplog/codec.encode_entry.
    source/pushed are local-only and are NEVER serialized to the transport.
    """

    entry_id: str
    client_id: str
    sequence_no: int
    op: str  # 'upsert' | 'invalidate'
    record_id: str  # == Atom.id
    blinded_project_id: str
    ts: str  # ISO-8601 UTC
    payload: bytes  # codec output (encrypted; opaque to hub)
    source: str = "local"  # 'local' | 'remote'
    pushed: bool = False

    @staticmethod
    def compute_entry_id(client_id: str, sequence_no: int) -> str:
        """SHA-256(f"{client_id}:{sequence_no}") hex-64. Content-addressed dedup key."""
        return hashlib.sha256(f"{client_id}:{sequence_no}".encode()).hexdigest()
