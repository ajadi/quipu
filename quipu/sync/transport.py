"""quipu.sync.transport — Transport Protocol + InMemoryTransport (hub stand-in).

The real HTTP transport is TASK-013. Here we provide:
  - Transport: the Protocol push/pull speak to.
  - InMemoryTransport: a zero-knowledge hub emulation for tests + convergence.

ZERO-KNOWLEDGE INVARIANT: the dicts handed to push() carry ONLY hub-visible
fields (entry_id, client_id, sequence_no, op, record_id, blinded_project_id, ts,
payload). InMemoryTransport stores EXACTLY what it is given — it never sees
source/pushed, plaintext content, the real project_id, or the key. It cannot
decrypt payload (it has no key). It partitions purely by blinded_project_id.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Transport(Protocol):
    """Client <-> hub transport. Real impl (HTTP) is TASK-013."""

    def push(self, blinded_project_id: str, entries: list[dict]) -> None:
        """Send hub-visible entry dicts for a blinded project. Idempotent by entry_id."""
        ...

    def pull(
        self, blinded_project_id: str, cursor: str | None
    ) -> tuple[list[dict], str | None]:
        """Fetch entries since *cursor*. Returns (entries, next_cursor)."""
        ...


class InMemoryTransport:
    """In-process zero-knowledge hub emulation.

    Stores ingested entry dicts per blinded_project_id in arrival order, keyed
    by entry_id for ingest-dedup. The pull cursor is the count of entries the
    caller has already consumed (a string offset) — opaque to the client.
    """

    def __init__(self) -> None:
        # blinded_project_id -> ordered list of entry dicts (arrival order)
        self._log: dict[str, list[dict]] = {}
        # blinded_project_id -> set of entry_ids already ingested (dedup)
        self._seen: dict[str, set[str]] = {}

    def push(self, blinded_project_id: str, entries: list[dict]) -> None:
        log = self._log.setdefault(blinded_project_id, [])
        seen = self._seen.setdefault(blinded_project_id, set())
        for d in entries:
            eid = d["entry_id"]
            if eid in seen:
                continue  # ingest dedup — hub keeps first copy
            seen.add(eid)
            log.append(dict(d))  # store exactly what we were given (copy)

    def pull(
        self, blinded_project_id: str, cursor: str | None
    ) -> tuple[list[dict], str | None]:
        log = self._log.get(blinded_project_id, [])
        offset = int(cursor) if cursor else 0
        new = [dict(d) for d in log[offset:]]
        next_cursor = str(len(log))
        return new, next_cursor
