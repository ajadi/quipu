"""Tests for merge: invalidate-wins latch, LWW tie-break, two-client convergence,
and cross-project blob rejection.
"""

import pytest

from tests.sync.conftest import write_local

from quipu.crypto.errors import DecryptError
from quipu.sync import pull, push
from quipu.sync.merge import resolve_record
from quipu.sync.oplog_store import OplogStore


def test_invalidate_wins_then_later_upsert_stays_invalidated(
    make_store, transport, key, project_id
):
    """Apply an invalidate, then a LATER upsert -> record stays invalidated (latch)."""
    writer = make_store("w")
    reader = make_store("r")

    # invalidate at t2, then a *later* upsert at t3 on the same record
    write_local(writer, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:01Z", content="v1")
    write_local(writer, key, project_id, "c1", op="invalidate", record_id="r1",
                ts="2026-06-21T00:00:02Z")
    write_local(writer, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:03Z", content="v2-after-invalidate")
    push(project_id, store=writer, transport=transport, key=key, client_id="c1")

    pull(project_id, store=reader, transport=transport, key=key, client_id="r")
    atom = reader.get("r1")
    assert atom is not None
    assert atom.invalidated is True  # latch holds despite later upsert


def test_lww_tie_break_highest_entry_id_wins(make_store, key, project_id, blinded):
    """Equal ts: higher entry_id (lexicographic) is the LWW winner.

    Falsifiable: entry_id("zzz:1") = f4f8... > entry_id("aaa:1") = b766..., so the
    expected winner is the literal "from-zzz" (NOT a runtime max() of the inputs).
    """
    store = make_store("s")
    # two clients, same ts, different content -> tie-break decides
    write_local(store, key, project_id, "aaa", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:05Z", content="from-aaa")
    write_local(store, key, project_id, "zzz", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:05Z", content="from-zzz")

    oplog = OplogStore(store._conn)
    resolve_record(oplog, store, blinded, "r1", key)

    # SHA-256("zzz:1") > SHA-256("aaa:1") lexicographically -> zzz wins the tie.
    assert store.get("r1").content == "from-zzz"


def test_preexisting_atom_keeps_content_converges_invalidated_flag(
    make_store, key, project_id, blinded
):
    """R-002 (V1 append-only constraint): a later-ts UPSERT for an existing atom
    keeps its content intact, and a remote INVALIDATE converges the flag.

    Content is immutable post-insert in V1; the same record_id never carries
    differing content. So the pre-existing atom must end up invalidated with its
    content untouched — not corrupted or dropped.
    """
    store = make_store("s")

    # Atom already present locally (e.g. produced by an earlier local write).
    store.insert(content="original", project_id=project_id, id="r1",
                 created_at="2026-06-21T00:00:01Z")

    # A later-ts upsert (same content, append-only model) + an invalidate land.
    write_local(store, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:02Z", content="original")
    write_local(store, key, project_id, "c1", op="invalidate", record_id="r1",
                ts="2026-06-21T00:00:03Z")

    oplog = OplogStore(store._conn)
    resolve_record(oplog, store, blinded, "r1", key)

    atom = store.get("r1")
    assert atom is not None
    assert atom.content == "original"   # content intact, not corrupted/dropped
    assert atom.invalidated is True     # invalidated flag converged


def test_invalidate_never_seen_then_later_upsert_stays_invalidated(
    make_store, transport, key, project_id
):
    """R-003: invalidate of a never-before-seen record_id must latch durably.

    Apply an invalidate first (no atom exists yet) -> a tombstone is inserted and
    flagged invalidated. A LATER upsert for the same record_id must respect the
    latch: the record stays invalidated, content not resurrected.
    """
    writer = make_store("w")
    reader = make_store("r")

    # invalidate (t2) is pushed/pulled BEFORE the reader ever saw an upsert for r1
    write_local(writer, key, project_id, "c1", op="invalidate", record_id="r1",
                ts="2026-06-21T00:00:02Z")
    push(project_id, store=writer, transport=transport, key=key, client_id="c1")
    pull(project_id, store=reader, transport=transport, key=key, client_id="r")

    atom = reader.get("r1")
    assert atom is not None          # durable tombstone inserted
    assert atom.invalidated is True  # latch set

    # a LATER upsert for the same record_id arrives -> latch must hold
    write_local(writer, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:05Z", content="resurrected")
    push(project_id, store=writer, transport=transport, key=key, client_id="c1")
    pull(project_id, store=reader, transport=transport, key=key, client_id="r")

    atom = reader.get("r1")
    assert atom.invalidated is True  # latch held; invalidate wins


def test_two_clients_converge(make_store, transport, key, project_id):
    """Two replicas share one InMemoryTransport: each writes, pushes, cross-pulls
    -> both converge to the same resolved atom state.
    """
    a = make_store("A")
    b = make_store("B")

    write_local(a, key, project_id, "A", op="upsert", record_id="shared",
                ts="2026-06-21T00:00:01Z", content="from-A")
    write_local(b, key, project_id, "B", op="upsert", record_id="shared",
                ts="2026-06-21T00:00:02Z", content="from-B-newer")

    push(project_id, store=a, transport=transport, key=key, client_id="A")
    push(project_id, store=b, transport=transport, key=key, client_id="B")

    pull(project_id, store=a, transport=transport, key=key, client_id="A")
    pull(project_id, store=b, transport=transport, key=key, client_id="B")

    atom_a = a.get("shared")
    atom_b = b.get("shared")
    assert atom_a is not None and atom_b is not None
    # LWW: B's upsert has the higher ts -> both converge to from-B-newer
    assert atom_a.content == atom_b.content == "from-B-newer"
    assert atom_a.invalidated == atom_b.invalidated is False


def test_cross_project_blob_rejected_on_decode(make_store, transport, key, project_id):
    """A payload blob from another project (different blinded AAD) fails to decode."""
    other_project = "proj-OTHER"
    writer = make_store("w")
    reader = make_store("r")

    # writer produces an entry for `other_project`, but we deliver it into the
    # reader's pull for `project_id` by relabeling the blinded id on the wire.
    entry = write_local(writer, key, other_project, "c1", op="upsert", record_id="x1",
                        ts="2026-06-21T00:00:01Z", content="cross")

    # Forge a transport dict that claims project_id's partition but carries the
    # other project's encrypted blob (AAD bound to other_project's blinded id).
    from quipu.sync._aad import aad_for
    target_blinded = aad_for(project_id, key).decode()
    import base64
    forged = {
        "entry_id": entry.entry_id,
        "client_id": "c1",
        "sequence_no": entry.sequence_no,
        "op": "upsert",
        "record_id": "x1",
        "blinded_project_id": target_blinded,
        "ts": entry.ts,
        "payload": base64.b64encode(entry.payload).decode("ascii"),
    }
    transport.push(target_blinded, [forged])

    with pytest.raises(DecryptError):
        pull(project_id, store=reader, transport=transport, key=key, client_id="r")
