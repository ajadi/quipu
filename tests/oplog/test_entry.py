"""Tests for quipu.oplog.entry — OplogEntry + compute_entry_id."""

import hashlib

from quipu.oplog.entry import OplogEntry


def test_compute_entry_id_deterministic():
    a = OplogEntry.compute_entry_id("client-1", 5)
    b = OplogEntry.compute_entry_id("client-1", 5)
    assert a == b
    assert a == hashlib.sha256(b"client-1:5").hexdigest()
    assert len(a) == 64


def test_compute_entry_id_distinct_per_seq_and_client():
    assert OplogEntry.compute_entry_id("c", 1) != OplogEntry.compute_entry_id("c", 2)
    assert OplogEntry.compute_entry_id("c1", 1) != OplogEntry.compute_entry_id("c2", 1)


def test_entry_defaults():
    e = OplogEntry(
        entry_id="x",
        client_id="c",
        sequence_no=1,
        op="upsert",
        record_id="r",
        blinded_project_id="b",
        ts="2026-06-21T00:00:00Z",
        payload=b"blob",
    )
    assert e.source == "local"
    assert e.pushed is False
