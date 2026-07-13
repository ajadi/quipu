"""Tests for TASK-067: batched, best-effort access-count bump in search().

Covers:
- search() on a read-only connection returns results and does not raise.
- search() while another connection holds a write lock does not raise from
  the access-count bump path.
"""

from __future__ import annotations

import sqlite3

from quipu.retrieval._search import search
from quipu.storage import store as open_store
from quipu.storage.store import Store, pack_embedding


def test_search_batches_access_bump_for_all_returned_ids(tmp_store, monkeypatch):
    atoms = [
        tmp_store.insert(content="same result", project_id="proj1")
        for _ in range(2)
    ]
    calls: list[list[str]] = []
    original_batch = tmp_store.increment_access_batch

    def record_batch(atom_ids: list[str]) -> int:
        calls.append(atom_ids)
        return original_batch(atom_ids)

    monkeypatch.setattr(tmp_store, "increment_access_batch", record_batch)
    results = search("same result", tier="R0", project_id="proj1", store=tmp_store)

    assert calls == [[result.atom.id for result in results]]
    assert {atom.id for atom in atoms} == {result.atom.id for result in results}
    assert all(tmp_store.get(atom.id).access_count == 1 for atom in atoms)


def test_search_readonly_connection_returns_results(tmp_path):
    db_path = str(tmp_path / "t.db")

    s = open_store(db_path)
    s.insert(content="hello world", project_id="proj1")
    s.close()

    ro_conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    ro_conn.row_factory = sqlite3.Row
    ro_store = Store(ro_conn)
    try:
        results = search("hello world", tier="R0", project_id="proj1", top_k=10, store=ro_store)
        assert len(results) == 1
        assert results[0].atom.content == "hello world"
    finally:
        ro_conn.close()


def test_search_concurrent_write_lock_does_not_raise(tmp_path):
    db_path = str(tmp_path / "t.db")

    s = open_store(db_path)
    s.insert(content="hello world", project_id="proj1")
    s.close()

    # Writer connection holds an uncommitted write transaction, simulating a
    # concurrent drain/write in progress.
    writer_conn = sqlite3.connect(db_path, timeout=0)
    writer_conn.execute("BEGIN IMMEDIATE")
    writer_conn.execute("UPDATE atoms SET access_count = access_count + 1")

    # Reader connection with a zero busy-timeout so lock contention surfaces
    # immediately as sqlite3.OperationalError instead of blocking/retrying.
    reader_conn = sqlite3.connect(db_path, timeout=0)
    reader_conn.row_factory = sqlite3.Row
    reader_store = Store(reader_conn)

    try:
        results = search("hello world", tier="R0", project_id="proj1", top_k=10, store=reader_store)
        assert len(results) == 1
        assert results[0].atom.content == "hello world"
    finally:
        writer_conn.rollback()
        writer_conn.close()
        reader_conn.close()
