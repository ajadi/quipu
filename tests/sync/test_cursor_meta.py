"""Tests for read_cursor_meta in quipu/sync/cursor.py."""

from __future__ import annotations

import pytest

from quipu.sync.cursor import read_cursor_meta, upsert_cursor
from quipu.storage import store as open_store


@pytest.fixture()
def tmp_store(tmp_path):
    s = open_store(str(tmp_path / "meta.db"))
    yield s
    s.close()


BPID = "a" * 64
DIRECTION = "push"
PEER = "client-001"


class TestReadCursorMeta:
    def test_returns_none_none_when_absent(self, tmp_store):
        cursor, updated = read_cursor_meta(tmp_store._conn, BPID, DIRECTION, PEER)
        assert cursor is None
        assert updated is None

    def test_returns_cursor_after_upsert(self, tmp_store):
        upsert_cursor(tmp_store._conn, BPID, DIRECTION, PEER, 0, "42")
        cursor, updated = read_cursor_meta(tmp_store._conn, BPID, DIRECTION, PEER)
        assert cursor == "42"
        assert updated is not None

    def test_updated_at_is_iso_string(self, tmp_store):
        upsert_cursor(tmp_store._conn, BPID, DIRECTION, PEER, 0, "1")
        _, updated = read_cursor_meta(tmp_store._conn, BPID, DIRECTION, PEER)
        assert isinstance(updated, str)
        # Should contain 'T' (ISO 8601 format)
        assert "T" in updated

    def test_separate_directions(self, tmp_store):
        upsert_cursor(tmp_store._conn, BPID, "push", PEER, 0, "push-cur")
        upsert_cursor(tmp_store._conn, BPID, "pull", PEER, 0, "pull-cur")
        push_cursor, _ = read_cursor_meta(tmp_store._conn, BPID, "push", PEER)
        pull_cursor, _ = read_cursor_meta(tmp_store._conn, BPID, "pull", PEER)
        assert push_cursor == "push-cur"
        assert pull_cursor == "pull-cur"

    def test_separate_peers(self, tmp_store):
        upsert_cursor(tmp_store._conn, BPID, DIRECTION, "peer-a", 0, "a-cur")
        upsert_cursor(tmp_store._conn, BPID, DIRECTION, "peer-b", 0, "b-cur")
        a_cursor, _ = read_cursor_meta(tmp_store._conn, BPID, DIRECTION, "peer-a")
        b_cursor, _ = read_cursor_meta(tmp_store._conn, BPID, DIRECTION, "peer-b")
        assert a_cursor == "a-cur"
        assert b_cursor == "b-cur"

    def test_none_cursor_stored_and_returned(self, tmp_store):
        upsert_cursor(tmp_store._conn, BPID, DIRECTION, PEER, 0, None)
        cursor, updated = read_cursor_meta(tmp_store._conn, BPID, DIRECTION, PEER)
        assert cursor is None
        assert updated is not None  # row exists, just cursor is null
