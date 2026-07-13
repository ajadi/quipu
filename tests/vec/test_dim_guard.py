"""Tests for the cross-dim guard (VecDimMismatchError) — TASK-053.

The built vec index records the dim it was built with (ember_vec_meta.dim).
assert_dim_matches() compares that stored dim against the CURRENTLY active
model's dim (active_dim()) and fails closed with a clear rebuild message
instead of silently truncating/corrupting scores. It fires at two
chokepoints: query_ready() and ensure_index().

Hermetic: sqlite-vec is NOT required for any test here — is_loaded/try_load
are monkeypatched to simulate an available extension, so only the
meta-table bookkeeping + guard logic is exercised (no real vec0 table).
"""

from __future__ import annotations

import pytest

from quipu.vec._meta import ensure_meta_table, get_build_dim, set_build_status
from quipu.vec._query import VecDimMismatchError, assert_dim_matches, query_ready
from quipu.vec.index import VecState, ensure_index


@pytest.fixture()
def loaded(monkeypatch):
    """Simulate sqlite-vec being loaded, without requiring the real extension."""
    monkeypatch.setattr("quipu.vec._query.is_loaded", lambda conn: True)
    monkeypatch.setattr("quipu.vec.index.try_load", lambda conn: True)


# ---------------------------------------------------------------------------
# assert_dim_matches — direct unit tests
# ---------------------------------------------------------------------------

class TestAssertDimMatches:
    def test_raises_with_nd_to_md_message_on_mismatch(self, tmp_conn, monkeypatch):
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=768)
        tmp_conn.commit()

        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")  # 384

        with pytest.raises(VecDimMismatchError, match=r"768d.*384d"):
            assert_dim_matches(tmp_conn)

    def test_no_raise_when_dims_match(self, tmp_conn, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "nomic-embed-text-v1.5")  # 768
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=768)
        tmp_conn.commit()

        assert_dim_matches(tmp_conn)  # must not raise

    def test_no_raise_when_build_status_not_complete(self, tmp_conn, monkeypatch):
        """status='building' (not yet complete) must not trigger the guard,
        even if the recorded dim already mismatches the active model."""
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "building", dim=768)
        tmp_conn.commit()
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")

        assert_dim_matches(tmp_conn)  # must not raise

    def test_no_raise_on_fresh_db_no_build_row(self, tmp_conn):
        """No build row at all (fresh DB) — guard is a no-op."""
        assert get_build_dim(tmp_conn) is None
        assert_dim_matches(tmp_conn)  # must not raise

    def test_no_raise_on_null_dim_legacy_row(self, tmp_conn, monkeypatch):
        """Legacy build row with dim=NULL — guard skipped, no false positive."""
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=None)
        tmp_conn.commit()
        assert get_build_dim(tmp_conn) is None

        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        assert_dim_matches(tmp_conn)  # must not raise


# ---------------------------------------------------------------------------
# query_ready — chokepoint 1
# ---------------------------------------------------------------------------

class TestQueryReadyGuard:
    def test_raises_on_dim_mismatch(self, tmp_conn, loaded, monkeypatch):
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=768)
        tmp_conn.commit()
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")

        with pytest.raises(VecDimMismatchError, match=r"768d.*384d"):
            query_ready(tmp_conn)

    def test_true_when_dims_match(self, tmp_conn, loaded, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "nomic-embed-text-v1.5")  # 768
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=768)
        tmp_conn.commit()

        assert query_ready(tmp_conn) is True

    def test_false_on_fresh_db_no_raise(self, tmp_conn, loaded, monkeypatch):
        """No build row -> query_ready returns False; guard never triggered."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        assert query_ready(tmp_conn) is False


# ---------------------------------------------------------------------------
# ensure_index — chokepoint 2 (already-complete branch)
# ---------------------------------------------------------------------------

class TestEnsureIndexGuard:
    def test_raises_on_dim_mismatch_when_already_complete(self, tmp_conn, loaded, monkeypatch):
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=768)
        tmp_conn.commit()
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")

        with pytest.raises(VecDimMismatchError, match=r"768d.*384d"):
            ensure_index(tmp_conn)

    def test_ready_when_dims_match(self, tmp_conn, loaded, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "nomic-embed-text-v1.5")  # 768
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=768)
        tmp_conn.commit()

        assert ensure_index(tmp_conn) == VecState.READY

    def test_no_raise_on_fresh_db_below_threshold(self, tmp_conn, loaded, monkeypatch):
        """Fresh DB, no build row -> BELOW_THRESHOLD; guard not reached."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
        assert ensure_index(tmp_conn, threshold=100) == VecState.BELOW_THRESHOLD
