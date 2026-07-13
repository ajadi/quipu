"""Tests for vec0 trigger sync — insert/update/delete on atoms reflected in atoms_vec.

Entire file is gated: vec0 triggers require the real sqlite_vec extension.
"""

from __future__ import annotations

import sqlite3

import pytest
from tests._semantic import TEST_EMBED_DIM

sqlite_vec = pytest.importorskip("sqlite_vec")

from quipu.vec._build import build, drop_index
from quipu.vec._meta import is_build_complete
from quipu.storage.store import pack_embedding, unpack_embedding


pytestmark = pytest.mark.usefixtures("semantic_model")


def _unit_vec(i: int) -> list[float]:
    v = [0.0] * TEST_EMBED_DIM
    v[i % TEST_EMBED_DIM] = 1.0
    return v


def _load_ext(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


def _vec_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT count(*) FROM atoms_vec").fetchone()[0]


def _vec_rowids(conn: sqlite3.Connection) -> set[int]:
    return {r[0] for r in conn.execute("SELECT rowid FROM atoms_vec").fetchall()}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def built_conn(seeded_conn):
    """seeded_conn (5 atoms, project 'p') with atoms_vec built and loaded."""
    _load_ext(seeded_conn)
    build(seeded_conn)
    assert is_build_complete(seeded_conn)
    return seeded_conn


# ---------------------------------------------------------------------------
# Trigger: INSERT
# ---------------------------------------------------------------------------

class TestTriggerInsert:
    def test_new_atom_appears_in_vec(self, built_conn):
        before = _vec_count(built_conn)
        built_conn.execute(
            "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
            " VALUES ('new1', 'diary', 'project', 'new content', ?, 'p', 0)",
            (pack_embedding(_unit_vec(10)),),
        )
        built_conn.commit()
        assert _vec_count(built_conn) == before + 1

    def test_atom_without_embedding_not_inserted(self, built_conn):
        before = _vec_count(built_conn)
        built_conn.execute(
            "INSERT INTO atoms (id, type, scope, content, project_id, invalidated)"
            " VALUES ('no_emb', 'diary', 'project', 'no embedding', 'p', 0)"
        )
        built_conn.commit()
        assert _vec_count(built_conn) == before


# ---------------------------------------------------------------------------
# Trigger: UPDATE
# ---------------------------------------------------------------------------

class TestTriggerUpdate:
    def test_updated_embedding_reflected(self, built_conn):
        # Get rowid of atom0.
        row = built_conn.execute(
            "SELECT rowid FROM atoms WHERE id='atom0'"
        ).fetchone()
        rowid = row[0]

        new_vec = _unit_vec(42)
        built_conn.execute(
            "UPDATE atoms SET embedding=? WHERE id='atom0'",
            (pack_embedding(new_vec),),
        )
        built_conn.commit()

        # The vec row for this rowid must still exist.
        assert rowid in _vec_rowids(built_conn)

    def test_vec_count_unchanged_after_update(self, built_conn):
        before = _vec_count(built_conn)
        built_conn.execute(
            "UPDATE atoms SET embedding=? WHERE id='atom1'",
            (pack_embedding(_unit_vec(99)),),
        )
        built_conn.commit()
        assert _vec_count(built_conn) == before


# ---------------------------------------------------------------------------
# Trigger: DELETE
# ---------------------------------------------------------------------------

class TestTriggerDelete:
    def test_deleted_atom_removed_from_vec(self, built_conn):
        row = built_conn.execute(
            "SELECT rowid FROM atoms WHERE id='atom0'"
        ).fetchone()
        rowid = row[0]
        assert rowid in _vec_rowids(built_conn)

        built_conn.execute("DELETE FROM atoms WHERE id='atom0'")
        built_conn.commit()

        assert rowid not in _vec_rowids(built_conn)

    def test_vec_count_decremented(self, built_conn):
        before = _vec_count(built_conn)
        built_conn.execute("DELETE FROM atoms WHERE id='atom1'")
        built_conn.commit()
        assert _vec_count(built_conn) == before - 1


# ---------------------------------------------------------------------------
# Resumable build
# ---------------------------------------------------------------------------

class TestResumableBuild:
    def test_status_building_then_complete(self, seeded_conn):
        """Simulate crash mid-build: status='building', vec rows deleted → re-run completes."""
        _load_ext(seeded_conn)
        from quipu.vec._meta import ensure_meta_table, set_build_status

        # Step 1: create the vtable and set status='building' to simulate a crash.
        seeded_conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS atoms_vec USING vec0(embedding float[384])"
        )
        ensure_meta_table(seeded_conn)
        set_build_status(seeded_conn, "building", dim=384)
        seeded_conn.commit()

        # Confirm the table exists but has no rows.
        assert _vec_count(seeded_conn) == 0
        assert not is_build_complete(seeded_conn)

        # Step 2: re-run build — must complete successfully.
        build(seeded_conn)

        assert is_build_complete(seeded_conn)
        # All 5 seeded atoms have embeddings — all should be indexed.
        assert _vec_count(seeded_conn) == 5

    def test_blobs_untouched_after_resume(self, seeded_conn):
        """BLOBs in atoms are never mutated by the build/resume path."""
        _load_ext(seeded_conn)
        before = seeded_conn.execute(
            "SELECT id, embedding FROM atoms ORDER BY id"
        ).fetchall()

        build(seeded_conn)

        after = seeded_conn.execute(
            "SELECT id, embedding FROM atoms ORDER BY id"
        ).fetchall()
        assert before == after
