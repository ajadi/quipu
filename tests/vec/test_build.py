"""Tests for quipu.vec._build — atom_count, crossed, build lifecycle.

Tests that require the real sqlite-vec extension (vec0 virtual table) are
gated with pytest.importorskip("sqlite_vec").  The threshold/count/idempotency
tests that operate on plain SQLite tables run always.
"""

from __future__ import annotations

import sqlite3

import pytest

from quipu.vec._build import (
    atom_count,
    build,
    crossed,
    drop_index,
    install_triggers,
)
from quipu.vec._meta import (
    ensure_meta_table,
    get_build_status,
    is_build_complete,
)
from quipu.storage.store import pack_embedding
from quipu.models.cache import active_dim


def _unit_vec(i: int, dim: int = 384) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


# ---------------------------------------------------------------------------
# atom_count — no extension required
# ---------------------------------------------------------------------------

class TestAtomCount:
    def test_zero_on_empty(self, tmp_conn):
        assert atom_count(tmp_conn) == 0

    def test_count_matches_inserts(self, tmp_conn):
        for i in range(4):
            tmp_conn.execute(
                "INSERT INTO atoms (id, type, scope, content, project_id, invalidated)"
                " VALUES (?, 'diary', 'project', 'c', 'p', 0)",
                (f"id{i}",),
            )
        tmp_conn.commit()
        assert atom_count(tmp_conn) == 4


# ---------------------------------------------------------------------------
# crossed — threshold boolean; no extension required
# ---------------------------------------------------------------------------

class TestCrossed:
    def test_below(self, tmp_conn):
        assert crossed(tmp_conn, 10) is False

    def test_exactly_at_threshold(self, seeded_conn):
        # seeded_conn has 5 atoms
        assert crossed(seeded_conn, 5) is True

    def test_above_threshold(self, seeded_conn):
        assert crossed(seeded_conn, 3) is True

    def test_one_below(self, seeded_conn):
        assert crossed(seeded_conn, 6) is False


# ---------------------------------------------------------------------------
# drop_index on a fresh DB — no extension required, no build() ever called
# ---------------------------------------------------------------------------

class TestDropIndexFreshDb:
    def test_drop_index_on_fresh_db_no_crash(self, tmp_conn):
        """drop_index must not raise when build() was never called (meta table absent)."""
        # Precondition: no ember_vec_meta table exists yet.
        tables = {
            r[0]
            for r in tmp_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' OR type='shadow'"
            ).fetchall()
        }
        assert "ember_vec_meta" not in tables

        # Must be a no-op, not a crash.
        drop_index(tmp_conn)

        # After the call: meta table now exists (created by ensure_meta_table),
        # but the build row is still absent — drop was a clean no-op.
        status = get_build_status(tmp_conn)
        assert status is None


# ---------------------------------------------------------------------------
# build — requires sqlite_vec (vec0 virtual table)
# ---------------------------------------------------------------------------

sqlite_vec = pytest.importorskip("sqlite_vec")


def _load_ext(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


class TestBuild:
    def test_build_creates_meta_complete(self, seeded_conn):
        _load_ext(seeded_conn)
        build(seeded_conn)
        assert is_build_complete(seeded_conn)

    def test_build_populates_atoms_vec(self, seeded_conn):
        _load_ext(seeded_conn)
        build(seeded_conn)
        count = seeded_conn.execute("SELECT count(*) FROM atoms_vec").fetchone()[0]
        assert count == 5  # seeded_conn has 5 atoms with embeddings

    def test_build_idempotent_no_duplicates(self, seeded_conn):
        _load_ext(seeded_conn)
        build(seeded_conn)
        build(seeded_conn)  # second call — no-op
        count = seeded_conn.execute("SELECT count(*) FROM atoms_vec").fetchone()[0]
        assert count == 5

    def test_build_resumes_from_building(self, seeded_conn):
        """status='building' left by a crash → re-run completes."""
        _load_ext(seeded_conn)
        ensure_meta_table(seeded_conn)
        # Simulate a crash mid-build: table created, status='building', NO rows.
        seeded_conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS atoms_vec USING vec0(embedding float[384])"
        )
        from quipu.vec._meta import set_build_status
        set_build_status(seeded_conn, "building", dim=384)
        seeded_conn.commit()

        assert get_build_status(seeded_conn) == "building"

        # Re-run build — should complete.
        build(seeded_conn)
        assert is_build_complete(seeded_conn)
        count = seeded_conn.execute("SELECT count(*) FROM atoms_vec").fetchone()[0]
        assert count == 5


class TestDropIndex:
    def test_drop_removes_table_and_meta(self, seeded_conn):
        _load_ext(seeded_conn)
        build(seeded_conn)
        assert is_build_complete(seeded_conn)

        drop_index(seeded_conn)

        # atoms_vec gone
        tables = {
            r[0]
            for r in seeded_conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "atoms_vec" not in tables
        # meta row gone
        assert get_build_status(seeded_conn) is None

    def test_drop_then_rebuild(self, seeded_conn):
        _load_ext(seeded_conn)
        build(seeded_conn)
        drop_index(seeded_conn)
        # Must be able to rebuild cleanly.
        build(seeded_conn)
        assert is_build_complete(seeded_conn)

    def test_blobs_untouched_after_drop(self, seeded_conn):
        _load_ext(seeded_conn)
        build(seeded_conn)
        # Snapshot embeddings before.
        before = seeded_conn.execute(
            "SELECT id, embedding FROM atoms ORDER BY id"
        ).fetchall()
        drop_index(seeded_conn)
        after = seeded_conn.execute(
            "SELECT id, embedding FROM atoms ORDER BY id"
        ).fetchall()
        assert before == after, "BLOBs must be untouched by drop_index"


class TestInstallTriggers:
    def test_install_triggers_idempotent(self, seeded_conn):
        """install_triggers uses IF NOT EXISTS — calling twice is safe."""
        _load_ext(seeded_conn)
        build(seeded_conn)
        # Second call must not raise.
        install_triggers(seeded_conn)
        count = seeded_conn.execute("SELECT count(*) FROM atoms_vec").fetchone()[0]
        assert count == 5


# ---------------------------------------------------------------------------
# TASK-053 — vec0 DDL is parameterized by active_dim(), not hardcoded 384
# ---------------------------------------------------------------------------

class TestBuildDimAgnostic:
    def test_atoms_vec_ddl_uses_active_dim(self, tmp_conn, monkeypatch):
        """The CREATE VIRTUAL TABLE statement embeds float[<active_dim()>]."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")  # dim=384
        _load_ext(tmp_conn)
        build(tmp_conn)

        row = tmp_conn.execute(
            "SELECT sql FROM sqlite_master WHERE name='atoms_vec'"
        ).fetchone()
        assert row is not None
        assert f"float[{active_dim()}]" in row[0]
        assert active_dim() == 384

    def test_ember_vec_meta_dim_matches_active_dim(self, tmp_conn, monkeypatch):
        """set_build_status records dim == active_dim() at build time."""
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")  # dim=1024
        _load_ext(tmp_conn)
        build(tmp_conn)

        from quipu.vec._meta import get_build_dim
        assert get_build_dim(tmp_conn) == active_dim() == 1024
