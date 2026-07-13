"""Tests for quipu.vec._query — query_ready and knn.

query_ready's False/guard branches run always (no sqlite_vec needed).
The knn real-extension path is gated with pytest.importorskip("sqlite_vec").
"""

from __future__ import annotations

import sqlite3

import pytest

from quipu.vec._query import knn, query_ready
from quipu.vec._meta import ensure_meta_table, set_build_status
from quipu.storage.store import pack_embedding


def _unit_vec(i: int, dim: int = 384) -> list[float]:
    v = [0.0] * dim
    v[i % dim] = 1.0
    return v


# ---------------------------------------------------------------------------
# query_ready — False branches; no extension required
# ---------------------------------------------------------------------------

class TestQueryReadyFalse:
    def test_false_when_no_extension(self, tmp_conn):
        """query_ready is False when sqlite-vec is not loaded."""
        assert query_ready(tmp_conn) is False

    def test_false_when_meta_absent(self, tmp_conn, monkeypatch):
        """query_ready is False when extension 'loaded' but no meta row."""
        monkeypatch.setattr("quipu.vec._query.is_loaded", lambda conn: True)
        # No meta table at all → is_build_complete returns False.
        assert query_ready(tmp_conn) is False

    def test_false_when_status_building(self, tmp_conn, monkeypatch):
        """query_ready is False when build status is 'building', not 'complete'."""
        monkeypatch.setattr("quipu.vec._query.is_loaded", lambda conn: True)
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "building", dim=384)
        tmp_conn.commit()
        assert query_ready(tmp_conn) is False

    def test_true_when_loaded_and_complete(self, tmp_conn, monkeypatch):
        """query_ready is True when both conditions are met."""
        monkeypatch.setattr("quipu.vec._query.is_loaded", lambda conn: True)
        ensure_meta_table(tmp_conn)
        set_build_status(tmp_conn, "complete", dim=384)
        tmp_conn.commit()
        assert query_ready(tmp_conn) is True


# ---------------------------------------------------------------------------
# knn — real extension path
# ---------------------------------------------------------------------------

sqlite_vec = pytest.importorskip("sqlite_vec")


def _load_ext(conn: sqlite3.Connection) -> None:
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


@pytest.mark.usefixtures("semantic_model")
class TestKnn:
    def _build_index(self, conn):
        from quipu.vec._build import build
        build(conn)

    def test_knn_returns_correct_count(self, seeded_conn):
        """knn returns at most top_k results."""
        _load_ext(seeded_conn)
        self._build_index(seeded_conn)
        results = knn(seeded_conn, _unit_vec(0), project_id="p", top_k=3)
        assert len(results) <= 3

    def test_knn_score_higher_is_better(self, seeded_conn):
        """Scores are in dot/cosine scale — higher means more similar."""
        _load_ext(seeded_conn)
        self._build_index(seeded_conn)
        # Query at index 0 — atom0 (unit vec at 0) should score highest.
        results = knn(seeded_conn, _unit_vec(0), project_id="p", top_k=5)
        assert len(results) >= 1
        scores = [s for _, s in results]
        # Scores should be in descending order (higher == better).
        assert scores == sorted(scores, reverse=True)

    def test_knn_best_match_first(self, seeded_conn):
        """The atom most similar to the query appears first."""
        _load_ext(seeded_conn)
        self._build_index(seeded_conn)
        # Query identical to atom0's embedding → atom0 rowid should rank first.
        results = knn(seeded_conn, _unit_vec(0), project_id="p", top_k=5)
        assert len(results) >= 1
        best_rowid = results[0][0]
        # Fetch that atom and confirm it has embedding == unit_vec(0).
        row = seeded_conn.execute(
            "SELECT embedding FROM atoms WHERE rowid = ?", (best_rowid,)
        ).fetchone()
        assert row is not None
        from quipu.storage.store import unpack_embedding
        from quipu.ranking.cosine import dot
        vec = unpack_embedding(row[0])
        score = dot(_unit_vec(0), vec)
        assert score > 0.99, f"Expected near-1 dot score, got {score}"

    def test_knn_project_filter(self, tmp_conn):
        """knn only returns atoms from the specified project_id."""
        _load_ext(tmp_conn)
        # Insert atoms in two projects.
        for i in range(3):
            tmp_conn.execute(
                "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
                " VALUES (?, 'diary', 'project', ?, ?, 'proj_a', 0)",
                (f"a{i}", f"content {i}", pack_embedding(_unit_vec(i))),
            )
        for i in range(2):
            tmp_conn.execute(
                "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
                " VALUES (?, 'diary', 'project', ?, ?, 'proj_b', 0)",
                (f"b{i}", f"content {i}", pack_embedding(_unit_vec(i))),
            )
        tmp_conn.commit()
        self._build_index(tmp_conn)

        results_a = knn(tmp_conn, _unit_vec(0), project_id="proj_a", top_k=10)
        results_b = knn(tmp_conn, _unit_vec(0), project_id="proj_b", top_k=10)
        assert len(results_a) == 3
        assert len(results_b) == 2

    def test_knn_invalidated_filtered(self, tmp_conn):
        """knn excludes invalidated atoms."""
        _load_ext(tmp_conn)
        tmp_conn.execute(
            "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
            " VALUES ('valid', 'diary', 'project', 'c', ?, 'p', 0)",
            (pack_embedding(_unit_vec(0)),),
        )
        tmp_conn.execute(
            "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
            " VALUES ('invalid', 'diary', 'project', 'c', ?, 'p', 1)",
            (pack_embedding(_unit_vec(0)),),
        )
        tmp_conn.commit()
        self._build_index(tmp_conn)

        results = knn(tmp_conn, _unit_vec(0), project_id="p", top_k=10)
        assert len(results) == 1

    def test_knn_score_is_cosine_dot_scale(self, tmp_conn):
        """Dot score for identical unit vecs should be ~1.0."""
        _load_ext(tmp_conn)
        tmp_conn.execute(
            "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
            " VALUES ('a', 'diary', 'project', 'c', ?, 'p', 0)",
            (pack_embedding(_unit_vec(0)),),
        )
        tmp_conn.commit()
        self._build_index(tmp_conn)

        results = knn(tmp_conn, _unit_vec(0), project_id="p", top_k=1)
        assert len(results) == 1
        _, score = results[0]
        assert abs(score - 1.0) < 0.01, f"Expected dot~1.0 for identical vecs, got {score}"

    def test_knn_limit_fix_returns_top_k_after_filter(self, tmp_conn):
        """knn returns up to top_k after project/invalidated filter (FIX 2).

        Insert top_k*10+1 atoms but mark all except top_k as invalidated.
        Previously, LIMIT=top_k meant filtered rows could under-deliver;
        with LIMIT=fetch_k the buffer survives filtering and top_k survive.
        """
        _load_ext(tmp_conn)
        top_k = 3
        # Insert top_k valid atoms.
        for i in range(top_k):
            tmp_conn.execute(
                "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
                " VALUES (?, 'diary', 'project', ?, ?, 'p', 0)",
                (f"valid{i}", f"c{i}", pack_embedding(_unit_vec(i))),
            )
        # Insert top_k*(10-1) invalidated atoms in same project.
        # These fill positions that the old LIMIT=top_k would have consumed,
        # causing under-delivery of valid rows.
        for i in range(top_k * 9):
            tmp_conn.execute(
                "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
                " VALUES (?, 'diary', 'project', ?, ?, 'p', 1)",
                (f"inv{i}", f"c{i}", pack_embedding(_unit_vec(i % 384))),
            )
        tmp_conn.commit()
        self._build_index(tmp_conn)

        results = knn(tmp_conn, _unit_vec(0), project_id="p", top_k=top_k)
        assert len(results) == top_k, (
            f"Expected {top_k} results after filter, got {len(results)}"
        )
