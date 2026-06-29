"""CRITICAL regression guard: R1 vec routing.

Two scenarios:
1. query_ready=False (default, no extension) — search(tier="R1") must return
   results identical to calling tier_r1 directly on the same atoms.
2. query_ready=True (monkeypatched) + fake knn — search routes through the vec
   path and _results_from_rowids reconstructs the correct atoms/scores/tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.retrieval._search import search, _results_from_rowids
from quipu.retrieval.tiers import tier_r1
from quipu.storage.store import pack_embedding
from quipu.ranking.cosine import dot
from quipu.storage.store import unpack_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

EMBED_DIM = 384


def _unit_vec(i: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[i % EMBED_DIM] = 1.0
    return v


def _fake_embed_factory(vec: list[float]):
    def _embed(text: str) -> list[float]:
        return vec
    return _embed


# ---------------------------------------------------------------------------
# Fixture: seeded store with 5 atoms
# ---------------------------------------------------------------------------

@pytest.fixture()
def seeded_store(tmp_store):
    for i in range(5):
        tmp_store.insert(
            content=f"atom content {i}",
            embedding=pack_embedding(_unit_vec(i)),
            project_id="p",
        )
    return tmp_store


# ---------------------------------------------------------------------------
# Scenario 1: query_ready=False → pure-Python tier_r1, byte-identical results
# ---------------------------------------------------------------------------

class TestR1FallbackPurePython:
    """With no vec extension, search(tier="R1") must be identical to tier_r1."""

    def test_same_atom_ids(self, seeded_store, monkeypatch):
        query_vec = _unit_vec(2)
        monkeypatch.setattr(
            "quipu.retrieval._search.embed", _fake_embed_factory(query_vec)
        )

        atoms = seeded_store.list_by_project("p", include_invalidated=False)
        direct = tier_r1(atoms, query_vec, top_k=5)
        via_search = search("anything", tier="R1", project_id="p", top_k=5, store=seeded_store)

        assert [r.atom.id for r in via_search] == [r.atom.id for r in direct]

    def test_same_scores(self, seeded_store, monkeypatch):
        query_vec = _unit_vec(1)
        monkeypatch.setattr(
            "quipu.retrieval._search.embed", _fake_embed_factory(query_vec)
        )

        atoms = seeded_store.list_by_project("p", include_invalidated=False)
        direct = tier_r1(atoms, query_vec, top_k=5)
        via_search = search("anything", tier="R1", project_id="p", top_k=5, store=seeded_store)

        direct_scores = [r.score for r in direct]
        search_scores = [r.score for r in via_search]
        assert direct_scores == search_scores

    def test_tier_label_is_r1(self, seeded_store, monkeypatch):
        monkeypatch.setattr(
            "quipu.retrieval._search.embed", _fake_embed_factory(_unit_vec(0))
        )
        results = search("anything", tier="R1", project_id="p", top_k=5, store=seeded_store)
        assert all(r.tier == "R1" for r in results)

    def test_query_ready_false_by_default(self, seeded_store):
        """query_ready must be False when sqlite-vec is not loaded."""
        from quipu.vec._query import query_ready
        assert query_ready(seeded_store._conn) is False

    def test_descending_order(self, seeded_store, monkeypatch):
        """Results must be ordered by descending score."""
        monkeypatch.setattr(
            "quipu.retrieval._search.embed", _fake_embed_factory(_unit_vec(0))
        )
        results = search("anything", tier="R1", project_id="p", top_k=5, store=seeded_store)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# Scenario 2: query_ready=True + fake knn → vec path used
# ---------------------------------------------------------------------------

class TestR1VecRouting:
    """When query_ready is True, search(tier="R1") must use the vec path."""

    def test_routes_through_vec_path(self, seeded_store, monkeypatch):
        """With query_ready=True and a fake knn, results come from _results_from_rowids."""
        query_vec = _unit_vec(0)
        monkeypatch.setattr(
            "quipu.retrieval._search.embed", _fake_embed_factory(query_vec)
        )

        # Determine actual rowids by querying the store's connection.
        rows = seeded_store._conn.execute(
            "SELECT rowid, id FROM atoms WHERE project_id='p' AND invalidated=0 ORDER BY rowid"
        ).fetchall()
        # Fake knn returns (rowid, dot_score) — we control scores.
        fake_rowid_scores = [(row[0], 0.9 - i * 0.1) for i, row in enumerate(rows[:3])]

        def _fake_knn(conn, qvec, project_id, top_k):
            return fake_rowid_scores

        # _search.py does `from quipu.vec import query_ready, knn as vec_knn`
        # inline, so we must patch the names on the quipu.vec package itself.
        import quipu.vec as vec_mod
        monkeypatch.setattr(vec_mod, "query_ready", lambda conn: True)
        monkeypatch.setattr(vec_mod, "knn", _fake_knn)

        results = search("anything", tier="R1", project_id="p", top_k=3, store=seeded_store)

        # Results must match the fake knn's rowid ordering.
        assert len(results) == 3
        result_ids = [r.atom.id for r in results]
        expected_ids = [
            seeded_store._conn.execute(
                "SELECT id FROM atoms WHERE rowid=?", (rw,)
            ).fetchone()[0]
            for rw, _ in fake_rowid_scores
        ]
        assert result_ids == expected_ids

    def test_vec_scores_preserved(self, seeded_store, monkeypatch):
        """Scores from fake knn are preserved through _results_from_rowids."""
        rows = seeded_store._conn.execute(
            "SELECT rowid FROM atoms WHERE project_id='p' AND invalidated=0 ORDER BY rowid"
        ).fetchall()
        rowids = [r[0] for r in rows[:2]]
        fake_scores = [0.88, 0.72]
        fake_rowid_scores = list(zip(rowids, fake_scores))

        monkeypatch.setattr(
            "quipu.retrieval._search.embed", _fake_embed_factory(_unit_vec(0))
        )
        import quipu.vec as vec_mod
        monkeypatch.setattr(vec_mod, "knn", lambda conn, qvec, pid, tk: fake_rowid_scores)
        monkeypatch.setattr(vec_mod, "query_ready", lambda conn: True)

        results = search("anything", tier="R1", project_id="p", top_k=2, store=seeded_store)
        assert len(results) == 2
        assert abs(results[0].score - 0.88) < 1e-9
        assert abs(results[1].score - 0.72) < 1e-9

    def test_results_from_rowids_reconstructs_atom(self, seeded_store):
        """_results_from_rowids returns correct atom data."""
        row = seeded_store._conn.execute(
            "SELECT rowid, id FROM atoms WHERE project_id='p' LIMIT 1"
        ).fetchone()
        rowid, atom_id = row[0], row[1]

        results = _results_from_rowids(seeded_store, [(rowid, 0.77)], "R1")
        assert len(results) == 1
        assert results[0].atom.id == atom_id
        assert abs(results[0].score - 0.77) < 1e-9
        assert results[0].tier == "R1"

    def test_results_from_rowids_skips_missing(self, seeded_store):
        """_results_from_rowids silently skips rowids that no longer exist."""
        results = _results_from_rowids(seeded_store, [(999999, 0.5)], "R1")
        assert results == []

    def test_results_from_rowids_no_cross_project_leak(self, tmp_store):
        """Stale rowid reassigned to a different project must not leak (FIX 1).

        Simulates a rowid that exists in the DB under project 'other' being
        returned by the vec index for project 'p'.  The re-fetch filter
        (project_id AND invalidated=0) must exclude it → empty result.
        """
        # Insert one atom in project 'other'.
        tmp_store.insert(
            content="secret content in another project",
            embedding=pack_embedding(_unit_vec(0)),
            project_id="other",
        )
        # Get its rowid.
        row = tmp_store._conn.execute(
            "SELECT rowid FROM atoms WHERE project_id='other' LIMIT 1"
        ).fetchone()
        stale_rowid = row[0]

        # _results_from_rowids called as if the vec index returned this rowid
        # for project 'p' — project_id filter must exclude it.
        results = _results_from_rowids(
            tmp_store, [(stale_rowid, 0.99)], "R1", project_id="p"
        )
        assert results == [], (
            "Cross-project rowid must not leak through _results_from_rowids"
        )

    def test_results_from_rowids_no_cross_project_leak_invalidated(self, tmp_store):
        """Stale rowid that belongs to correct project but is invalidated is skipped."""
        tmp_store.insert(
            content="invalidated atom",
            embedding=pack_embedding(_unit_vec(1)),
            project_id="p",
        )
        # Invalidate it.
        row = tmp_store._conn.execute(
            "SELECT rowid, id FROM atoms WHERE project_id='p' LIMIT 1"
        ).fetchone()
        rowid, atom_id = row[0], row[1]
        tmp_store.update_invalidated(atom_id, True)

        results = _results_from_rowids(
            tmp_store, [(rowid, 0.9)], "R1", project_id="p"
        )
        assert results == [], "Invalidated atom must be excluded via project_id filter"


# ---------------------------------------------------------------------------
# Scenario 3: R3 identical vec-on vs vec-off (FIX 3)
# ---------------------------------------------------------------------------

class TestR3VecEquivalence:
    """R3 results must be identical whether vec is on or off.

    Since FIX 3 routes R3 entirely through pure-Python tier_r3, vec state
    must have zero effect on R3 output.
    """

    def test_r3_same_vec_on_vs_off(self, seeded_store, monkeypatch):
        """R3 results are identical regardless of query_ready state."""
        import quipu.vec as vec_mod

        query_vec = _unit_vec(0)
        monkeypatch.setattr(
            "quipu.retrieval._search.embed", _fake_embed_factory(query_vec)
        )

        # Result with vec OFF (default).
        monkeypatch.setattr(vec_mod, "query_ready", lambda conn: False)
        results_off = search("anything", tier="R3", project_id="p", top_k=5, store=seeded_store)

        # Result with vec ON (monkeypatched to True — knn should NOT be called).
        knn_called = []

        def _should_not_be_called(conn, qvec, pid, tk):
            knn_called.append(True)
            return []

        monkeypatch.setattr(vec_mod, "query_ready", lambda conn: True)
        monkeypatch.setattr(vec_mod, "knn", _should_not_be_called)
        results_on = search("anything", tier="R3", project_id="p", top_k=5, store=seeded_store)

        # knn must NOT have been called for R3.
        assert knn_called == [], "R3 must not invoke vec.knn regardless of query_ready"

        # Atom ordering must be identical.
        assert [r.atom.id for r in results_on] == [r.atom.id for r in results_off]
        # Scores must be identical (same pure-Python path).
        assert [r.score for r in results_on] == [r.score for r in results_off]
