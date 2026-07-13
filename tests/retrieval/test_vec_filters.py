"""Regression coverage for session and tag filters on the R1 vec route."""

from __future__ import annotations

from dataclasses import replace
import sys
from pathlib import Path

import pytest
from tests._semantic import TEST_EMBED_DIM

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.retrieval._search import search
from quipu.storage.store import pack_embedding


def _unit_vec(index: int) -> list[float]:
    vector = [0.0] * TEST_EMBED_DIM
    vector[index] = 1.0
    return vector


def _enable_vec_route(monkeypatch, rowid_scores):
    monkeypatch.setattr(
        "quipu.retrieval._search.embed", lambda query: _unit_vec(0)
    )
    import quipu.vec as vec_mod

    monkeypatch.setattr(vec_mod, "query_ready", lambda conn: True)
    monkeypatch.setattr(
        vec_mod, "knn", lambda conn, query_vec, project_id, top_k: rowid_scores[:top_k]
    )


def _rowid(store, atom_id: str) -> int:
    return store._conn.execute(
        "SELECT rowid FROM atoms WHERE id = ?", (atom_id,)
    ).fetchone()[0]


pytestmark = pytest.mark.usefixtures("semantic_model")


def test_vec_r1_honors_session_id_filter(tmp_store, monkeypatch):
    other = tmp_store.insert(
        content="other session", embedding=pack_embedding(_unit_vec(0)),
        project_id="project", session_id="other",
    )
    expected = tmp_store.insert(
        content="requested session", embedding=pack_embedding(_unit_vec(1)),
        project_id="project", session_id="requested",
    )
    _enable_vec_route(
        monkeypatch,
        [(_rowid(tmp_store, other.id), 0.99), (_rowid(tmp_store, expected.id), 0.90)],
    )

    results = search(
        "session", tier="R1", project_id="project", session_id="requested",
        top_k=1, store=tmp_store,
    )

    assert [result.atom.id for result in results] == [expected.id]


def test_vec_r1_honors_tags_filter(tmp_store, monkeypatch):
    other = tmp_store.insert(
        content="other tag", embedding=pack_embedding(_unit_vec(0)),
        project_id="project", tags=["other"],
    )
    expected = tmp_store.insert(
        content="requested tag", embedding=pack_embedding(_unit_vec(1)),
        project_id="project", tags=["requested"],
    )
    _enable_vec_route(
        monkeypatch,
        [(_rowid(tmp_store, other.id), 0.99), (_rowid(tmp_store, expected.id), 0.90)],
    )

    results = search(
        "tag", tier="R1", project_id="project", tags=["REQUESTED"],
        top_k=1, store=tmp_store,
    )

    assert [result.atom.id for result in results] == [expected.id]


def test_vec_r1_large_filtered_candidate_set_uses_exact_python_path(tmp_store, monkeypatch):
    expected = tmp_store.insert(
        content="requested session", embedding=pack_embedding(_unit_vec(0)),
        project_id="project", session_id="requested",
    )
    candidates = [expected] + [
        replace(expected, id=f"other-{index}", session_id="other")
        for index in range(10_000)
    ]
    monkeypatch.setattr(
        tmp_store, "list_by_project", lambda *args, **kwargs: candidates
    )
    monkeypatch.setattr(
        "quipu.retrieval._search.embed", lambda query: _unit_vec(0)
    )
    import quipu.vec as vec_mod

    monkeypatch.setattr(vec_mod, "query_ready", lambda conn: True)
    monkeypatch.setattr(
        vec_mod, "knn", lambda *args, **kwargs: pytest.fail("vec KNN buffer is insufficient")
    )

    results = search(
        "session", tier="R1", project_id="project", session_id="requested",
        top_k=1, store=tmp_store,
    )

    assert [result.atom.id for result in results] == [expected.id]


@pytest.mark.parametrize("tier", ["R1", "R3"])
def test_vec_ready_filters_remain_identical_for_r1_and_r3(tmp_store, monkeypatch, tier):
    expected = tmp_store.insert(
        content="requested item", embedding=pack_embedding(_unit_vec(0)),
        project_id="project", session_id="requested", tags=["requested"],
    )
    tmp_store.insert(
        content="other item", embedding=pack_embedding(_unit_vec(1)),
        project_id="project", session_id="other", tags=["other"],
    )
    _enable_vec_route(monkeypatch, [(_rowid(tmp_store, expected.id), 0.90)])

    results = search(
        "item", tier=tier, project_id="project", session_id="requested",
        tags=["requested"], top_k=5, store=tmp_store,
    )

    assert [result.atom.id for result in results] == [expected.id]
