"""Tests for project_id filter — records from other projects must not appear."""

from __future__ import annotations

import pytest

from quipu.retrieval._search import search
from quipu.storage.store import pack_embedding

EMBED_DIM = 384


def _unit_vec(index: int) -> list[float]:
    v = [0.0] * EMBED_DIM
    v[index] = 1.0
    return v


def _fake_embed(text: str) -> list[float]:
    return _unit_vec(0)


@pytest.fixture()
def seeded_store(tmp_store):
    """Store with atoms in two projects."""
    tmp_store.insert(
        content="project one content python",
        embedding=pack_embedding(_unit_vec(0)),
        project_id="p1",
    )
    tmp_store.insert(
        content="project one more python data",
        embedding=pack_embedding(_unit_vec(1)),
        project_id="p1",
    )
    tmp_store.insert(
        content="project two content python",
        embedding=pack_embedding(_unit_vec(2)),
        project_id="p2",
    )
    return tmp_store


@pytest.mark.parametrize("tier", ["R0", "R1", "R2", "R3"])
def test_project_filter_no_cross_project(tier, seeded_store, monkeypatch):
    """Atoms from p2 must never appear when searching p1."""
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    query = "project one content python" if tier == "R0" else "python"
    results = search(query, tier=tier, project_id="p1", top_k=20, store=seeded_store)

    for r in results:
        assert r.atom.project_id == "p1", (
            f"tier={tier}: got atom from project {r.atom.project_id!r}"
        )


def test_project_id_none_raises(tmp_store):
    with pytest.raises(ValueError, match="project_id is required"):
        search("query", project_id=None, store=tmp_store)


def test_search_empty_project_returns_empty(tmp_store, monkeypatch):
    """Searching a project with no atoms returns empty list."""
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)
    results = search("anything", tier="R3", project_id="nonexistent", store=tmp_store)
    assert results == []
