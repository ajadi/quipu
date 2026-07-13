"""Tests for quipu.retrieval.tiers (R0, R1, R2, R3 tier functions)."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass, field

import pytest
from tests._semantic import TEST_EMBED_DIM

from quipu.retrieval.tiers import tier_r0, tier_r1, tier_r2, tier_r3
from quipu.storage.store import pack_embedding
def _unit_vec(index: int) -> list[float]:
    v = [0.0] * TEST_EMBED_DIM
    v[index] = 1.0
    return v


@dataclass
class _Atom:
    id: str
    content: str
    embedding: bytes | None = None
    project_id: str | None = None
    type: str = "diary"
    scope: str = "project"
    metadata: dict = field(default_factory=dict)
    refs: list = field(default_factory=list)
    invalidated: bool = False
    created_at: str = ""
    updated_at: str = ""
    session_id: str | None = None
    access_count: int = 0
    last_accessed: str | None = None
    tags: list[str] | None = None


# ---------------------------------------------------------------------------
# R0 — exact match
# ---------------------------------------------------------------------------

class TestTierR0:
    def test_exact_match(self):
        atoms = [
            _Atom(id="a", content="hello world"),  # type: ignore[arg-type]
            _Atom(id="b", content="hello"),  # type: ignore[arg-type]
            _Atom(id="c", content="goodbye"),  # type: ignore[arg-type]
        ]
        results = tier_r0(atoms, "hello", top_k=10)  # type: ignore[arg-type]
        assert len(results) == 1
        assert results[0].atom.id == "b"
        assert results[0].score == 1.0
        assert results[0].tier == "R0"

    def test_no_match(self):
        atoms = [_Atom(id="a", content="something else")]  # type: ignore[arg-type]
        assert tier_r0(atoms, "missing", top_k=10) == []  # type: ignore[arg-type]

    def test_top_k(self):
        atoms = [_Atom(id=str(i), content="exact") for i in range(5)]  # type: ignore[arg-type]
        results = tier_r0(atoms, "exact", top_k=3)  # type: ignore[arg-type]
        assert len(results) == 3

    def test_empty_atoms(self):
        assert tier_r0([], "query", top_k=5) == []

    def test_multiple_exact_matches(self):
        atoms = [
            _Atom(id="a", content="match"),  # type: ignore[arg-type]
            _Atom(id="b", content="no"),  # type: ignore[arg-type]
            _Atom(id="c", content="match"),  # type: ignore[arg-type]
        ]
        results = tier_r0(atoms, "match", top_k=10)  # type: ignore[arg-type]
        assert len(results) == 2
        ids = {r.atom.id for r in results}
        assert ids == {"a", "c"}


# ---------------------------------------------------------------------------
# R1 — cosine
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestTierR1:
    def test_ordered_by_cosine_desc(self):
        """Atom B at index 1 closer to query at index 1 than atom A at index 0."""
        atom_a = _Atom(id="A", content="a", embedding=pack_embedding(_unit_vec(0)))  # type: ignore[arg-type]
        atom_b = _Atom(id="B", content="b", embedding=pack_embedding(_unit_vec(1)))  # type: ignore[arg-type]
        query_vec = _unit_vec(1)  # closest to B

        results = tier_r1([atom_a, atom_b], query_vec, top_k=10)  # type: ignore[arg-type]
        assert results[0].atom.id == "B"
        assert results[1].atom.id == "A"
        # Scores strictly descending
        assert results[0].score > results[1].score
        assert all(r.tier == "R1" for r in results)

    def test_skips_none_embedding(self):
        atom_a = _Atom(id="A", content="a", embedding=None)  # type: ignore[arg-type]
        atom_b = _Atom(id="B", content="b", embedding=pack_embedding(_unit_vec(0)))  # type: ignore[arg-type]
        results = tier_r1([atom_a, atom_b], _unit_vec(0), top_k=10)  # type: ignore[arg-type]
        assert len(results) == 1
        assert results[0].atom.id == "B"

    def test_top_k(self):
        atoms = [
            _Atom(id=str(i), content=str(i), embedding=pack_embedding(_unit_vec(i)))  # type: ignore[arg-type]
            for i in range(5)
        ]
        results = tier_r1(atoms, _unit_vec(0), top_k=3)  # type: ignore[arg-type]
        assert len(results) == 3

    def test_empty_atoms(self):
        assert tier_r1([], _unit_vec(0), top_k=5) == []

    def test_all_none_embeddings(self):
        atoms = [_Atom(id="a", content="a", embedding=None)]  # type: ignore[arg-type]
        assert tier_r1(atoms, _unit_vec(0), top_k=5) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# R2 — BM25
# ---------------------------------------------------------------------------

class TestTierR2:
    def test_ordered_by_bm25_desc(self):
        atoms = [
            _Atom(id="rich", content="python python python programming language"),  # type: ignore[arg-type]
            _Atom(id="poor", content="python"),  # type: ignore[arg-type]
        ]
        results = tier_r2(atoms, "python", top_k=10)  # type: ignore[arg-type]
        # 'rich' has higher TF → should rank first or tied
        assert len(results) >= 1
        # All scores positive (negated FTS5 bm25)
        for r in results:
            assert r.score > 0
            assert r.tier == "R2"
        # Scores descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_no_match_returns_empty(self):
        atoms = [_Atom(id="a", content="completely unrelated")]  # type: ignore[arg-type]
        assert tier_r2(atoms, "python", top_k=10) == []  # type: ignore[arg-type]

    def test_top_k(self):
        atoms = [
            _Atom(id=str(i), content=f"python programming example {i}")  # type: ignore[arg-type]
            for i in range(6)
        ]
        results = tier_r2(atoms, "python", top_k=3)  # type: ignore[arg-type]
        assert len(results) == 3

    def test_empty_atoms(self):
        assert tier_r2([], "query", top_k=5) == []


# ---------------------------------------------------------------------------
# R3 — fusion
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestTierR3:
    def _make_atoms(self):
        """Three atoms: X wins cosine, Y wins BM25, Z is mid."""
        atom_x = _Atom(
            id="X",
            content="unrelated text xyz",
            embedding=pack_embedding(_unit_vec(0)),  # closest to query
        )  # type: ignore[arg-type]
        atom_y = _Atom(
            id="Y",
            content="python python python language",
            embedding=pack_embedding(_unit_vec(100)),  # far from query
        )  # type: ignore[arg-type]
        atom_z = _Atom(
            id="Z",
            content="python programming",
            embedding=pack_embedding(_unit_vec(50)),  # mid distance
        )  # type: ignore[arg-type]
        return [atom_x, atom_y, atom_z]

    def test_r3_dedup(self):
        """R3 must not produce duplicate atom ids."""
        atoms = self._make_atoms()
        query_vec = _unit_vec(0)
        results = tier_r3(atoms, "python", query_vec, top_k=10)  # type: ignore[arg-type]
        ids = [r.atom.id for r in results]
        assert len(ids) == len(set(ids))

    def test_r3_tier_label(self):
        atoms = self._make_atoms()
        results = tier_r3(atoms, "python", _unit_vec(0), top_k=10)  # type: ignore[arg-type]
        assert all(r.tier == "R3" for r in results)

    def test_r3_scores_in_01(self):
        atoms = self._make_atoms()
        results = tier_r3(atoms, "python", _unit_vec(0), top_k=10)  # type: ignore[arg-type]
        for r in results:
            assert 0.0 <= r.score <= 1.0

    def test_r3_top_k(self):
        atoms = self._make_atoms()
        results = tier_r3(atoms, "python", _unit_vec(0), top_k=2)  # type: ignore[arg-type]
        assert len(results) == 2

    def test_r3_cosine_only_atom_appears(self):
        """Atom with embedding but no BM25 match still appears in R3."""
        atom_cos = _Atom(
            id="cos_only",
            content="zzz no keyword match here",
            embedding=pack_embedding(_unit_vec(0)),
        )  # type: ignore[arg-type]
        atom_bm25 = _Atom(
            id="bm25_only",
            content="python programming language",
            embedding=None,  # no embedding
        )  # type: ignore[arg-type]
        results = tier_r3([atom_cos, atom_bm25], "python", _unit_vec(0), top_k=10)  # type: ignore[arg-type]
        ids = {r.atom.id for r in results}
        assert "cos_only" in ids
        assert "bm25_only" in ids

    def test_r3_empty_atoms(self):
        results = tier_r3([], "python", _unit_vec(0), top_k=5)
        assert results == []
