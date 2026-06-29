"""Tests for quipu.ranking.fusion (normalize_scores, fuse)."""

from __future__ import annotations

import pytest
from dataclasses import dataclass, field
from unittest.mock import MagicMock

from quipu.ranking.fusion import normalize_scores, fuse
from quipu.ranking.result import SearchResult


# ---------------------------------------------------------------------------
# Minimal Atom stub for testing (no storage import needed)
# ---------------------------------------------------------------------------

@dataclass
class _Atom:
    id: str
    content: str = ""
    embedding: bytes | None = None
    project_id: str | None = None
    type: str = "diary"
    scope: str = "project"
    metadata: dict = field(default_factory=dict)
    refs: list = field(default_factory=list)
    invalidated: bool = False
    created_at: str = ""
    updated_at: str = ""
    access_count: int = 0


def _sr(atom_id: str, score: float, tier: str = "R1") -> SearchResult:
    atom = _Atom(id=atom_id)
    return SearchResult(atom=atom, score=score, tier=tier)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# normalize_scores
# ---------------------------------------------------------------------------

def test_normalize_empty():
    assert normalize_scores([]) == []


def test_normalize_single():
    """Single element → all-equal → 1.0."""
    assert normalize_scores([5.0]) == [1.0]


def test_normalize_all_equal():
    """All-equal values → all 1.0."""
    assert normalize_scores([3.0, 3.0, 3.0]) == [1.0, 1.0, 1.0]


def test_normalize_range():
    scores = [0.0, 0.5, 1.0]
    result = normalize_scores(scores)
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_negative_inputs():
    """Works with negative scores (e.g. raw cosine can be negative)."""
    scores = [-1.0, 0.0, 1.0]
    result = normalize_scores(scores)
    assert result == pytest.approx([0.0, 0.5, 1.0])


def test_normalize_preserves_order():
    scores = [3.0, 1.0, 2.0]
    result = normalize_scores(scores)
    # 3→1.0, 1→0.0, 2→0.5
    assert result == pytest.approx([1.0, 0.0, 0.5])


# ---------------------------------------------------------------------------
# fuse
# ---------------------------------------------------------------------------

def test_fuse_empty_both():
    result = fuse([], [])
    assert result == []


def test_fuse_r1_only():
    r1 = [_sr("a", 1.0), _sr("b", 0.5)]
    result = fuse(r1, [])
    ids = [r.atom.id for r in result]
    # Both from R1 only; all-equal BM25 contrib = 0
    assert set(ids) == {"a", "b"}
    # a should rank above b (higher cosine)
    assert result[0].atom.id == "a"
    for r in result:
        assert r.tier == "R3"


def test_fuse_r2_only():
    r2 = [_sr("x", 2.0, "R2"), _sr("y", 1.0, "R2")]
    result = fuse([], r2)
    assert result[0].atom.id == "x"
    assert all(r.tier == "R3" for r in result)


def test_fuse_dedup_same_id():
    """Same atom in R1 and R2 → one result."""
    r1 = [_sr("z", 0.9)]
    r2 = [_sr("z", 1.5, "R2")]
    result = fuse(r1, r2)
    assert len(result) == 1
    assert result[0].atom.id == "z"
    assert result[0].tier == "R3"


def test_fuse_ordering():
    """X wins cosine, Y wins BM25, Z is mid — check ordering consistency."""
    # atom X: high cosine, low BM25
    # atom Y: low cosine, high BM25
    # atom Z: mid both
    r1 = [
        _sr("X", 1.0),   # cosine = 1.0 (normalized → 1.0)
        _sr("Z", 0.5),   # cosine = 0.5 (normalized → 0.5)
        _sr("Y", 0.0),   # cosine = 0.0 (normalized → 0.0)
    ]
    r2 = [
        _sr("Y", 2.0, "R2"),  # BM25 = 2.0 (normalized → 1.0)
        _sr("Z", 1.0, "R2"),  # BM25 = 1.0 (normalized → 0.5)
        _sr("X", 0.0, "R2"),  # BM25 = 0.0 (normalized → 0.0)
    ]
    # With w_cos=0.6, w_bm25=0.4:
    # X: 0.6*1.0 + 0.4*0.0 = 0.6
    # Y: 0.6*0.0 + 0.4*1.0 = 0.4
    # Z: 0.6*0.5 + 0.4*0.5 = 0.5
    # Order: X(0.6) > Z(0.5) > Y(0.4)
    result = fuse(r1, r2, w_cos=0.6, w_bm25=0.4)
    ids = [r.atom.id for r in result]
    assert ids == ["X", "Z", "Y"]


def test_fuse_scores_in_01():
    """Fused scores should be in [0, 1]."""
    r1 = [_sr("a", 0.9), _sr("b", 0.1)]
    r2 = [_sr("b", 3.0, "R2"), _sr("c", 1.0, "R2")]
    result = fuse(r1, r2)
    for r in result:
        assert 0.0 <= r.score <= 1.0


# ---------------------------------------------------------------------------
# TASK-021 — access-frequency boost
# ---------------------------------------------------------------------------

def test_fuse_access_boost_raises_high_access():
    """Atom with high access_count gets a boost over identical signals."""
    r1 = [_sr("x", 0.8), _sr("y", 0.8)]
    r2 = []
    access = {"x": 100, "y": 0}

    result = fuse(r1, r2, w_access=0.5, access_counts=access)
    # x should rank higher than y due to access boost
    assert result[0].atom.id == "x"
    assert result[1].atom.id == "y"

def test_fuse_access_boost_same_signals():
    """All else equal, sorting respects access boost."""
    r1 = [_sr("a", 1.0), _sr("b", 1.0), _sr("c", 1.0)]
    r2 = []
    access = {"a": 50, "b": 10, "c": 0}

    result = fuse(r1, r2, w_access=1.0, access_counts=access)
    ids = [r.atom.id for r in result]
    assert ids == ["a", "b", "c"]

def test_fuse_access_boost_no_boost_when_none():
    """When access_counts is None, behaviour is unchanged."""
    r1 = [_sr("a", 0.9), _sr("b", 0.5)]
    r2 = []
    result_no = fuse(r1, r2, w_access=0.5, access_counts=None)
    result_zero = fuse(r1, r2, w_access=0.0, access_counts={"a": 100, "b": 0})
    # Both should produce same ordering (by original score)
    assert result_no[0].atom.id == "a"
    assert result_zero[0].atom.id == "a"

def test_fuse_access_boost_missing_ids():
    """Atoms not in access_counts get access_count=0 treatment."""
    r1 = [_sr("a", 0.8), _sr("b", 0.8)]
    r2 = []
    access = {"a": 50}  # b not included

    result = fuse(r1, r2, w_access=1.0, access_counts=access)
    assert result[0].atom.id == "a"
    assert result[1].atom.id == "b"
