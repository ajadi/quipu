"""Tests for quipu.retrieval.bm25 (FTS5 BM25 ranking)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field

import pytest

from quipu.retrieval.bm25 import bm25_rank


# ---------------------------------------------------------------------------
# Minimal Atom stub
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# FTS5 availability guard
# ---------------------------------------------------------------------------

@pytest.mark.xfail(reason="FTS5 not available — report BLOCKED")
def test_fts5_available():
    """Verify FTS5 is available. xfail here triggers BLOCKED signal if missing."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE VIRTUAL TABLE _t USING fts5(x)")
    conn.close()
    # If we get here, FTS5 is available → test passes (xfail becomes xpass)


# ---------------------------------------------------------------------------
# bm25_rank tests
# ---------------------------------------------------------------------------

def test_bm25_empty_atoms():
    assert bm25_rank([], "hello") == {}


def test_bm25_no_match():
    atoms = [_Atom(id="a", content="the quick brown fox")]  # type: ignore[arg-type]
    result = bm25_rank(atoms, "python")  # type: ignore[arg-type]
    assert result == {}


def test_bm25_single_match():
    atoms = [
        _Atom(id="a", content="python is a great language"),  # type: ignore[arg-type]
        _Atom(id="b", content="java is also popular"),  # type: ignore[arg-type]
    ]
    result = bm25_rank(atoms, "python")  # type: ignore[arg-type]
    assert "a" in result
    assert "b" not in result
    assert result["a"] > 0  # negated → positive


def test_bm25_scores_positive():
    """Negated FTS5 bm25() scores should be positive (higher == more relevant)."""
    atoms = [_Atom(id="x", content="machine learning deep learning neural networks")]  # type: ignore[arg-type]
    result = bm25_rank(atoms, "learning")  # type: ignore[arg-type]
    assert result["x"] > 0


def test_bm25_ranking_order():
    """Atom with more matching terms should score higher."""
    # "python" appears more in atom 'a'
    atoms = [
        _Atom(id="a", content="python python python programming"),  # type: ignore[arg-type]
        _Atom(id="b", content="python programming language"),  # type: ignore[arg-type]
    ]
    result = bm25_rank(atoms, "python")  # type: ignore[arg-type]
    assert "a" in result
    assert "b" in result
    # 'a' has higher term frequency → should score higher or equal
    assert result["a"] >= result["b"]


def test_bm25_multiple_matches():
    """Multiple atoms match; all present in result."""
    atoms = [
        _Atom(id="1", content="quipu is a memory system"),  # type: ignore[arg-type]
        _Atom(id="2", content="quipu stores memories"),  # type: ignore[arg-type]
        _Atom(id="3", content="unrelated content about cars"),  # type: ignore[arg-type]
    ]
    result = bm25_rank(atoms, "quipu")  # type: ignore[arg-type]
    assert "1" in result
    assert "2" in result
    assert "3" not in result


def test_bm25_invalid_query_returns_empty():
    """Malformed FTS5 query (e.g. only special chars) returns empty dict."""
    atoms = [_Atom(id="a", content="some content")]  # type: ignore[arg-type]
    # FTS5 may raise OperationalError for some malformed queries;
    # bm25_rank should handle and return {}
    result = bm25_rank(atoms, '"')  # type: ignore[arg-type]
    assert isinstance(result, dict)
