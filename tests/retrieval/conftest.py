"""conftest.py for tests/retrieval — sys.path bootstrap and shared fixtures."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# Insert repo root so `import quipu` works without editable install.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.embeddings.engine import _reset
from quipu.storage import store as open_store
from quipu.storage.store import pack_embedding


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, index: int) -> list[float]:
    """Return an L2-normalized unit vector with 1.0 at position index."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _make_fake_embed(vec: list[float]):
    """Return a fake embed function that always returns *vec*."""
    def _embed(text: str) -> list[float]:
        return vec
    return _embed


# ---------------------------------------------------------------------------
# Autouse fixture: reset embedding singleton after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_embedding_engine():
    """Prevent embedding singleton from leaking between tests."""
    yield
    _reset()


# ---------------------------------------------------------------------------
# Store fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path):
    """Open a temp SQLite store; close after test."""
    s = open_store(str(tmp_path / "t.db"))
    yield s
    s.close()
