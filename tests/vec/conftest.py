"""conftest.py for tests/vec — sys.path bootstrap and shared fixtures."""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

# Insert repo root so `import quipu` works without editable install.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.db import get_connection
from quipu.storage.store import pack_embedding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(index: int, dim: int | None = None) -> list[float]:
    """Return a unit vector with 1.0 at position *index*.

    dim defaults to the active model's dim (resolved at call time) so
    seeded fixtures stay consistent with pack_embedding/unpack_embedding,
    which also derive dim from active_dim() rather than a hardcoded 384.
    """
    if dim is None:
        from quipu.models.cache import active_dim
        dim = active_dim()
    v = [0.0] * dim
    v[index] = 1.0
    return v


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_conn(tmp_path):
    """Open a temporary in-memory-backed DB with atoms schema applied."""
    conn = get_connection(str(tmp_path / "test.db"))
    yield conn
    conn.close()


@pytest.fixture()
def seeded_conn(tmp_conn, semantic_model):
    """A tmp_conn pre-seeded with 5 atoms with embeddings in project 'p'."""
    for i in range(5):
        tmp_conn.execute(
            "INSERT INTO atoms (id, type, scope, content, embedding, project_id, invalidated)"
            " VALUES (?, 'diary', 'project', ?, ?, 'p', 0)",
            (f"atom{i}", f"content {i}", pack_embedding(_unit_vec(i))),
        )
    tmp_conn.commit()
    yield tmp_conn
