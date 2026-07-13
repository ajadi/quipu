"""conftest.py for tests/drain — shared fixtures for drain/secrets tests.

Mirrors tests/write/conftest.py: sys.path insert, fake embedding engine,
reset autouse, and tmp_store. The real ONNX model is NOT in this env, so
any test that exercises write()/drain() MUST use the fake_engine fixture.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tests._semantic import TEST_EMBED_DIM

# Insert repo root so `import quipu` works without editable install.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.embeddings.engine import _reset, set_engine, _Engine
from quipu.storage import store as open_store


# ---------------------------------------------------------------------------
# Fake engine helpers (no ONNX, no model files)
# ---------------------------------------------------------------------------

class _FakeTokenizerEncoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask


class _FakeTokenizer:
    """Minimal tokenizer stub."""

    def __init__(self, seq_len: int = 8) -> None:
        self._seq_len = seq_len

    def encode_batch(self, texts):
        return [
            _FakeTokenizerEncoding(
                ids=[1] * self._seq_len,
                mask=[1] * self._seq_len,
            )
            for _ in texts
        ]


class _N:
    """Name holder for session inputs/outputs."""
    def __init__(self, name: str) -> None:
        self.name = name
        self.type = "tensor(int64)"


class _FakeSession:
    """Returns a deterministic rank-2 output (already pooled)."""

    def __init__(self, value: float = 1.0, seq_len: int = 8) -> None:
        self._value = value
        self._seq_len = seq_len

    def get_inputs(self):
        return [_N("input_ids"), _N("attention_mask")]

    def get_outputs(self):
        return [_N("sentence_embedding")]

    def run(self, output_names, feeds):
        import numpy as np
        n = feeds["input_ids"].shape[0]
        arr = np.full((n, TEST_EMBED_DIM), self._value, dtype=np.float32)
        return [arr]


def _make_fake_engine(value: float = 1.0) -> _Engine:
    """Build a fake engine returning uniform vectors (will be L2-normalized)."""
    return _Engine(
        session=_FakeSession(value=value),
        tokenizer=_FakeTokenizer(),
    )


# ---------------------------------------------------------------------------
# Autouse fixture: reset embedding singleton after each test
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_embedding_engine():
    """Reset embedding singleton in teardown to avoid test contamination."""
    yield
    _reset()


# ---------------------------------------------------------------------------
# Fake engine fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_engine(semantic_model):
    """Inject a fake engine that returns L2-normalized uniform vectors."""
    engine = _make_fake_engine(value=1.0)
    set_engine(engine)
    return engine


# ---------------------------------------------------------------------------
# Store fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_store(tmp_path):
    """Open a temp-path SQLite store; close after test."""
    db_path = str(tmp_path / "t.db")
    s = open_store(db_path)
    yield s
    s.close()
