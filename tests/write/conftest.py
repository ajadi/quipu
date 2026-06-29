"""conftest.py for tests/write — ensures quipu package is importable and
provides shared fixtures for the write track tests.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Insert repo root so `import quipu` works without editable install.
_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.embeddings.engine import _reset, set_engine, EMBED_DIM, _Engine
from quipu.storage import store as open_store


def get_flush_module():
    """Return the quipu.write.flush MODULE (not the flush function).

    quipu.write.__init__ does ``from .flush import flush``, which causes
    ``import quipu.write.flush`` to resolve to the function through the
    package namespace. Use sys.modules to get the real module object so
    tests can monkeypatch _http_post_json on it.
    """
    import quipu.write  # ensure package is loaded
    return sys.modules["quipu.write.flush"]


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
        arr = np.full((n, EMBED_DIM), self._value, dtype=np.float32)
        return [arr]


class _VecSession:
    """Returns a specific pre-set vector for all inputs."""

    def __init__(self, vec: list[float]) -> None:
        self._vec = vec

    def get_inputs(self):
        return [_N("input_ids"), _N("attention_mask")]

    def get_outputs(self):
        return [_N("sentence_embedding")]

    def run(self, output_names, feeds):
        import numpy as np
        n = feeds["input_ids"].shape[0]
        arr = np.array([self._vec] * n, dtype=np.float32)
        return [arr]


def _unit_vec(dim: int, index: int) -> list[float]:
    """Return a unit vector with 1.0 at position index (L2-norm=1)."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _make_fake_engine(value: float = 1.0) -> _Engine:
    """Build a fake engine returning uniform vectors (will be L2-normalized)."""
    return _Engine(
        session=_FakeSession(value=value),
        tokenizer=_FakeTokenizer(),
    )


def _make_vec_engine(vec: list[float]) -> _Engine:
    """Build a fake engine returning a specific vector (should be pre-normalized)."""
    return _Engine(
        session=_VecSession(vec),
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
def fake_engine():
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
