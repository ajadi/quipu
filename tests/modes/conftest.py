"""conftest.py for tests/modes — shared fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.embeddings.engine import _reset, set_engine, EMBED_DIM, _Engine
from quipu.storage import store as open_store


# ---------------------------------------------------------------------------
# Fake embedding engine (no ONNX) — mirrors tests/mcp/conftest.py
# ---------------------------------------------------------------------------


class _FakeTokenizerEncoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask


class _FakeTokenizer:
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
    def __init__(self, name: str) -> None:
        self.name = name
        self.type = "tensor(int64)"


class _FakeSession:
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


def _make_fake_engine(value: float = 1.0) -> _Engine:
    return _Engine(
        session=_FakeSession(value=value),
        tokenizer=_FakeTokenizer(),
    )


# ---------------------------------------------------------------------------
# Autouse: reset embedding singleton after each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def reset_embedding_engine():
    yield
    _reset()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_engine():
    """Inject a fake embedding engine (no ONNX)."""
    engine = _make_fake_engine(value=1.0)
    set_engine(engine)
    return engine


@pytest.fixture()
def tmp_store(tmp_path):
    """Open a temp-path SQLite store; close after test."""
    db_path = str(tmp_path / "t.db")
    s = open_store(db_path)
    yield s
    s.close()


@pytest.fixture()
def tmp_store2(tmp_path):
    """Second temp store on a different DB file (for global_store tests)."""
    db_path = str(tmp_path / "g.db")
    s = open_store(db_path)
    yield s
    s.close()


@pytest.fixture()
def project_id() -> str:
    return "test_project"
