"""Unit tests for quipu/behavior/prime.py — degradation matrix."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tests._semantic import TEST_EMBED_DIM

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.behavior.prime import prime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def fake_engine(semantic_model):
    """Inject fake embedding engine (no ONNX)."""
    from quipu.embeddings.engine import _reset, set_engine, _Engine

    class _FakeTok:
        class _Enc:
            def __init__(self):
                self.ids = [1] * 8
                self.attention_mask = [1] * 8

        def encode_batch(self, texts):
            return [self._Enc() for _ in texts]

    class _FakeSess:
        def get_inputs(self):
            class _N:
                name = "input_ids"
                type = "tensor(int64)"
            class _N2:
                name = "attention_mask"
                type = "tensor(int64)"
            return [_N(), _N2()]

        def get_outputs(self):
            class _N:
                name = "sentence_embedding"
            return [_N()]

        def run(self, output_names, feeds):
            import numpy as np
            n = feeds["input_ids"].shape[0]
            return [np.ones((n, TEST_EMBED_DIM), dtype=np.float32)]

    engine = _Engine(session=_FakeSess(), tokenizer=_FakeTok())
    set_engine(engine)
    yield engine
    _reset()


@pytest.fixture()
def tmp_store(tmp_path):
    from quipu.storage import store as open_store
    db_path = str(tmp_path / "t.db")
    s = open_store(db_path)
    yield s
    s.close()


# ---------------------------------------------------------------------------
# Degradation matrix
# ---------------------------------------------------------------------------


class TestPrimeDegradation:
    def test_no_project_id_returns_primed_false(self, tmp_store):
        result = prime(tmp_store, project_id=None)
        assert result == {"primed": False, "results": [], "note": "no project_id"}

    def test_empty_store_returns_primed_true_no_results(self, tmp_store, fake_engine):
        result = prime(tmp_store, project_id="proj", topic="anything")
        assert result["primed"] is True
        assert result["results"] == []
        assert result["note"] == "no memory yet"

    def test_write_then_prime_finds_atom(self, tmp_store, fake_engine):
        from quipu.write import write
        atom_id = write("quipu test marker alpha", {}, "proj", store=tmp_store)

        result = prime(tmp_store, project_id="proj", topic="quipu test marker")
        assert result["primed"] is True
        result_ids = [r["id"] for r in result["results"]]
        assert atom_id in result_ids

    def test_search_raises_returns_recall_unavailable(self, tmp_store):
        from unittest.mock import patch

        def _raise(*args, **kwargs):
            raise RuntimeError("simulated failure")

        # Target the module attribute directly so the patch works regardless of
        # where `from quipu.retrieval import search` binds inside prime().
        with patch("quipu.retrieval.search", side_effect=_raise):
            from quipu.behavior.prime import prime as _prime
            result = _prime(tmp_store, project_id="proj", topic="x")
        assert result == {"primed": True, "results": [], "note": "recall unavailable"}

    def test_default_topic_no_raise(self, tmp_store, fake_engine):
        # topic=None uses default seed, should not raise even on empty store
        result = prime(tmp_store, project_id="proj", topic=None)
        assert "primed" in result
        assert "results" in result
        assert "note" in result

    def test_result_shape_mirrors_search(self, tmp_store, fake_engine):
        from quipu.write import write
        write("architecture decision: use sqlite", {}, "proj", store=tmp_store)

        result = prime(tmp_store, project_id="proj", top_k=5)
        if result["results"]:
            r = result["results"][0]
            for field in ("id", "content", "score", "tier", "type", "scope",
                          "invalidated", "metadata"):
                assert field in r, f"missing field: {field}"
            assert "embedding" not in r

    def test_top_k_limits_results(self, tmp_store, fake_engine):
        from quipu.write import write
        for i in range(5):
            write(f"record number {i} about patterns and decisions", {}, "proj", store=tmp_store)

        result = prime(tmp_store, project_id="proj", top_k=2)
        assert result["primed"] is True
        assert len(result["results"]) <= 2

    def test_never_raises_on_bad_store(self, monkeypatch):
        """prime() must NEVER raise, even with a completely broken store."""
        class _BrokenStore:
            pass

        result = prime(_BrokenStore(), project_id="proj", topic="anything")
        assert result["primed"] is True
        assert result["results"] == []
        assert result["note"] == "recall unavailable"
