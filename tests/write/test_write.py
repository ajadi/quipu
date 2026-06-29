"""Tests for quipu.write.pipeline.write()."""

from __future__ import annotations

import math

import pytest

from quipu.embeddings.engine import set_engine, EMBED_DIM, _Engine
from quipu.storage.store import pack_embedding, unpack_embedding
from quipu.write.pipeline import write
from tests.write.conftest import get_flush_module


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, index: int) -> list[float]:
    v = [0.0] * dim
    v[index] = 1.0
    return v


class _N:
    def __init__(self, name: str) -> None:
        self.name = name
        self.type = "tensor(int64)"


class _VecSession:
    """Returns a specific pre-set L2-normalized vector for all inputs."""

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


class _FakeTokenizer:
    def __init__(self, seq_len: int = 8) -> None:
        self._seq_len = seq_len

    def encode_batch(self, texts):
        class _Enc:
            def __init__(self, sl):
                self.ids = [1] * sl
                self.attention_mask = [1] * sl
        return [_Enc(self._seq_len) for _ in texts]


def _inject_vec_engine(vec: list[float]) -> _Engine:
    """Inject a fake engine that always returns the given vector."""
    engine = _Engine(
        session=_VecSession(vec),
        tokenizer=_FakeTokenizer(),
    )
    set_engine(engine)
    return engine


# ---------------------------------------------------------------------------
# Tests: write() basic correctness
# ---------------------------------------------------------------------------

class TestWriteBasic:
    def test_returns_string_id(self, fake_engine, tmp_store):
        atom_id = write("hello world", store=tmp_store)
        assert isinstance(atom_id, str)
        assert len(atom_id) > 0

    def test_stored_record_retrievable(self, fake_engine, tmp_store):
        atom_id = write("hello world", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom is not None
        assert atom.content == "hello world"

    def test_embedding_stored(self, fake_engine, tmp_store):
        atom_id = write("hello world", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.embedding is not None
        vec = unpack_embedding(atom.embedding)
        assert len(vec) == EMBED_DIM

    def test_embedding_is_normalized(self, fake_engine, tmp_store):
        atom_id = write("hello world", store=tmp_store)
        atom = tmp_store.get(atom_id)
        vec = unpack_embedding(atom.embedding)
        norm = math.sqrt(sum(x * x for x in vec))
        assert abs(norm - 1.0) < 1e-4

    def test_metadata_contains_entities(self, fake_engine, tmp_store):
        atom_id = write("Alice and Bob went to London", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert "entities" in atom.metadata
        assert isinstance(atom.metadata["entities"], list)

    def test_metadata_contains_keywords(self, fake_engine, tmp_store):
        atom_id = write("machine learning algorithms", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert "keywords" in atom.metadata
        assert isinstance(atom.metadata["keywords"], list)

    def test_enriched_flag_false_at_write(self, fake_engine, tmp_store):
        atom_id = write("some content", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.metadata.get("enriched") is False

    def test_project_id_stored(self, fake_engine, tmp_store):
        atom_id = write("content", project_id="myproject", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.project_id == "myproject"

    def test_default_type_is_diary(self, fake_engine, tmp_store):
        atom_id = write("content", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.type == "diary"

    def test_custom_type(self, fake_engine, tmp_store):
        atom_id = write("content", type="decision", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.type == "decision"

    def test_caller_metadata_merged(self, fake_engine, tmp_store):
        atom_id = write("content", metadata={"source": "test"}, store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.metadata.get("source") == "test"

    def test_caller_cannot_override_enriched_flag(self, fake_engine, tmp_store):
        """caller metadata enriched=True must be overridden to False."""
        atom_id = write("content", metadata={"enriched": True}, store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.metadata.get("enriched") is False


# ---------------------------------------------------------------------------
# Tests: auto-invalidation
# ---------------------------------------------------------------------------

class TestWriteAutoInvalidation:
    def test_near_duplicate_no_longer_auto_invalidated(self, tmp_store):
        """Writing a near-duplicate to same project no longer silently invalidates the older one.

        write() detects conflicts at the MCP layer; write() itself never mutates existing atoms.
        """
        v = _unit_vec(EMBED_DIM, 0)

        # First write: insert atom with unit vec along dim 0
        _inject_vec_engine(v)
        old_id = write("first content", project_id="proj", store=tmp_store)

        # Second write: same direction -> old atom must remain ACTIVE (not auto-invalidated)
        _inject_vec_engine(v)
        new_id = write("second content", project_id="proj", store=tmp_store)

        old_atom = tmp_store.get(old_id)
        assert not old_atom.invalidated, "Older atom must NOT be auto-invalidated by write()"

        new_atom = tmp_store.get(new_id)
        assert not new_atom.invalidated, "New atom must not be self-invalidated"

    def test_no_invalidation_without_project_id(self, tmp_store):
        """Without a project_id, no invalidation scan occurs."""
        v = _unit_vec(EMBED_DIM, 0)

        _inject_vec_engine(v)
        old_id = write("first content", store=tmp_store)  # no project_id

        _inject_vec_engine(v)
        write("second content", store=tmp_store)  # no project_id

        old_atom = tmp_store.get(old_id)
        assert not old_atom.invalidated

    def test_no_cross_project_invalidation(self, tmp_store):
        """Atoms in different projects are not invalidated by each other."""
        v = _unit_vec(EMBED_DIM, 0)

        _inject_vec_engine(v)
        atom_a = write("content A", project_id="proj_a", store=tmp_store)

        _inject_vec_engine(v)
        write("content B", project_id="proj_b", store=tmp_store)

        # proj_a atom should be untouched
        assert not tmp_store.get(atom_a).invalidated

    def test_unrelated_atom_not_invalidated(self, tmp_store):
        """Orthogonal vectors in same project do not trigger invalidation."""
        v_old = _unit_vec(EMBED_DIM, 1)
        v_new = _unit_vec(EMBED_DIM, 2)

        _inject_vec_engine(v_old)
        old_id = write("old content", project_id="proj", store=tmp_store)

        _inject_vec_engine(v_new)
        write("new content", project_id="proj", store=tmp_store)

        assert not tmp_store.get(old_id).invalidated


# ---------------------------------------------------------------------------
# TASK-023 — session_id passthrough
# ---------------------------------------------------------------------------

class TestWriteSessionId:
    def test_write_with_session_id_populates_column(self, fake_engine, tmp_store):
        atom_id = write("content with session", session_id="sess-abc", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.session_id == "sess-abc"

    def test_write_without_session_id_leaves_null(self, fake_engine, tmp_store):
        atom_id = write("content without session", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.session_id is None


# ---------------------------------------------------------------------------
# TASK-024 — auto-tagging on write
# ---------------------------------------------------------------------------

class TestWriteTags:
    def test_write_populates_tags_from_entities_and_keywords(self, fake_engine, tmp_store):
        atom_id = write("Alice and Bob went to machine learning conference in London",
                         store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.tags is not None
        assert len(atom.tags) > 0
        # Tags should include top-5 derived from entities (lowered) + keywords
        tag_set = set(atom.tags)
        assert len(atom.tags) <= 5

    def test_write_tags_capped_at_five(self, fake_engine, tmp_store):
        atom_id = write(
            "Alice Bob Charlie Delta Echo Forest Green Hotel India Juliet "
            "Kilo Lima Mike November Oscar Papa Quebec Romeo Sierra Tango "
            "Uniform Victor Whiskey Xray Yankee Zulu",
            store=tmp_store,
        )
        atom = tmp_store.get(atom_id)
        assert atom.tags is not None
        assert len(atom.tags) == 5

    def test_write_tags_none_for_short_content(self, fake_engine, tmp_store):
        atom_id = write("it is a", store=tmp_store)
        atom = tmp_store.get(atom_id)
        # "it", "is", "a" are all stopwords or <3 chars → no keywords,
        # and no capitalized entities → tags should be None
        assert atom.tags is None or atom.tags == []

    def test_write_tags_preserved_on_get(self, fake_engine, tmp_store):
        atom_id = write("Database Indexing Performance Tuning", store=tmp_store)
        atom = tmp_store.get(atom_id)
        assert atom.tags is not None
        fetched = tmp_store.get(atom_id)
        assert fetched.tags == atom.tags


# ---------------------------------------------------------------------------
# Tests: no Haiku/network during write
# ---------------------------------------------------------------------------

class TestWriteNoNetwork:
    def test_no_http_call_during_write(self, fake_engine, tmp_store, monkeypatch):
        """write() must never call _http_post_json (Haiku is flush-only)."""
        flush_mod = get_flush_module()
        calls = []

        def _fake_http(url, headers, payload):
            calls.append((url, headers, payload))
            return {}

        monkeypatch.setattr(flush_mod, "_http_post_json", _fake_http)

        write("some content", store=tmp_store)
        assert calls == [], f"Expected zero network calls during write, got {calls}"

    def test_flush_not_called_by_write(self, fake_engine, tmp_store, monkeypatch):
        """write() must not invoke flush()."""
        import quipu.write.pipeline as pipeline_mod
        calls = []

        def _fake_flush(*args, **kwargs):
            calls.append((args, kwargs))

        monkeypatch.setattr(pipeline_mod, "flush", _fake_flush, raising=False)

        write("some content", store=tmp_store)
        assert calls == []
