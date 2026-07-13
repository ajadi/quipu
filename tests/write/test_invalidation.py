"""Tests for quipu.invalidation.cosine module."""

from __future__ import annotations

import math
import os
import struct

import pytest
from tests._semantic import TEST_EMBED_DIM

from quipu.invalidation.cosine import (
    cosine,
    resolve_threshold,
    find_superseded,
    find_conflicts,
    invalidate_superseded,
)
from quipu.storage.store import Atom, pack_embedding
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_vec(dim: int, index: int) -> list[float]:
    """Return a unit vector with 1.0 at position index."""
    v = [0.0] * dim
    v[index] = 1.0
    return v


def _make_atom(
    atom_id: str,
    embedding: list[float] | None,
    project_id: str = "proj",
) -> Atom:
    """Build a minimal Atom with the given embedding."""
    emb_bytes = pack_embedding(embedding) if embedding is not None else None
    return Atom(
        id=atom_id,
        content="test content",
        embedding=emb_bytes,
        project_id=project_id,
        type="diary",
        scope="project",
        metadata={},
        refs=[],
        invalidated=False,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


def _make_bad_atom(atom_id: str, raw_blob: bytes, project_id: str = "proj") -> Atom:
    """Build an Atom with a raw (unvalidated) embedding blob — bypasses
    pack_embedding's dim check to simulate legacy/corrupt embeddings."""
    return Atom(
        id=atom_id,
        content="bad embedding atom",
        embedding=raw_blob,
        project_id=project_id,
        type="diary",
        scope="project",
        metadata={},
        refs=[],
        invalidated=False,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
    )


# ---------------------------------------------------------------------------
# cosine()
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical_unit_vectors(self):
        v = _unit_vec(TEST_EMBED_DIM, 0)
        assert cosine(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        a = _unit_vec(TEST_EMBED_DIM, 0)
        b = _unit_vec(TEST_EMBED_DIM, 1)
        assert cosine(a, b) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        v = _unit_vec(TEST_EMBED_DIM, 0)
        neg_v = [-x for x in v]
        assert cosine(v, neg_v) == pytest.approx(-1.0)

    def test_hand_crafted_pair(self):
        # a = [1, 0, 0], b = [0.6, 0.8, 0] — dot product = 0.6
        a = [1.0, 0.0, 0.0]
        b = [0.6, 0.8, 0.0]
        assert cosine(a, b) == pytest.approx(0.6)

    def test_symmetric(self):
        a = _unit_vec(TEST_EMBED_DIM, 5)
        b = _unit_vec(TEST_EMBED_DIM, 10)
        assert cosine(a, b) == pytest.approx(cosine(b, a))

    def test_returns_float(self):
        v = _unit_vec(TEST_EMBED_DIM, 0)
        result = cosine(v, v)
        assert isinstance(result, float)

    def test_dim_mismatch_raises(self):
        """Unequal-length vectors must raise, never silently truncate."""
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0]
        with pytest.raises(ValueError):
            cosine(a, b)


# ---------------------------------------------------------------------------
# resolve_threshold()
# ---------------------------------------------------------------------------

class TestResolveThreshold:
    def test_default_when_env_absent(self, monkeypatch):
        monkeypatch.delenv("QUIPU_INVALIDATION_THRESHOLD", raising=False)
        assert resolve_threshold() == pytest.approx(0.92)

    def test_reads_valid_env_value(self, monkeypatch):
        monkeypatch.setenv("QUIPU_INVALIDATION_THRESHOLD", "0.85")
        assert resolve_threshold() == pytest.approx(0.85)

    def test_bad_string_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("QUIPU_INVALIDATION_THRESHOLD", "not_a_float")
        assert resolve_threshold() == pytest.approx(0.92)

    def test_zero_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("QUIPU_INVALIDATION_THRESHOLD", "0.0")
        assert resolve_threshold() == pytest.approx(0.92)

    def test_negative_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("QUIPU_INVALIDATION_THRESHOLD", "-0.5")
        assert resolve_threshold() == pytest.approx(0.92)

    def test_value_above_one_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("QUIPU_INVALIDATION_THRESHOLD", "1.5")
        assert resolve_threshold() == pytest.approx(0.92)

    def test_exactly_one_is_valid(self, monkeypatch):
        monkeypatch.setenv("QUIPU_INVALIDATION_THRESHOLD", "1.0")
        assert resolve_threshold() == pytest.approx(1.0)

    def test_bad_value_logs_warning(self, monkeypatch, caplog):
        import logging
        monkeypatch.setenv("QUIPU_INVALIDATION_THRESHOLD", "garbage")
        with caplog.at_level(logging.WARNING, logger="quipu.invalidation.cosine"):
            resolve_threshold()
        assert caplog.records, "Expected a warning log for bad threshold value"


# ---------------------------------------------------------------------------
# find_superseded()
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestFindSuperseded:
    def test_no_existing_atoms(self):
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        assert find_superseded(new_vec, []) == []

    def test_identical_vector_above_threshold(self):
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", v)
        result = find_superseded(v, [atom], threshold=0.92)
        assert "a1" in result

    def test_orthogonal_vector_below_threshold(self):
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", _unit_vec(TEST_EMBED_DIM, 1))
        result = find_superseded(new_vec, [atom], threshold=0.92)
        assert result == []

    def test_threshold_boundary_exactly_equal_included(self):
        """Similarity exactly at threshold should be included (>= not >)."""
        # Build two vectors with known dot product = 0.92
        # v1 = [cos(theta), sin(theta), 0, ...] with cos(theta) = 0.92
        theta = math.acos(0.92)
        v1 = [0.0] * TEST_EMBED_DIM
        v1[0] = math.cos(theta)  # 0.92
        v1[1] = math.sin(theta)

        # new_vec = [1, 0, 0, ...] (unit vector along dim 0)
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)

        # cosine(new_vec, v1) = 0.92 exactly
        assert cosine(new_vec, v1) == pytest.approx(0.92, abs=1e-6)

        atom = _make_atom("boundary", v1)
        result = find_superseded(new_vec, [atom], threshold=0.92)
        assert "boundary" in result, "Atom at exactly threshold should be included"

    def test_threshold_boundary_just_below_excluded(self):
        """Similarity just below threshold should be excluded."""
        # Similarity = 0.91 < 0.92
        theta = math.acos(0.91)
        v1 = [0.0] * TEST_EMBED_DIM
        v1[0] = math.cos(theta)  # 0.91
        v1[1] = math.sin(theta)

        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("below", v1)
        result = find_superseded(new_vec, [atom], threshold=0.92)
        assert "below" not in result

    def test_skips_atoms_with_none_embedding(self):
        atom_no_emb = _make_atom("no_emb", None)
        atom_with_emb = _make_atom("has_emb", _unit_vec(TEST_EMBED_DIM, 0))
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        result = find_superseded(new_vec, [atom_no_emb, atom_with_emb], threshold=0.92)
        assert "no_emb" not in result
        assert "has_emb" in result

    def test_multiple_matches(self):
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        atoms = [_make_atom(f"a{i}", new_vec) for i in range(3)]
        result = find_superseded(new_vec, atoms, threshold=0.92)
        assert set(result) == {"a0", "a1", "a2"}

    def test_uses_default_threshold_when_none(self, monkeypatch):
        monkeypatch.delenv("QUIPU_INVALIDATION_THRESHOLD", raising=False)
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", new_vec)  # similarity = 1.0 >= 0.92
        result = find_superseded(new_vec, [atom])  # threshold=None -> default
        assert "a1" in result

    def test_mismatched_dim_embedding_skipped_no_crash(self):
        """Atom with a shorter/longer embedding blob (dim mismatch) is
        skipped, not raised — legacy atom or model-switch scenario."""
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        bad_blob = struct.pack(f"<{TEST_EMBED_DIM - 1}f", *([1.0] * (TEST_EMBED_DIM - 1)))
        bad_atom = _make_bad_atom("bad", bad_blob)
        good_atom = _make_atom("good", new_vec)
        result = find_superseded(new_vec, [bad_atom, good_atom], threshold=0.92)
        assert "bad" not in result
        assert "good" in result

    def test_corrupt_blob_skipped_no_crash(self):
        """Atom with an arbitrary corrupt blob (wrong byte length) is
        skipped, not raised."""
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        corrupt_blob = b"\x00\x01\x02"  # arbitrary, not dim*4 bytes
        bad_atom = _make_bad_atom("corrupt", corrupt_blob)
        good_atom = _make_atom("good", new_vec)
        result = find_superseded(new_vec, [bad_atom, good_atom], threshold=0.92)
        assert "corrupt" not in result
        assert "good" in result

    def test_only_bad_embeddings_returns_empty_no_crash(self):
        """All atoms have bad embeddings — scan completes with empty result."""
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        bad_blob = struct.pack(f"<{TEST_EMBED_DIM - 1}f", *([1.0] * (TEST_EMBED_DIM - 1)))
        bad_atom = _make_bad_atom("bad", bad_blob)
        result = find_superseded(new_vec, [bad_atom], threshold=0.92)
        assert result == []


# ---------------------------------------------------------------------------
# find_conflicts()
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestFindConflicts:
    def test_above_threshold_returns_id_similarity_snippet(self):
        """High-similarity atom returns conflict with id, similarity, snippet."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", v)
        atom.content = "short content"
        result = find_conflicts(v, [atom], threshold=0.92)
        assert len(result) == 1
        entry = result[0]
        assert entry["id"] == "a1"
        assert "similarity" in entry
        assert "snippet" in entry

    def test_similarity_rounded_to_4dp(self):
        """similarity field is rounded to 4 decimal places."""
        # Build vector with known sub-unity dot product
        theta = math.acos(0.9750)
        v1 = [0.0] * TEST_EMBED_DIM
        v1[0] = math.cos(theta)
        v1[1] = math.sin(theta)
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", v1)
        result = find_conflicts(new_vec, [atom], threshold=0.92)
        assert len(result) == 1
        sim = result[0]["similarity"]
        # Must be a float rounded to at most 4 decimal places
        assert sim == round(sim, 4)
        assert isinstance(sim, float)

    def test_snippet_truncated_at_160_with_ellipsis(self):
        """Content longer than 160 chars → snippet is content[:160] + '…'."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", v)
        long_content = "x" * 200
        atom.content = long_content
        result = find_conflicts(v, [atom], threshold=0.92)
        assert len(result) == 1
        snippet = result[0]["snippet"]
        assert snippet == "x" * 160 + "…"

    def test_snippet_no_ellipsis_for_short_content(self):
        """Content <= 160 chars → snippet has no ellipsis appended."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", v)
        short_content = "hello world"
        atom.content = short_content
        result = find_conflicts(v, [atom], threshold=0.92)
        assert len(result) == 1
        snippet = result[0]["snippet"]
        assert snippet == short_content
        assert not snippet.endswith("…")

    def test_snippet_exactly_160_no_ellipsis(self):
        """Content exactly 160 chars → no ellipsis (not longer than 160)."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", v)
        atom.content = "y" * 160
        result = find_conflicts(v, [atom], threshold=0.92)
        assert len(result) == 1
        snippet = result[0]["snippet"]
        assert snippet == "y" * 160
        assert not snippet.endswith("…")

    def test_threshold_inclusive_boundary(self):
        """Atom at exactly threshold similarity is included (>= inclusive)."""
        theta = math.acos(0.92)
        v1 = [0.0] * TEST_EMBED_DIM
        v1[0] = math.cos(theta)  # 0.92
        v1[1] = math.sin(theta)
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        assert cosine(new_vec, v1) == pytest.approx(0.92, abs=1e-6)
        atom = _make_atom("boundary", v1)
        result = find_conflicts(new_vec, [atom], threshold=0.92)
        assert any(e["id"] == "boundary" for e in result), "Atom at threshold should be included"

    def test_skips_none_embedding(self):
        """Atom with embedding=None is skipped (no crash)."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom_none = _make_atom("no_emb", None)
        atom_with = _make_atom("has_emb", v)
        result = find_conflicts(v, [atom_none, atom_with], threshold=0.92)
        ids = [e["id"] for e in result]
        assert "no_emb" not in ids
        assert "has_emb" in ids

    def test_skips_exclude_id(self):
        """Atom matching exclude_id is skipped (self-conflict prevention)."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("self", v)
        result = find_conflicts(v, [atom], threshold=0.92, exclude_id="self")
        assert result == []

    def test_orthogonal_vectors_returns_empty(self):
        """Orthogonal vectors produce no conflicts."""
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", _unit_vec(TEST_EMBED_DIM, 1))
        result = find_conflicts(new_vec, [atom], threshold=0.92)
        assert result == []

    def test_empty_existing_returns_empty(self):
        """No existing atoms → empty conflict list."""
        new_vec = _unit_vec(TEST_EMBED_DIM, 0)
        result = find_conflicts(new_vec, [], threshold=0.92)
        assert result == []

    def test_uses_default_threshold_when_none(self, monkeypatch):
        """threshold=None uses resolve_threshold() (default 0.92)."""
        monkeypatch.delenv("QUIPU_INVALIDATION_THRESHOLD", raising=False)
        v = _unit_vec(TEST_EMBED_DIM, 0)
        atom = _make_atom("a1", v)  # similarity=1.0 >= 0.92
        result = find_conflicts(v, [atom])
        assert any(e["id"] == "a1" for e in result)

    def test_mismatched_dim_embedding_skipped_no_crash(self):
        """Atom with a shorter/longer embedding blob (dim mismatch) is
        skipped, not raised — legacy atom or model-switch scenario."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        bad_blob = struct.pack(f"<{TEST_EMBED_DIM - 1}f", *([1.0] * (TEST_EMBED_DIM - 1)))
        bad_atom = _make_bad_atom("bad", bad_blob)
        good_atom = _make_atom("good", v)
        result = find_conflicts(v, [bad_atom, good_atom], threshold=0.92)
        ids = [e["id"] for e in result]
        assert "bad" not in ids
        assert "good" in ids

    def test_corrupt_blob_skipped_no_crash(self):
        """Atom with an arbitrary corrupt blob (wrong byte length) is
        skipped, not raised."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        corrupt_blob = b"\x00\x01\x02"
        bad_atom = _make_bad_atom("corrupt", corrupt_blob)
        good_atom = _make_atom("good", v)
        result = find_conflicts(v, [bad_atom, good_atom], threshold=0.92)
        ids = [e["id"] for e in result]
        assert "corrupt" not in ids
        assert "good" in ids

    def test_only_bad_embeddings_returns_empty_no_crash(self):
        """All atoms have bad embeddings — scan completes with empty result."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        bad_blob = struct.pack(f"<{TEST_EMBED_DIM - 1}f", *([1.0] * (TEST_EMBED_DIM - 1)))
        bad_atom = _make_bad_atom("bad", bad_blob)
        result = find_conflicts(v, [bad_atom], threshold=0.92)
        assert result == []


# ---------------------------------------------------------------------------
# invalidate_superseded()
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestInvalidateSuperseded:
    def test_marks_older_matching_atom(self, tmp_store):
        """Existing atom with high similarity to new_vec gets invalidated."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        emb_bytes = pack_embedding(v)

        # Insert existing atom
        existing = tmp_store.insert(
            content="existing content",
            embedding=emb_bytes,
            project_id="proj1",
        )
        assert not existing.invalidated

        # Run invalidation with the same vector
        ids = invalidate_superseded(tmp_store, v, "proj1", threshold=0.92)
        assert existing.id in ids

        # Verify it's actually marked invalidated in the store
        updated = tmp_store.get(existing.id)
        assert updated.invalidated

    def test_does_not_affect_different_direction(self, tmp_store):
        """Atom with low similarity is not invalidated."""
        v_new = _unit_vec(TEST_EMBED_DIM, 0)
        v_old = _unit_vec(TEST_EMBED_DIM, 1)  # orthogonal
        emb_bytes = pack_embedding(v_old)

        existing = tmp_store.insert(
            content="unrelated content",
            embedding=emb_bytes,
            project_id="proj1",
        )

        ids = invalidate_superseded(tmp_store, v_new, "proj1", threshold=0.92)
        assert existing.id not in ids
        assert not tmp_store.get(existing.id).invalidated

    def test_returns_list_of_invalidated_ids(self, tmp_store):
        v = _unit_vec(TEST_EMBED_DIM, 5)
        emb_bytes = pack_embedding(v)
        a1 = tmp_store.insert(content="one", embedding=emb_bytes, project_id="p")
        a2 = tmp_store.insert(content="two", embedding=emb_bytes, project_id="p")

        ids = invalidate_superseded(tmp_store, v, "p", threshold=0.92)
        assert set(ids) == {a1.id, a2.id}

    def test_empty_project_returns_empty(self, tmp_store):
        v = _unit_vec(TEST_EMBED_DIM, 0)
        ids = invalidate_superseded(tmp_store, v, "empty_project", threshold=0.92)
        assert ids == []

    def test_self_exclusion_via_exclude_id(self, tmp_store):
        """An atom should not self-invalidate when exclude_id equals its own id."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        emb_bytes = pack_embedding(v)

        atom = tmp_store.insert(
            content="the atom itself",
            embedding=emb_bytes,
            project_id="proj_self",
        )

        # Call with a vector identical to the atom's embedding and exclude its own id
        ids = invalidate_superseded(
            tmp_store, v, "proj_self", threshold=0.92, exclude_id=atom.id
        )
        assert atom.id not in ids, "Atom must not self-invalidate when exclude_id is set"
        assert not tmp_store.get(atom.id).invalidated

    def test_bad_embedding_atom_does_not_crash_scan(self, tmp_store):
        """A pre-existing atom with a bad/legacy embedding blob must not
        crash invalidate_superseded; other atoms are still processed."""
        v = _unit_vec(TEST_EMBED_DIM, 0)
        bad_blob = struct.pack(f"<{TEST_EMBED_DIM - 1}f", *([1.0] * (TEST_EMBED_DIM - 1)))

        bad_atom = tmp_store.insert(
            content="legacy/corrupt embedding atom",
            embedding=bad_blob,
            project_id="proj_mixed",
        )
        good_atom = tmp_store.insert(
            content="good embedding atom",
            embedding=pack_embedding(v),
            project_id="proj_mixed",
        )

        ids = invalidate_superseded(tmp_store, v, "proj_mixed", threshold=0.92)

        assert bad_atom.id not in ids
        assert good_atom.id in ids
        assert not tmp_store.get(bad_atom.id).invalidated
        assert tmp_store.get(good_atom.id).invalidated


# ---------------------------------------------------------------------------
# list_by_project excludes invalidated (retrieval-flag correctness)
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("semantic_model")
class TestListByProjectExcludesInvalidated:
    def test_invalidated_atom_excluded_by_default(self, tmp_store):
        v = _unit_vec(TEST_EMBED_DIM, 0)
        emb_bytes = pack_embedding(v)

        atom = tmp_store.insert(
            content="content",
            embedding=emb_bytes,
            project_id="proj",
        )
        tmp_store.update_invalidated(atom.id, True)

        active = tmp_store.list_by_project("proj", include_invalidated=False)
        ids = [a.id for a in active]
        assert atom.id not in ids

    def test_invalidated_included_when_flag_set(self, tmp_store):
        v = _unit_vec(TEST_EMBED_DIM, 0)
        emb_bytes = pack_embedding(v)

        atom = tmp_store.insert(
            content="content",
            embedding=emb_bytes,
            project_id="proj",
        )
        tmp_store.update_invalidated(atom.id, True)

        all_atoms = tmp_store.list_by_project("proj", include_invalidated=True)
        ids = [a.id for a in all_atoms]
        assert atom.id in ids
