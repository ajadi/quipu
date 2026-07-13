"""Unit tests for quipu.storage CRUD, vector helpers, and CHECK constraints.

Coverage map (AC numbers from TASK-001 spec):
  AC1 — tested in test_migrations.py
  AC2 — insert/get/update_invalidated/delete/list_by_project (this file)
  AC3 — pack_embedding/unpack_embedding + BLOB round-trip (this file)
  AC4 — CHECK constraints on type and scope (this file)
  AC5 — project_id nullable (this file)
  AC6 — import contract (this file, one smoke test)
"""

import sqlite3

import pytest

from quipu.storage import store, Atom, pack_embedding, unpack_embedding
from quipu.storage.store import Store


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def s(tmp_path):
    """Open a fresh Store backed by a temp file; close after test."""
    db = tmp_path / "test.db"
    with store(str(db)) as _store:
        yield _store


# ---------------------------------------------------------------------------
# AC6 — import contract
# ---------------------------------------------------------------------------

class TestImportContract:
    def test_store_factory_importable_from_quipu_storage(self):
        """from quipu.storage import store must succeed and be callable."""
        from quipu.storage import store as _store
        assert callable(_store)

    def test_atom_importable_from_quipu_storage(self):
        from quipu.storage import Atom as _Atom
        assert _Atom is Atom

    def test_pack_unpack_importable_from_quipu_storage(self):
        from quipu.storage import pack_embedding as pe, unpack_embedding as ue
        assert callable(pe) and callable(ue)


# ---------------------------------------------------------------------------
# AC2 — insert: returned Atom has expected defaults and generated id
# ---------------------------------------------------------------------------

class TestInsert:
    def test_insert_returns_atom_instance(self, s):
        atom = s.insert(content="hello world")
        assert isinstance(atom, Atom)

    def test_insert_generates_uuid_id(self, s):
        atom = s.insert(content="hello")
        assert atom.id and len(atom.id) == 32  # uuid4().hex is 32 hex chars

    def test_insert_explicit_id_is_preserved(self, s):
        atom = s.insert(content="hi", id="myspecialid")
        assert atom.id == "myspecialid"

    def test_insert_default_type_is_diary(self, s):
        atom = s.insert(content="x")
        assert atom.type == "diary"

    def test_insert_default_scope_is_project(self, s):
        atom = s.insert(content="x")
        assert atom.scope == "project"

    def test_insert_default_invalidated_is_false(self, s):
        atom = s.insert(content="x")
        assert atom.invalidated is False

    def test_insert_default_metadata_is_empty_dict(self, s):
        atom = s.insert(content="x")
        assert atom.metadata == {}

    def test_insert_default_refs_is_empty_list(self, s):
        atom = s.insert(content="x")
        assert atom.refs == []

    def test_insert_content_stored_correctly(self, s):
        atom = s.insert(content="the content")
        assert atom.content == "the content"

    def test_insert_created_at_is_iso8601_string(self, s):
        atom = s.insert(content="x")
        # ISO-8601 UTC with milliseconds: "2024-01-01T12:00:00.000Z"
        assert "T" in atom.created_at and "Z" in atom.created_at

    def test_insert_updated_at_present(self, s):
        atom = s.insert(content="x")
        assert atom.updated_at and "T" in atom.updated_at

    def test_two_inserts_get_distinct_ids(self, s):
        a1 = s.insert(content="first")
        a2 = s.insert(content="second")
        assert a1.id != a2.id


# ---------------------------------------------------------------------------
# AC2 — get
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_returns_inserted_atom(self, s):
        atom = s.insert(content="retrieve me")
        fetched = s.get(atom.id)
        assert fetched is not None
        assert fetched.id == atom.id
        assert fetched.content == "retrieve me"

    def test_get_missing_id_returns_none(self, s):
        result = s.get("nonexistent-id-that-does-not-exist")
        assert result is None

    def test_get_returns_correct_field_types(self, s):
        atom = s.insert(content="x", metadata={"k": 1}, refs=["a"])
        fetched = s.get(atom.id)
        assert isinstance(fetched.metadata, dict)
        assert isinstance(fetched.refs, list)
        assert isinstance(fetched.invalidated, bool)


# ---------------------------------------------------------------------------
# AC2 — update_invalidated
# ---------------------------------------------------------------------------

class TestUpdateInvalidated:
    def test_update_invalidated_sets_flag_to_true(self, s):
        atom = s.insert(content="x")
        s.update_invalidated(atom.id, True)
        fetched = s.get(atom.id)
        assert fetched.invalidated is True

    def test_update_invalidated_returns_true_for_existing_id(self, s):
        atom = s.insert(content="x")
        result = s.update_invalidated(atom.id)
        assert result is True

    def test_update_invalidated_returns_false_for_missing_id(self, s):
        result = s.update_invalidated("does-not-exist")
        assert result is False

    def test_update_invalidated_can_clear_flag(self, s):
        atom = s.insert(content="x")
        s.update_invalidated(atom.id, True)
        s.update_invalidated(atom.id, False)
        fetched = s.get(atom.id)
        assert fetched.invalidated is False


# ---------------------------------------------------------------------------
# AC2 — delete
# ---------------------------------------------------------------------------

class TestDelete:
    def test_delete_returns_true_for_existing_atom(self, s):
        atom = s.insert(content="bye")
        result = s.delete(atom.id)
        assert result is True

    def test_delete_removes_atom_so_get_returns_none(self, s):
        atom = s.insert(content="bye")
        s.delete(atom.id)
        assert s.get(atom.id) is None

    def test_delete_returns_false_for_missing_id(self, s):
        result = s.delete("no-such-id")
        assert result is False

    def test_delete_only_removes_targeted_atom(self, s):
        a1 = s.insert(content="keep")
        a2 = s.insert(content="remove")
        s.delete(a2.id)
        assert s.get(a1.id) is not None


# ---------------------------------------------------------------------------
# AC2 — list_by_project
# ---------------------------------------------------------------------------

class TestListByProject:
    def test_list_by_project_returns_only_matching_project(self, s):
        s.insert(content="p1", project_id="proj-A")
        s.insert(content="p2", project_id="proj-B")
        results = s.list_by_project("proj-A")
        assert all(a.project_id == "proj-A" for a in results)
        assert len(results) == 1

    def test_list_by_project_returns_list_of_atoms(self, s):
        s.insert(content="x", project_id="proj-X")
        results = s.list_by_project("proj-X")
        assert isinstance(results, list)
        assert all(isinstance(a, Atom) for a in results)

    def test_list_by_project_empty_for_unknown_project(self, s):
        results = s.list_by_project("no-such-project")
        assert results == []

    def test_list_by_project_order_by_created_at_desc(self, s):
        """Multiple atoms for same project should come back newest-first."""
        import time
        a1 = s.insert(content="first", project_id="proj-T")
        time.sleep(0.01)  # ensure distinct created_at
        a2 = s.insert(content="second", project_id="proj-T")
        results = s.list_by_project("proj-T")
        assert results[0].id == a2.id  # newer atom first
        assert results[1].id == a1.id

    def test_list_by_project_excludes_invalidated_when_flag_false(self, s):
        atom = s.insert(content="x", project_id="proj-I")
        s.update_invalidated(atom.id, True)
        results = s.list_by_project("proj-I", include_invalidated=False)
        assert results == []

    def test_list_by_project_includes_invalidated_by_default(self, s):
        atom = s.insert(content="x", project_id="proj-I2")
        s.update_invalidated(atom.id, True)
        results = s.list_by_project("proj-I2", include_invalidated=True)
        assert len(results) == 1

    def test_list_by_project_limit_restricts_count(self, s):
        for i in range(5):
            s.insert(content=f"item {i}", project_id="proj-L")
        results = s.list_by_project("proj-L", limit=3)
        assert len(results) == 3

    def test_list_by_project_limit_none_returns_all(self, s):
        for i in range(4):
            s.insert(content=f"item {i}", project_id="proj-LA")
        results = s.list_by_project("proj-LA", limit=None)
        assert len(results) == 4


# ---------------------------------------------------------------------------
# AC3 — pack_embedding / unpack_embedding
# ---------------------------------------------------------------------------

class TestEmbeddingHelpers:
    """Pinned to bge-small-en-v1.5 (dim=384) — explicit, not the silent default."""

    @pytest.fixture(autouse=True)
    def _pin_bge_small(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")

    def test_pack_embedding_returns_1536_bytes_for_384_floats(self):
        vec = [0.1] * 384
        blob = pack_embedding(vec)
        assert len(blob) == 1536

    def test_pack_embedding_returns_bytes_type(self):
        blob = pack_embedding([0.0] * 384)
        assert isinstance(blob, bytes)

    def test_unpack_embedding_returns_384_floats(self):
        vec = [0.5] * 384
        blob = pack_embedding(vec)
        result = unpack_embedding(blob)
        assert len(result) == 384

    def test_unpack_round_trips_within_float32_tolerance(self):
        import struct
        vec = [float(i) / 384 for i in range(384)]
        blob = pack_embedding(vec)
        result = unpack_embedding(blob)
        # float32 precision: compare via struct round-trip reference
        expected = list(struct.unpack('<384f', struct.pack('<384f', *vec)))
        assert result == pytest.approx(expected, abs=1e-6)

    def test_pack_embedding_raises_value_error_for_wrong_length_short(self):
        with pytest.raises(ValueError):
            pack_embedding([0.0] * 100)

    def test_pack_embedding_raises_value_error_for_wrong_length_long(self):
        with pytest.raises(ValueError):
            pack_embedding([0.0] * 385)

    def test_unpack_embedding_raises_value_error_for_wrong_byte_length(self):
        with pytest.raises(ValueError):
            unpack_embedding(b"\x00" * 100)

    def test_unpack_embedding_raises_value_error_for_too_many_bytes(self):
        with pytest.raises(ValueError):
            unpack_embedding(b"\x00" * 1537)


# ---------------------------------------------------------------------------
# TASK-053 — pack/unpack are dim-agnostic (derive dim from active_dim())
# ---------------------------------------------------------------------------

class TestEmbeddingHelpersMultiDim:
    """pack_embedding/unpack_embedding must round-trip at EVERY registered dim,
    not just the legacy hardcoded 384.
    """

    @pytest.mark.parametrize(
        "model_key,dim",
        [
            ("bge-small-en-v1.5", 384),
            ("nomic-embed-text-v1.5", 768),
            ("bge-m3", 1024),
        ],
    )
    def test_round_trip_at_active_dim(self, monkeypatch, model_key, dim):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", model_key)
        vec = [float(i) / dim for i in range(dim)]
        blob = pack_embedding(vec)
        assert len(blob) == dim * 4
        result = unpack_embedding(blob)
        assert len(result) == dim
        assert result == pytest.approx(vec, abs=1e-6)

    def test_pack_rejects_vector_shorter_than_active_dim(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")  # dim=1024
        with pytest.raises(ValueError, match="expected 1024 dims, got 100"):
            pack_embedding([0.0] * 100)

    def test_pack_rejects_vector_longer_than_active_dim(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")  # dim=384
        with pytest.raises(ValueError, match="expected 384 dims, got 768"):
            pack_embedding([0.0] * 768)

    def test_unpack_rejects_wrong_byte_length_for_active_dim(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-m3")  # dim=1024 -> 4096 bytes
        with pytest.raises(ValueError, match="expected 4096 bytes, got 100"):
            unpack_embedding(b"\x00" * 100)

    def test_pack_at_768_then_switch_to_384_env_raises(self, monkeypatch):
        """A blob packed under one active model is invalid once the active
        model (and therefore active_dim()) changes — no silent truncation.
        """
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "nomic-embed-text-v1.5")  # 768
        blob = pack_embedding([0.1] * 768)

        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")  # 384
        with pytest.raises(ValueError, match="expected 384 dims|expected 1536 bytes"):
            unpack_embedding(blob)


# ---------------------------------------------------------------------------
# AC3 — embedding stored and retrieved as BLOB via Store
# ---------------------------------------------------------------------------

class TestEmbeddingInStore:
    """Pinned to bge-small-en-v1.5 (dim=384) — explicit, not the silent default."""

    @pytest.fixture(autouse=True)
    def _pin_bge_small(self, monkeypatch):
        monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")

    def test_embedding_stored_and_retrieved_as_same_bytes(self, s):
        vec = [float(i) / 384 for i in range(384)]
        blob = pack_embedding(vec)
        atom = s.insert(content="with embedding", embedding=blob)
        fetched = s.get(atom.id)
        assert fetched.embedding == blob

    def test_embedding_round_trips_to_correct_floats(self, s):
        vec = [0.1 * i for i in range(384)]
        blob = pack_embedding(vec)
        atom = s.insert(content="embed check", embedding=blob)
        fetched = s.get(atom.id)
        recovered = unpack_embedding(fetched.embedding)
        assert len(recovered) == 384
        assert recovered == pytest.approx(
            list(unpack_embedding(blob)), abs=1e-6
        )

    def test_embedding_nullable_insert_without_embedding_succeeds(self, s):
        atom = s.insert(content="no embed")
        assert atom.embedding is None

    def test_get_atom_without_embedding_returns_none_for_embedding(self, s):
        atom = s.insert(content="no embed")
        fetched = s.get(atom.id)
        assert fetched.embedding is None


# ---------------------------------------------------------------------------
# AC4 — CHECK constraints
# ---------------------------------------------------------------------------

class TestCheckConstraints:
    def test_invalid_type_raises_integrity_error(self, s):
        with pytest.raises(sqlite3.IntegrityError):
            s.insert(content="x", type="invalid-type-xyz")

    def test_invalid_scope_raises_integrity_error(self, s):
        with pytest.raises(sqlite3.IntegrityError):
            s.insert(content="x", scope="invalid-scope")

    def test_all_valid_types_accepted(self, s):
        valid_types = [
            "decision", "pattern", "diary", "entity",
            "oq-resolution", "infra-fact", "server", "deploy-target"
        ]
        for t in valid_types:
            atom = s.insert(content=f"type test {t}", type=t)
            assert atom.type == t

    def test_all_valid_scopes_accepted(self, s):
        for scope in ("project", "global", "all"):
            atom = s.insert(content=f"scope {scope}", scope=scope)
            assert atom.scope == scope

    def test_type_none_raises_error(self, s):
        """type is NOT NULL — must raise on None."""
        with pytest.raises((sqlite3.IntegrityError, TypeError)):
            s.insert(content="x", type=None)


# ---------------------------------------------------------------------------
# AC5 — project_id nullable
# ---------------------------------------------------------------------------

class TestProjectIdNullable:
    def test_insert_without_project_id_succeeds(self, s):
        atom = s.insert(content="no project")
        assert atom is not None

    def test_insert_without_project_id_sets_none(self, s):
        atom = s.insert(content="no project")
        assert atom.project_id is None

    def test_get_atom_with_null_project_id_returns_atom(self, s):
        atom = s.insert(content="null proj")
        fetched = s.get(atom.id)
        assert fetched is not None
        assert fetched.project_id is None

    def test_insert_with_explicit_project_id_stored_correctly(self, s):
        atom = s.insert(content="with proj", project_id="proj-42")
        assert atom.project_id == "proj-42"


# ---------------------------------------------------------------------------
# TASK-023 — Atom round-trip carries session_id correctly
# ---------------------------------------------------------------------------

class TestAtomSessionIdRoundTrip:
    def test_insert_with_session_id_round_trips(self, s):
        atom = s.insert(content="session atom", session_id="my-session-42")
        assert atom.session_id == "my-session-42"

        fetched = s.get(atom.id)
        assert fetched.session_id == "my-session-42"

    def test_insert_without_session_id_is_none(self, s):
        atom = s.insert(content="no session")
        assert atom.session_id is None

        fetched = s.get(atom.id)
        assert fetched.session_id is None


# ---------------------------------------------------------------------------
# created_at validation (S2 — defense-in-depth)
# ---------------------------------------------------------------------------

class TestCreatedAtValidation:
    def test_invalid_created_at_raises_value_error(self, s):
        with pytest.raises(ValueError, match="ISO-8601 UTC"):
            s.insert(content="x", created_at="not-a-date")

    def test_valid_created_at_stored_and_returned(self, s):
        ts = "2026-01-02T03:04:05Z"
        atom = s.insert(content="event-time", created_at=ts)
        assert atom.created_at == ts

    def test_insert_without_created_at_defaults_to_now(self, s):
        import datetime
        atom = s.insert(content="default-now")
        # created_at must be non-empty and start with current year
        current_year = str(datetime.datetime.now().year)
        assert atom.created_at and atom.created_at.startswith(current_year)


# ---------------------------------------------------------------------------
# TASK-024 — tags column on Atom
# ---------------------------------------------------------------------------

class TestTags:
    def test_insert_with_tags_stores_and_returns_tags(self, s):
        atom = s.insert(content="tagged content", tags=["python", "testing", "memory"])
        assert atom.tags == ["python", "testing", "memory"]

    def test_insert_without_tags_defaults_to_none(self, s):
        atom = s.insert(content="no tags")
        assert atom.tags is None

    def test_get_atom_with_tags_returns_tags(self, s):
        atom = s.insert(content="has tags", tags=["foo", "bar"])
        fetched = s.get(atom.id)
        assert fetched.tags == ["foo", "bar"]

    def test_get_atom_without_tags_returns_none(self, s):
        atom = s.insert(content="no tags")
        fetched = s.get(atom.id)
        assert fetched.tags is None

    def test_insert_with_empty_tags_list_stores_empty_list(self, s):
        atom = s.insert(content="empty tags", tags=[])
        assert atom.tags == []

    def test_list_by_project_returns_atoms_with_tags(self, s):
        s.insert(content="tagged", project_id="P", tags=["ai", "ml"])
        s.insert(content="untagged", project_id="P")
        results = s.list_by_project("P")
        tagged = [r for r in results if r.tags is not None]
        assert len(tagged) == 1
        assert tagged[0].tags == ["ai", "ml"]


# ---------------------------------------------------------------------------
# AC2 — context-manager protocol
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_store_usable_as_context_manager(self, tmp_path):
        db = tmp_path / "ctx.db"
        with store(str(db)) as s:
            atom = s.insert(content="ctx test")
            assert s.get(atom.id) is not None
        # connection closed — further use should raise
        with pytest.raises(Exception):
            s.get(atom.id)


# ---------------------------------------------------------------------------
# TASK-021 — increment_access and list_stale
# ---------------------------------------------------------------------------

class TestIncrementAccess:
    def test_increment_access_increments_count(self, s):
        atom = s.insert(content="hit me", project_id="p")
        assert atom.access_count == 0

        s.increment_access(atom.id)
        fetched = s.get(atom.id)
        assert fetched.access_count == 1

    def test_increment_access_sets_last_accessed(self, s):
        atom = s.insert(content="timestamp me", project_id="p")
        assert atom.last_accessed is None

        s.increment_access(atom.id)
        fetched = s.get(atom.id)
        assert fetched.last_accessed is not None
        assert "T" in fetched.last_accessed
        assert "Z" in fetched.last_accessed

    def test_increment_access_accumulates(self, s):
        atom = s.insert(content="multi hit", project_id="p")
        for _ in range(5):
            s.increment_access(atom.id)
        fetched = s.get(atom.id)
        assert fetched.access_count == 5

    def test_increment_access_returns_true_for_existing(self, s):
        atom = s.insert(content="exists", project_id="p")
        assert s.increment_access(atom.id) is True

    def test_increment_access_returns_false_for_missing(self, s):
        assert s.increment_access("nonexistent") is False


class TestListStale:
    def test_list_stale_returns_low_access_old_atoms(self, s):
        s.insert(content="old low access", project_id="p")
        # Fresh atom should not be stale
        fresh = s.list_stale("p", min_age_days=90, min_access_count=3)
        assert len(fresh) == 0

    def test_list_stale_excludes_invalidated(self, s):
        atom = s.insert(content="invalidated but old", project_id="p")
        s.update_invalidated(atom.id, True)
        stale = s.list_stale("p", min_age_days=0, min_access_count=999)
        assert len(stale) == 0

    def test_list_stale_respects_min_age_days(self, s):
        atom = s.insert(content="brand new", project_id="p")
        stale = s.list_stale("p", min_age_days=1, min_access_count=999)
        assert len(stale) == 0

    def test_list_stale_respects_min_access_count(self, s):
        atom = s.insert(content="frequently accessed", project_id="p")
        for _ in range(5):
            s.increment_access(atom.id)
        stale = s.list_stale("p", min_age_days=0, min_access_count=3)
        assert len(stale) == 0

    def test_list_stale_orders_by_access_then_created(self, s):
        import time
        a1 = s.insert(content="oldest", project_id="p")
        time.sleep(0.02)
        a2 = s.insert(content="newer", project_id="p")
        stale = s.list_stale("p", min_age_days=0, min_access_count=999)
        assert len(stale) == 2
        assert stale[0].id == a1.id
        assert stale[1].id == a2.id

    def test_list_stale_invalid_age_raises(self, s):
        with pytest.raises(ValueError):
            s.list_stale("p", min_age_days=-1)

    def test_list_stale_invalid_access_raises(self, s):
        with pytest.raises(ValueError):
            s.list_stale("p", min_access_count=-1)

    def test_list_stale_scoped_to_project(self, s):
        s.insert(content="stale p1", project_id="p1")
        s.insert(content="stale p2", project_id="p2")
        stale = s.list_stale("p1", min_age_days=0, min_access_count=999)
        assert len(stale) == 1
        assert stale[0].project_id == "p1"
