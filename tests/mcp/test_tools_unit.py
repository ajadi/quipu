"""Unit tests for quipu/mcp/tools.py — dispatch called directly with injected Store."""

from __future__ import annotations

import json

import pytest

from quipu.mcp.tools import dispatch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse(result) -> dict:
    """Parse the first TextContent's JSON text."""
    assert len(result) == 1
    return json.loads(result[0].text)


# ---------------------------------------------------------------------------
# quipu_write
# ---------------------------------------------------------------------------


class TestQuipuWrite:
    def test_write_returns_id(self, tmp_store, project_id, fake_engine):
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "hello world"},
        )
        data = _parse(result)
        assert "id" in data
        assert isinstance(data["id"], str)
        assert len(data["id"]) > 0

    def test_write_returns_conflicts_key(self, tmp_store, project_id, fake_engine):
        """quipu_write always returns a 'conflicts' key."""
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "first atom"},
        )
        data = _parse(result)
        assert "conflicts" in data
        assert isinstance(data["conflicts"], list)

    def test_write_with_metadata(self, tmp_store, project_id, fake_engine):
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "meta test", "metadata": {"tag": "test"}},
        )
        data = _parse(result)
        assert "id" in data

    def test_write_missing_content(self, tmp_store, project_id):
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_write_empty_content(self, tmp_store, project_id):
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": ""},
        )
        data = _parse(result)
        assert "error" in data

    def test_write_bad_metadata_type(self, tmp_store, project_id):
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "x", "metadata": "not-a-dict"},
        )
        data = _parse(result)
        assert "error" in data

    def test_write_uses_arg_project_id(self, tmp_store, fake_engine):
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=None,
            arguments={"content": "hello", "project_id": "explicit_proj"},
        )
        data = _parse(result)
        assert "id" in data
        atom = tmp_store.get(data["id"])
        assert atom.project_id == "explicit_proj"


# ---------------------------------------------------------------------------
# quipu_search
# ---------------------------------------------------------------------------


class TestQuipuSearch:
    def test_search_no_project_id_returns_error(self, tmp_store):
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=None,
            arguments={"query": "hello"},
        )
        data = _parse(result)
        assert "error" in data

    def test_search_missing_query(self, tmp_store, project_id):
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_search_bad_top_k(self, tmp_store, project_id):
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "hello", "top_k": 0},
        )
        data = _parse(result)
        assert "error" in data

    def test_search_returns_results_list(self, tmp_store, project_id, fake_engine):
        # Write something first
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "the quick brown fox"},
        )
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "quick fox"},
        )
        data = _parse(result)
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_search_result_fields(self, tmp_store, project_id, fake_engine):
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "search result fields test"},
        )
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "search result", "top_k": 5},
        )
        data = _parse(result)
        if data["results"]:
            r = data["results"][0]
            for field in ("id", "content", "score", "tier", "type", "scope", "invalidated", "metadata"):
                assert field in r, f"missing field {field}"
            assert "embedding" not in r

    def test_write_search_roundtrip(self, tmp_store, project_id, fake_engine):
        # Write then search — written id must appear in results
        write_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "unique content for roundtrip test"},
        )
        written_id = _parse(write_result)["id"]

        search_result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "unique content roundtrip"},
        )
        data = _parse(search_result)
        ids = [r["id"] for r in data["results"]]
        assert written_id in ids

    def test_search_tag_filter(self, tmp_store, project_id, fake_engine):
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "python machine learning libraries"},
        )
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "javascript frontend frameworks"},
        )
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "learning", "tags": ["python"]},
        )
        data = _parse(result)
        assert "results" in data
        assert len(data["results"]) == 1
        assert "python" in data["results"][0]["content"]

    def test_search_invalid_tags_type(self, tmp_store, project_id):
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "hello", "tags": "not-a-list"},
        )
        data = _parse(result)
        assert "error" in data
        assert "tags" in data["error"]

    def test_search_tags_non_string_items(self, tmp_store, project_id):
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "hello", "tags": [1, 2, 3]},
        )
        data = _parse(result)
        assert "error" in data
        assert "tags" in data["error"]


# ---------------------------------------------------------------------------
# quipu_get
# ---------------------------------------------------------------------------


class TestQuipuGet:
    def test_get_found(self, tmp_store, project_id, fake_engine):
        write_r = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "get me"},
        )
        atom_id = _parse(write_r)["id"]

        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert data["id"] == atom_id
        assert data["content"] == "get me"
        assert "embedding" not in data

    def test_get_not_found(self, tmp_store, project_id):
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": "nonexistent_id"},
        )
        data = _parse(result)
        assert data == {"found": False}

    def test_get_missing_id(self, tmp_store, project_id):
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# quipu_list
# ---------------------------------------------------------------------------


class TestQuipuList:
    def test_list_no_project_id_returns_error(self, tmp_store):
        result = dispatch(
            "quipu_list",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_list_returns_atoms(self, tmp_store, project_id, fake_engine):
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "atom1"},
        )
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "atom2"},
        )
        result = dispatch(
            "quipu_list",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "atoms" in data
        assert len(data["atoms"]) == 2
        # newest first: atom2 should come before atom1
        assert data["atoms"][0]["content"] == "atom2"

    def test_list_with_limit(self, tmp_store, project_id, fake_engine):
        for i in range(5):
            dispatch(
                "quipu_write",
                store=tmp_store,
                default_project_id=project_id,
                arguments={"content": f"atom{i}"},
            )
        result = dispatch(
            "quipu_list",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"limit": 3},
        )
        data = _parse(result)
        assert len(data["atoms"]) == 3

    def test_list_bad_limit(self, tmp_store, project_id):
        result = dispatch(
            "quipu_list",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"limit": 0},
        )
        data = _parse(result)
        assert "error" in data

    def test_list_no_embedding_in_atoms(self, tmp_store, project_id, fake_engine):
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "check fields"},
        )
        result = dispatch(
            "quipu_list",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert len(data["atoms"]) == 1
        assert "embedding" not in data["atoms"][0]


# ---------------------------------------------------------------------------
# quipu_invalidate
# ---------------------------------------------------------------------------


class TestQuipuInvalidate:
    def test_invalidate_existing(self, tmp_store, project_id, fake_engine):
        write_r = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "to be invalidated"},
        )
        atom_id = _parse(write_r)["id"]

        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert data["id"] == atom_id
        assert data["invalidated"] is True
        assert data["existed"] is True

    def test_invalidate_nonexistent(self, tmp_store, project_id):
        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": "ghost_id"},
        )
        data = _parse(result)
        assert data["existed"] is False

    def test_invalidate_missing_id(self, tmp_store, project_id):
        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_invalidate_concurrent_delete_returns_not_success(
        self, tmp_store, project_id, fake_engine, monkeypatch
    ):
        """Concurrent-delete race: store.get returns None after update_invalidated.

        When the row vanishes between update_invalidated and the re-fetch, the
        handler must NOT claim success (sync divergence — oplog entry was not
        emitted). Returns invalidated:False, existed:False.
        """
        write_r = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "will be deleted concurrently"},
        )
        atom_id = _parse(write_r)["id"]

        # Patch store.get to return None after the first real call (pre-check
        # in the handler sees the atom; re-fetch after update sees None).
        _original_get = tmp_store.get
        call_count = {"n": 0}

        def _patched_get(aid):
            call_count["n"] += 1
            if aid == atom_id and call_count["n"] >= 2:
                return None
            return _original_get(aid)

        monkeypatch.setattr(tmp_store, "get", _patched_get)

        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert data["invalidated"] is False
        assert data["existed"] is False


# ---------------------------------------------------------------------------
# quipu_flush
# ---------------------------------------------------------------------------


class TestQuipuFlush:
    def test_flush_no_api_key_skips(self, tmp_store, project_id, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = dispatch(
            "quipu_flush",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["enriched"] == 0
        assert data["skipped"] is True
        assert data["reason"] == "no_api_key"

    def test_flush_returns_expected_keys(self, tmp_store, project_id, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = dispatch(
            "quipu_flush",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "enriched" in data
        assert "skipped" in data
        assert "reason" in data


# ---------------------------------------------------------------------------
# quipu_stats
# ---------------------------------------------------------------------------


class TestQuipuStats:
    def test_stats_empty(self, tmp_store, project_id):
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["total"] == 0
        assert data["active"] == 0
        assert data["invalidated"] == 0
        assert data["last_flush"] is None

    def test_stats_counts(self, tmp_store, project_id):
        # Insert 3 atoms directly (bypasses auto-invalidation from identical fake embeddings)
        ids = []
        for i in range(3):
            atom = tmp_store.insert(content=f"atom{i}", project_id=project_id, metadata={})
            ids.append(atom.id)
        dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": ids[0]},
        )
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["total"] == 3
        assert data["invalidated"] == 1
        assert data["active"] == 2

    def test_stats_last_flush_from_enriched_at(self, tmp_store, project_id):
        # Insert atoms directly with enriched_at metadata
        tmp_store.insert(
            content="first",
            project_id=project_id,
            metadata={"enriched_at": "2024-01-01T10:00:00"},
        )
        tmp_store.insert(
            content="second",
            project_id=project_id,
            metadata={"enriched_at": "2024-06-15T12:30:00"},
        )
        tmp_store.insert(
            content="third",
            project_id=project_id,
            metadata={},  # no enriched_at
        )
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        # lexicographic max of ISO strings
        assert data["last_flush"] == "2024-06-15T12:30:00"

    def test_stats_no_project_id_returns_error(self, tmp_store):
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# Unknown tool
# ---------------------------------------------------------------------------


class TestUnknownTool:
    def test_unknown_tool_returns_error(self, tmp_store, project_id):
        result = dispatch(
            "quipu_nonexistent",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data
        assert "unknown tool" in data["error"]


# ---------------------------------------------------------------------------
# Security: cross-project isolation (FINDING 1)
# ---------------------------------------------------------------------------


class TestCrossProjectIsolation:
    def test_get_foreign_project_atom_returns_not_found(self, tmp_store, fake_engine):
        """quipu_get of an atom from a different project returns {"found": false}."""
        # Write atom to project_a
        write_r = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id="project_a",
            arguments={"content": "secret data in A"},
        )
        atom_id = _parse(write_r)["id"]

        # Attempt to get from project_b — must not return the atom
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id="project_b",
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert data == {"found": False}

    def test_get_no_project_id_returns_error(self, tmp_store):
        """quipu_get with no project_id (default None) raises structured error."""
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id=None,
            arguments={"id": "some_id"},
        )
        data = _parse(result)
        assert "error" in data
        assert "project_id" in data["error"]

    def test_get_same_project_still_works(self, tmp_store, project_id, fake_engine):
        """quipu_get of an atom in the same project still returns the atom."""
        write_r = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "my own data"},
        )
        atom_id = _parse(write_r)["id"]

        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert data["id"] == atom_id
        assert data["content"] == "my own data"

    def test_invalidate_foreign_project_atom_does_not_invalidate(
        self, tmp_store, fake_engine
    ):
        """quipu_invalidate of a foreign-project atom returns existed=False and leaves atom active."""
        # Write atom to project_a
        write_r = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id="project_a",
            arguments={"content": "sensitive atom in A"},
        )
        atom_id = _parse(write_r)["id"]

        # Attempt to invalidate from project_b
        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id="project_b",
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert data["existed"] is False
        assert data["invalidated"] is False

        # Confirm the original atom is still active
        atom = tmp_store.get(atom_id)
        assert atom is not None
        assert atom.invalidated is False

    def test_invalidate_no_project_id_returns_error(self, tmp_store):
        """quipu_invalidate with no project_id (default None) raises structured error."""
        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=None,
            arguments={"id": "some_id"},
        )
        data = _parse(result)
        assert "error" in data
        assert "project_id" in data["error"]

    def test_invalidate_same_project_still_works(self, tmp_store, project_id, fake_engine):
        """quipu_invalidate of an atom in the same project still invalidates it."""
        write_r = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "to be invalidated"},
        )
        atom_id = _parse(write_r)["id"]

        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert data["existed"] is True
        assert data["invalidated"] is True

        atom = tmp_store.get(atom_id)
        assert atom.invalidated is True


# ---------------------------------------------------------------------------
# Bool-as-int guard (FINDING 2)
# ---------------------------------------------------------------------------


class TestBoolAsIntGuard:
    def test_search_top_k_true_returns_error(self, tmp_store, project_id):
        """quipu_search with top_k=True (JSON true, bool subclass of int) → error."""
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "hello", "top_k": True},
        )
        data = _parse(result)
        assert "error" in data
        assert "top_k" in data["error"]

    def test_search_top_k_false_returns_error(self, tmp_store, project_id):
        """quipu_search with top_k=False → error (False == 0 < 1)."""
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "hello", "top_k": False},
        )
        data = _parse(result)
        assert "error" in data

    def test_list_limit_true_returns_error(self, tmp_store, project_id):
        """quipu_list with limit=True → error."""
        result = dispatch(
            "quipu_list",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"limit": True},
        )
        data = _parse(result)
        assert "error" in data
        assert "limit" in data["error"]


# ---------------------------------------------------------------------------
# Security hardening: bound-project scope lock (FIX 1)
# ---------------------------------------------------------------------------


class TestBoundProjectScopeLock:
    """When server is bound (default_project_id != None), a client-supplied
    project_id that differs from the bound project MUST be rejected with a
    structured error — not silently served."""

    def _write_to_project(self, tmp_store, project_id, content, fake_engine):
        """Helper: write an atom directly via dispatch to the given project."""
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": content},
        )
        return _parse(result)["id"]

    def test_get_scope_escape_denied(self, tmp_store, fake_engine):
        """quipu_get with foreign project_id on bound server → scope error, not atom."""
        atom_id = self._write_to_project(tmp_store, "B", "secret in B", fake_engine)
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id="A",
            arguments={"id": atom_id, "project_id": "B"},
        )
        data = _parse(result)
        assert "error" in data
        assert "not permitted" in data["error"] or "bound" in data["error"]
        # Confirm it is NOT the atom and NOT {"found": false}
        assert "found" not in data
        assert "content" not in data

    def test_invalidate_scope_escape_denied(self, tmp_store, fake_engine):
        """quipu_invalidate with foreign project_id on bound server → scope error."""
        atom_id = self._write_to_project(tmp_store, "B", "data in B", fake_engine)
        result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id="A",
            arguments={"id": atom_id, "project_id": "B"},
        )
        data = _parse(result)
        assert "error" in data
        assert "not permitted" in data["error"] or "bound" in data["error"]
        # Confirm atom is still active (not invalidated)
        atom = tmp_store.get(atom_id)
        assert atom is not None
        assert atom.invalidated is False

    def test_list_scope_escape_denied(self, tmp_store, fake_engine):
        """quipu_list with foreign project_id on bound server → scope error."""
        self._write_to_project(tmp_store, "B", "data in B", fake_engine)
        result = dispatch(
            "quipu_list",
            store=tmp_store,
            default_project_id="A",
            arguments={"project_id": "B"},
        )
        data = _parse(result)
        assert "error" in data
        assert "not permitted" in data["error"] or "bound" in data["error"]

    def test_write_scope_escape_denied(self, tmp_store):
        """quipu_write with foreign project_id on bound server → scope error."""
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id="A",
            arguments={"content": "injected into B", "project_id": "B"},
        )
        data = _parse(result)
        assert "error" in data
        assert "not permitted" in data["error"] or "bound" in data["error"]

    def test_stats_scope_escape_denied(self, tmp_store):
        """quipu_stats with foreign project_id on bound server → scope error."""
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id="A",
            arguments={"project_id": "B"},
        )
        data = _parse(result)
        assert "error" in data
        assert "not permitted" in data["error"] or "bound" in data["error"]

    def test_same_project_allowed(self, tmp_store, fake_engine):
        """Supplying project_id == default_project_id on bound server → allowed."""
        atom_id = self._write_to_project(tmp_store, "A", "my data", fake_engine)
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id="A",
            arguments={"id": atom_id, "project_id": "A"},
        )
        data = _parse(result)
        assert "error" not in data
        assert data["id"] == atom_id

    def test_absent_project_id_uses_default(self, tmp_store, fake_engine):
        """Omitting project_id on bound server → uses default (no error)."""
        atom_id = self._write_to_project(tmp_store, "A", "my data 2", fake_engine)
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id="A",
            arguments={"id": atom_id},
        )
        data = _parse(result)
        assert "error" not in data
        assert data["id"] == atom_id

    def test_unbound_server_allows_any_project(self, tmp_store, fake_engine):
        """Unbound server (default_project_id=None) + project_id arg → no scope error."""
        atom_id = self._write_to_project(tmp_store, "X", "data in X", fake_engine)
        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id=None,
            arguments={"id": atom_id, "project_id": "X"},
        )
        data = _parse(result)
        # No scope error — proceeds normally (finds the atom)
        assert "not permitted" not in data.get("error", "")
        assert "bound" not in data.get("error", "")
        assert data.get("id") == atom_id


# ---------------------------------------------------------------------------
# Security hardening: top_k upper cap (FIX 3)
# ---------------------------------------------------------------------------


class TestTopKCap:
    def test_top_k_5000_rejected(self, tmp_store, project_id):
        """quipu_search with top_k=5000 → structured error (not silently clamped)."""
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "anything", "top_k": 5000},
        )
        data = _parse(result)
        assert "error" in data
        assert "1000" in data["error"]

    def test_top_k_1000_allowed(self, tmp_store, project_id, fake_engine):
        """quipu_search with top_k=1000 → not rejected (boundary is inclusive)."""
        result = dispatch(
            "quipu_search",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"query": "anything", "top_k": 1000},
        )
        data = _parse(result)
        assert "error" not in data or "1000" not in data.get("error", "")


# ---------------------------------------------------------------------------
# Security hardening: content length cap (FIX 4)
# ---------------------------------------------------------------------------


class TestContentLengthCap:
    def test_content_100001_rejected(self, tmp_store, project_id):
        """quipu_write with content 100001 chars → structured error."""
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "x" * 100_001},
        )
        data = _parse(result)
        assert "error" in data
        assert "100000" in data["error"]

    def test_content_100000_allowed(self, tmp_store, project_id, fake_engine):
        """quipu_write with content exactly 100000 chars → accepted."""
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "x" * 100_000},
        )
        data = _parse(result)
        assert "id" in data


# ---------------------------------------------------------------------------
# Conflict detection (TASK-020)
# ---------------------------------------------------------------------------


def _make_vec_engine(vec):
    """Build a fake engine returning a specific pre-set L2-normalized vector."""
    from quipu.embeddings.engine import _Engine, set_engine, EMBED_DIM

    class _N:
        def __init__(self, name):
            self.name = name
            self.type = "tensor(int64)"

    class _VecSession:
        def __init__(self, v):
            self._v = v

        def get_inputs(self):
            return [_N("input_ids"), _N("attention_mask")]

        def get_outputs(self):
            return [_N("sentence_embedding")]

        def run(self, output_names, feeds):
            import numpy as np
            n = feeds["input_ids"].shape[0]
            arr = np.array([self._v] * n, dtype=np.float32)
            return [arr]

    class _FakeTok:
        def encode_batch(self, texts):
            class _E:
                ids = [1] * 8
                attention_mask = [1] * 8
            return [_E() for _ in texts]

    engine = _Engine(session=_VecSession(vec), tokenizer=_FakeTok())
    set_engine(engine)
    return engine


def _unit_vec_mcp(dim, index):
    v = [0.0] * dim
    v[index] = 1.0
    return v


class TestWriteConflictDetection:
    """MCP handler returns conflict signal on near-duplicate write (TASK-020)."""

    def test_no_conflict_first_write(self, tmp_store, project_id):
        """First write to an empty project → conflicts == []."""
        from quipu.embeddings.engine import EMBED_DIM
        v = _unit_vec_mcp(EMBED_DIM, 0)
        _make_vec_engine(v)
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "first atom"},
        )
        data = _parse(result)
        assert data["conflicts"] == []

    def test_near_dup_returns_conflict(self, tmp_store, project_id):
        """Near-duplicate write returns non-empty conflicts with id, similarity, snippet."""
        from quipu.embeddings.engine import EMBED_DIM
        v = _unit_vec_mcp(EMBED_DIM, 0)

        # Write first atom
        _make_vec_engine(v)
        first_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "original content"},
        )
        first_id = _parse(first_result)["id"]

        # Write near-duplicate (same vector)
        _make_vec_engine(v)
        second_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "duplicate content"},
        )
        data = _parse(second_result)
        assert "id" in data
        assert len(data["conflicts"]) > 0
        conflict = data["conflicts"][0]
        assert conflict["id"] == first_id
        assert "similarity" in conflict
        assert "snippet" in conflict
        assert isinstance(conflict["similarity"], float)
        assert conflict["snippet"] == "original content"

    def test_near_dup_new_atom_persisted_active(self, tmp_store, project_id):
        """After near-dup write, the NEW atom is persisted and active."""
        from quipu.embeddings.engine import EMBED_DIM
        v = _unit_vec_mcp(EMBED_DIM, 0)

        _make_vec_engine(v)
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "old"},
        )

        _make_vec_engine(v)
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "new"},
        )
        new_id = _parse(result)["id"]
        atom = tmp_store.get(new_id)
        assert atom is not None
        assert not atom.invalidated

    def test_near_dup_old_atom_still_active(self, tmp_store, project_id):
        """After near-dup write, the OLD conflicting atom remains active (no auto-invalidation)."""
        from quipu.embeddings.engine import EMBED_DIM
        v = _unit_vec_mcp(EMBED_DIM, 0)

        _make_vec_engine(v)
        first_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "first"},
        )
        first_id = _parse(first_result)["id"]

        _make_vec_engine(v)
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "second"},
        )

        old_atom = tmp_store.get(first_id)
        assert not old_atom.invalidated, "Old atom must remain ACTIVE (keep-both, AC4)"

    def test_unscoped_no_conflicts(self, tmp_store):
        """project_id=None (unscoped) → conflicts == []."""
        from quipu.embeddings.engine import EMBED_DIM
        v = _unit_vec_mcp(EMBED_DIM, 0)

        _make_vec_engine(v)
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=None,
            arguments={"content": "unscoped first", "project_id": None},
        )

        _make_vec_engine(v)
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=None,
            arguments={"content": "unscoped second", "project_id": None},
        )
        # quipu_write does NOT require project_id, so unscoped write always succeeds.
        # When project_id is None, conflicts must be [].
        raw = result[0].text
        import json as _json
        data = _json.loads(raw)
        assert "error" not in data, f"unscoped write must not error: {data}"
        assert data["conflicts"] == []

    def test_keep_both_no_second_call_old_stays_active(self, tmp_store, project_id):
        """AC4: write near-dup, no 2nd call → old atom still active, zero invalidate entries in store."""
        from quipu.embeddings.engine import EMBED_DIM
        v = _unit_vec_mcp(EMBED_DIM, 0)

        _make_vec_engine(v)
        first_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "keep-both first"},
        )
        first_id = _parse(first_result)["id"]

        _make_vec_engine(v)
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "keep-both second"},
        )

        # No further call — old atom must be active (keep-both by default)
        old_atom = tmp_store.get(first_id)
        assert not old_atom.invalidated, "AC4: keep-both → old atom must stay active"

        # Confirm zero new invalidate entries in the store
        all_atoms = tmp_store.list_by_project(project_id, include_invalidated=True)
        invalidated_atoms = [a for a in all_atoms if a.invalidated]
        assert len(invalidated_atoms) == 0, "AC4: no atoms should be invalidated"

    def test_supersede_path_invalidates_conflict(self, tmp_store, project_id):
        """AC3: write near-dup → quipu_invalidate(conflict_id) → old atom invalidated."""
        from quipu.embeddings.engine import EMBED_DIM
        v = _unit_vec_mcp(EMBED_DIM, 0)

        _make_vec_engine(v)
        first_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "supersede-me"},
        )
        first_id = _parse(first_result)["id"]

        _make_vec_engine(v)
        second_result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "supersede-target"},
        )
        data = _parse(second_result)
        conflicts = data["conflicts"]
        assert len(conflicts) > 0, "Expected a conflict"
        conflict_id = conflicts[0]["id"]
        assert conflict_id == first_id

        # Caller supersedes explicitly
        inv_result = dispatch(
            "quipu_invalidate",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": conflict_id},
        )
        inv_data = _parse(inv_result)
        assert inv_data["invalidated"] is True

        old_atom = tmp_store.get(first_id)
        assert old_atom.invalidated, "AC3: old atom must be invalidated after quipu_invalidate"

    def test_orthogonal_write_no_conflict(self, tmp_store, project_id):
        """Orthogonal embeddings produce no conflicts."""
        from quipu.embeddings.engine import EMBED_DIM
        v1 = _unit_vec_mcp(EMBED_DIM, 0)
        v2 = _unit_vec_mcp(EMBED_DIM, 1)

        _make_vec_engine(v1)
        dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "vector one"},
        )

        _make_vec_engine(v2)
        result = dispatch(
            "quipu_write",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"content": "vector two"},
        )
        data = _parse(result)
        assert data["conflicts"] == []


# ---------------------------------------------------------------------------
# Security hardening: error info hygiene (FIX 5)
# ---------------------------------------------------------------------------


class TestErrorInfoHygiene:
    def test_unexpected_exception_returns_generic_message(
        self, tmp_store, project_id, monkeypatch
    ):
        """Unexpected RuntimeError in a handler → client sees 'internal error', not the message."""
        # Monkeypatch store.get to raise with a path that must NOT leak to client.
        def _raising_get(atom_id):
            raise RuntimeError("secret path /etc/x")

        monkeypatch.setattr(tmp_store, "get", _raising_get)

        result = dispatch(
            "quipu_get",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"id": "any_id"},
        )
        data = _parse(result)
        assert data == {"error": "internal error"}, (
            f"expected exactly {{\"error\": \"internal error\"}}, got: {data}"
        )


# ---------------------------------------------------------------------------
# TASK-021 — quipu_gc
# ---------------------------------------------------------------------------

class TestQuipuGc:
    def test_gc_dry_run_lists_stale(self, tmp_store, project_id):
        tmp_store.insert(content="old record", project_id=project_id)
        result = dispatch(
            "quipu_gc",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"min_age_days": 0, "min_access_count": 999},
        )
        data = _parse(result)
        assert data["dry_run"] is True
        assert data["invalidated"] == 0
        assert data["stale_count"] >= 1
        assert len(data["stale"]) == data["stale_count"]

    def test_gc_run_invalidates_stale(self, tmp_store, project_id):
        atom = tmp_store.insert(content="stale record", project_id=project_id)
        assert atom.invalidated is False

        result = dispatch(
            "quipu_gc",
            store=tmp_store,
            default_project_id=project_id,
            arguments={
                "run": True,
                "min_age_days": 0,
                "min_access_count": 999,
            },
        )
        data = _parse(result)
        assert data["dry_run"] is False
        assert data["invalidated"] >= 1

        fetched = tmp_store.get(atom.id)
        assert fetched.invalidated is True

    def test_gc_dry_run_does_not_invalidate(self, tmp_store, project_id):
        atom = tmp_store.insert(content="keep me", project_id=project_id)

        result = dispatch(
            "quipu_gc",
            store=tmp_store,
            default_project_id=project_id,
            arguments={
                "min_age_days": 0,
                "min_access_count": 999,
            },
        )
        data = _parse(result)
        assert data["dry_run"] is True
        assert data["invalidated"] == 0

        fetched = tmp_store.get(atom.id)
        assert fetched.invalidated is False

    def test_gc_no_stale_when_accessed(self, tmp_store, project_id):
        atom = tmp_store.insert(content="popular record", project_id=project_id)
        for _ in range(10):
            tmp_store.increment_access(atom.id)

        result = dispatch(
            "quipu_gc",
            store=tmp_store,
            default_project_id=project_id,
            arguments={
                "min_age_days": 0,
                "min_access_count": 3,
            },
        )
        data = _parse(result)
        assert data["stale_count"] == 0

    def test_gc_requires_project_id(self, tmp_store):
        result = dispatch(
            "quipu_gc",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_gc_rejects_invalid_min_age(self, tmp_store, project_id):
        result = dispatch(
            "quipu_gc",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"min_age_days": -1},
        )
        data = _parse(result)
        assert "error" in data

    def test_gc_rejects_invalid_min_access(self, tmp_store, project_id):
        result = dispatch(
            "quipu_gc",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"min_access_count": -5},
        )
        data = _parse(result)
        assert "error" in data


# ---------------------------------------------------------------------------
# TASK-025 — quipu_receipts
# ---------------------------------------------------------------------------


class TestQuipuReceipts:
    @staticmethod
    def _insert_oplog_entry(store, blinded, *, op="upsert", record_id="r1", ts="2024-01-01T00:00:00Z"):
        from quipu.sync.oplog_store import OplogStore
        from quipu.oplog.entry import OplogEntry

        oplog = OplogStore(store._conn)
        seq = oplog.next_sequence_no("test-client")
        entry = OplogEntry(
            entry_id=OplogEntry.compute_entry_id("test-client", seq),
            client_id="test-client",
            sequence_no=seq,
            op=op,
            record_id=record_id,
            blinded_project_id=blinded,
            ts=ts,
            payload=b"encrypted_payload_should_never_leak",
        )
        oplog.append_local(entry)

    def test_receipts_returns_hashed_entries(self, tmp_store, project_id, monkeypatch):
        import base64
        from quipu.crypto import blind_project_id

        test_key = bytes(range(32))
        blinded = blind_project_id(project_id, test_key)

        self._insert_oplog_entry(tmp_store, blinded, op="upsert", record_id="rec-1")
        self._insert_oplog_entry(tmp_store, blinded, op="invalidate", record_id="rec-2")

        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "receipts" in data
        receipts = data["receipts"]
        assert len(receipts) == 2
        for r in receipts:
            assert "op" in r
            assert "ts" in r
            assert "record_hash" in r
            assert r["op"] in ("write", "invalidate")
            # record_hash should be 64 hex chars (SHA-256)
            assert len(r["record_hash"]) == 64

    def test_receipts_no_content_leaked(self, tmp_store, project_id, monkeypatch):
        import base64
        from quipu.crypto import blind_project_id

        test_key = bytes(range(32))
        blinded = blind_project_id(project_id, test_key)

        self._insert_oplog_entry(tmp_store, blinded, op="upsert", record_id="secret-rec")

        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        receipt = data["receipts"][0]
        # NEVER expose content, payload, or plaintext data
        for key in receipt:
            assert key in ("op", "ts", "record_hash"), f"leaked field: {key}"
        # Verify payload is NOT in the receipt
        raw_text = result[0].text
        assert "encrypted_payload" not in raw_text

    def test_receipts_empty_no_entries(self, tmp_store, project_id, monkeypatch):
        test_key = bytes(range(32))
        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["receipts"] == []

    def test_receipts_op_filter_write(self, tmp_store, project_id, monkeypatch):
        from quipu.crypto import blind_project_id

        test_key = bytes(range(32))
        blinded = blind_project_id(project_id, test_key)

        self._insert_oplog_entry(tmp_store, blinded, op="upsert", record_id="w1")
        self._insert_oplog_entry(tmp_store, blinded, op="invalidate", record_id="i1")

        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"op": "write"},
        )
        data = _parse(result)
        receipts = data["receipts"]
        assert len(receipts) == 1
        assert receipts[0]["op"] == "write"

    def test_receipts_op_filter_invalidate(self, tmp_store, project_id, monkeypatch):
        from quipu.crypto import blind_project_id

        test_key = bytes(range(32))
        blinded = blind_project_id(project_id, test_key)

        self._insert_oplog_entry(tmp_store, blinded, op="upsert", record_id="w1")
        self._insert_oplog_entry(tmp_store, blinded, op="invalidate", record_id="i1")

        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"op": "invalidate"},
        )
        data = _parse(result)
        receipts = data["receipts"]
        assert len(receipts) == 1
        assert receipts[0]["op"] == "invalidate"

    def test_receipts_format_text(self, tmp_store, project_id, monkeypatch):
        from quipu.crypto import blind_project_id

        test_key = bytes(range(32))
        blinded = blind_project_id(project_id, test_key)

        self._insert_oplog_entry(tmp_store, blinded, op="upsert", record_id="rec-1", ts="2024-01-01T00:00:00Z")

        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"format": "text"},
        )
        data = _parse(result)
        assert "receipts" in data
        text = data["receipts"]
        assert isinstance(text, str)
        assert "\t" in text
        assert "write" in text
        assert "2024-01-01T00:00:00Z" in text

    def test_receipts_no_project_id_error(self, tmp_store):
        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data
        assert "project_id" in data["error"]

    def test_receipts_bad_limit(self, tmp_store, project_id):
        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"limit": 0},
        )
        data = _parse(result)
        assert "error" in data

    def test_receipts_bad_format(self, tmp_store, project_id):
        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"format": "xml"},
        )
        data = _parse(result)
        assert "error" in data

    def test_receipts_bad_op(self, tmp_store, project_id):
        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"op": "delete"},
        )
        data = _parse(result)
        assert "error" in data

    def test_receipts_record_hash_deterministic(self, tmp_store, project_id, monkeypatch):
        from quipu.crypto import blind_project_id

        test_key = bytes(range(32))
        blinded = blind_project_id(project_id, test_key)

        self._insert_oplog_entry(tmp_store, blinded, op="upsert", record_id="rc1")

        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result1 = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        result2 = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        hash1 = _parse(result1)["receipts"][0]["record_hash"]
        hash2 = _parse(result2)["receipts"][0]["record_hash"]
        assert hash1 == hash2

    def test_receipts_with_limit(self, tmp_store, project_id, monkeypatch):
        from quipu.crypto import blind_project_id

        test_key = bytes(range(32))
        blinded = blind_project_id(project_id, test_key)

        for i in range(5):
            self._insert_oplog_entry(tmp_store, blinded, op="upsert", record_id=f"r{i}", ts=f"2024-01-01T00:00:0{i}Z")

        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", lambda pid: test_key)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"limit": 2},
        )
        data = _parse(result)
        assert len(data["receipts"]) == 2

    def test_receipts_key_unavailable_error(self, tmp_store, project_id, monkeypatch):
        def _raise(*args, **kwargs):
            raise RuntimeError("no key")
        monkeypatch.setattr("quipu.keystore._backend.get_or_derive_key", _raise)

        result = dispatch(
            "quipu_receipts",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data
        assert "key" in data["error"].lower() or "unavailable" in data["error"].lower()


# ---------------------------------------------------------------------------
# TASK-022 — quipu_graph
# ---------------------------------------------------------------------------


class TestQuipuGraph:
    def test_graph_exact_atom_id_returns_subgraph(self, tmp_store, project_id, fake_engine):
        a1 = tmp_store.insert(content="root note", project_id=project_id)
        a2 = tmp_store.insert(content="linked note", project_id=project_id)
        tmp_store.insert_edge(
            from_atom_id=a1.id, to_atom_id=a2.id, edge_type="depends_on",
            project_id=project_id,
        )

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": a1.id},
        )
        data = _parse(result)
        assert "nodes" in data
        assert "edges" in data
        node_ids = {n["id"] for n in data["nodes"]}
        assert a1.id in node_ids
        assert a2.id in node_ids
        assert len(data["edges"]) >= 1

    def test_graph_search_term_finds_roots(self, tmp_store, project_id, fake_engine):
        a1 = tmp_store.insert(content="python programming", project_id=project_id)
        a2 = tmp_store.insert(content="java programming", project_id=project_id)
        tmp_store.insert_edge(
            from_atom_id=a1.id, to_atom_id=a2.id, edge_type="depends_on",
            project_id=project_id,
        )

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": "python", "top_k": 2},
        )
        data = _parse(result)
        assert "nodes" in data
        assert "edges" in data
        node_ids = {n["id"] for n in data["nodes"]}
        assert a1.id in node_ids

    def test_graph_missing_entity_returns_error(self, tmp_store, project_id):
        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_graph_empty_entity_returns_error(self, tmp_store, project_id):
        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": ""},
        )
        data = _parse(result)
        assert "error" in data

    def test_graph_requires_project_id(self, tmp_store):
        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=None,
            arguments={"entity": "some_id"},
        )
        data = _parse(result)
        assert "error" in data

    def test_graph_respects_max_depth(self, tmp_store, project_id, fake_engine):
        a1 = tmp_store.insert(content="root", project_id=project_id)
        a2 = tmp_store.insert(content="mid", project_id=project_id)
        a3 = tmp_store.insert(content="leaf", project_id=project_id)
        tmp_store.insert_edge(
            from_atom_id=a1.id, to_atom_id=a2.id, edge_type="depends_on",
            project_id=project_id,
        )
        tmp_store.insert_edge(
            from_atom_id=a2.id, to_atom_id=a3.id, edge_type="depends_on",
            project_id=project_id,
        )

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": a1.id, "max_depth": 1},
        )
        data = _parse(result)
        node_ids = {n["id"] for n in data["nodes"]}
        assert a1.id in node_ids
        assert a2.id in node_ids
        assert a3.id not in node_ids

    def test_graph_edge_types_filter(self, tmp_store, project_id, fake_engine):
        a1 = tmp_store.insert(content="root", project_id=project_id)
        a2 = tmp_store.insert(content="dep", project_id=project_id)
        a3 = tmp_store.insert(content="blk", project_id=project_id)
        tmp_store.insert_edge(
            from_atom_id=a1.id, to_atom_id=a2.id, edge_type="depends_on",
            project_id=project_id,
        )
        tmp_store.insert_edge(
            from_atom_id=a1.id, to_atom_id=a3.id, edge_type="blocks",
            project_id=project_id,
        )

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": a1.id, "edge_types": ["depends_on"]},
        )
        data = _parse(result)
        node_ids = {n["id"] for n in data["nodes"]}
        assert a2.id in node_ids
        assert a3.id not in node_ids

    def test_graph_no_edges_returns_root_only(self, tmp_store, project_id, fake_engine):
        a1 = tmp_store.insert(content="lonely atom", project_id=project_id)

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": a1.id},
        )
        data = _parse(result)
        assert len(data["nodes"]) == 1
        assert data["nodes"][0]["id"] == a1.id
        assert data["edges"] == []

    def test_graph_nonexistent_atom_id_searches(self, tmp_store, project_id, fake_engine):
        """When entity is not a valid atom_id, fall back to search."""
        a1 = tmp_store.insert(content="test atom", project_id=project_id)

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": "nonexistent_atom_id_1234"},
        )
        # Should not error — falls back to search (may return empty)
        data = _parse(result)
        assert "nodes" in data
        assert "edges" in data

    def test_graph_union_dedup_across_roots(self, tmp_store, project_id, fake_engine):
        """When multiple search roots share connected atoms, nodes deduplicated."""
        a1 = tmp_store.insert(content="comp A", project_id=project_id)
        a2 = tmp_store.insert(content="comp B", project_id=project_id)
        a3 = tmp_store.insert(content="shared dep", project_id=project_id)
        tmp_store.insert_edge(
            from_atom_id=a1.id, to_atom_id=a3.id, edge_type="depends_on",
            project_id=project_id,
        )
        tmp_store.insert_edge(
            from_atom_id=a2.id, to_atom_id=a3.id, edge_type="depends_on",
            project_id=project_id,
        )

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": "comp", "top_k": 3},
        )
        data = _parse(result)
        node_ids = {n["id"] for n in data["nodes"]}
        # a3 should appear only once
        count_a3 = sum(1 for n in data["nodes"] if n["id"] == a3.id)
        assert count_a3 == 1

    def test_graph_no_embedding_in_nodes(self, tmp_store, project_id, fake_engine):
        a1 = tmp_store.insert(content="clean node", project_id=project_id)

        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": a1.id},
        )
        data = _parse(result)
        for node in data["nodes"]:
            assert "embedding" not in node

    def test_graph_invalid_max_depth(self, tmp_store, project_id):
        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": "x", "max_depth": 0},
        )
        data = _parse(result)
        assert "error" in data

    def test_graph_invalid_edge_types(self, tmp_store, project_id):
        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": "x", "edge_types": "not-a-list"},
        )
        data = _parse(result)
        assert "error" in data

    def test_graph_invalid_top_k(self, tmp_store, project_id):
        result = dispatch(
            "quipu_graph",
            store=tmp_store,
            default_project_id=project_id,
            arguments={"entity": "x", "top_k": 0},
        )
        data = _parse(result)
        assert "error" in data
