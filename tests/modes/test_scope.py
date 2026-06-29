"""Tests for quipu.modes.scope: resolve_scope, filtered_atoms, filtered_search_results."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.modes.scope import (
    VALID_SCOPES,
    resolve_scope,
    global_db_path,
    project_db_path,
    filtered_atoms,
    filtered_search_results,
)


# ---------------------------------------------------------------------------
# resolve_scope
# ---------------------------------------------------------------------------


class TestResolveScope:
    def test_none_defaults_to_project(self):
        assert resolve_scope(None) == "project"

    def test_project_passthrough(self):
        assert resolve_scope("project") == "project"

    def test_global_passthrough(self):
        assert resolve_scope("global") == "global"

    def test_all_passthrough(self):
        assert resolve_scope("all") == "all"

    def test_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="scope must be one of"):
            resolve_scope("other")


# ---------------------------------------------------------------------------
# filtered_atoms — scope=project
# ---------------------------------------------------------------------------


class TestFilteredAtomsProject:
    def test_returns_only_project_records(self, tmp_store, project_id):
        tmp_store.insert(content="p1", project_id=project_id, scope="project")
        tmp_store.insert(content="p2", project_id=project_id, scope="project")
        tmp_store.insert(content="other", project_id="other_proj", scope="project")

        atoms = filtered_atoms("project", project_id, project_store=tmp_store)
        contents = {a.content for a in atoms}
        assert contents == {"p1", "p2"}

    def test_excludes_invalidated_by_default(self, tmp_store, project_id):
        a = tmp_store.insert(content="valid", project_id=project_id, scope="project")
        b = tmp_store.insert(content="invalid", project_id=project_id, scope="project")
        tmp_store.update_invalidated(b.id, True)

        atoms = filtered_atoms(
            "project", project_id,
            project_store=tmp_store,
            include_invalidated=False,
        )
        assert all(not a.invalidated for a in atoms)
        assert any(a.content == "valid" for a in atoms)

    def test_limit_applied(self, tmp_store, project_id):
        for i in range(5):
            tmp_store.insert(content=f"item{i}", project_id=project_id, scope="project")

        atoms = filtered_atoms("project", project_id, project_store=tmp_store, limit=2)
        assert len(atoms) == 2

    def test_none_scope_is_project(self, tmp_store, project_id):
        tmp_store.insert(content="x", project_id=project_id, scope="project")
        atoms_none = filtered_atoms(None, project_id, project_store=tmp_store)
        atoms_proj = filtered_atoms("project", project_id, project_store=tmp_store)
        assert len(atoms_none) == len(atoms_proj)

    def test_injected_store_not_closed(self, tmp_store, project_id):
        """Store must remain open after call (caller owns lifetime)."""
        filtered_atoms("project", project_id, project_store=tmp_store)
        # If store was closed, this would raise
        tmp_store.list_by_project(project_id)


# ---------------------------------------------------------------------------
# filtered_atoms — scope=global
# ---------------------------------------------------------------------------


class TestFilteredAtomsGlobal:
    def test_returns_global_scope_records(self, tmp_store, project_id):
        tmp_store.insert(content="g1", project_id=project_id, scope="global")
        tmp_store.insert(content="p1", project_id=project_id, scope="project")

        atoms = filtered_atoms("global", project_id, project_store=tmp_store)
        contents = {a.content for a in atoms}
        assert "g1" in contents
        assert "p1" not in contents

    def test_uses_global_store_when_provided(self, tmp_store, tmp_store2, project_id):
        """global_store is used as source for global scope."""
        tmp_store2.insert(content="g-in-global-store", project_id=project_id, scope="global")
        tmp_store.insert(content="p-in-project", project_id=project_id, scope="project")

        atoms = filtered_atoms(
            "global", project_id,
            project_store=tmp_store,
            global_store=tmp_store2,
        )
        contents = {a.content for a in atoms}
        assert "g-in-global-store" in contents
        assert "p-in-project" not in contents

    def test_global_store_not_closed(self, tmp_store, tmp_store2, project_id):
        filtered_atoms(
            "global", project_id,
            project_store=tmp_store,
            global_store=tmp_store2,
        )
        # Both stores remain open
        tmp_store.list_by_project(project_id)
        tmp_store2.list_by_project(project_id)


# ---------------------------------------------------------------------------
# filtered_atoms — scope=all
# ---------------------------------------------------------------------------


class TestFilteredAtomsAll:
    def test_merges_project_and_global(self, tmp_store, project_id):
        tmp_store.insert(content="proj", project_id=project_id, scope="project")
        tmp_store.insert(content="glob", project_id=project_id, scope="global")

        atoms = filtered_atoms("all", project_id, project_store=tmp_store)
        contents = {a.content for a in atoms}
        assert "proj" in contents
        assert "glob" in contents

    def test_deduplicates_by_id(self, tmp_store, tmp_store2, project_id):
        """If same atom id appears in both stores, include it only once."""
        atom = tmp_store.insert(
            content="shared", project_id=project_id, scope="global",
            id="shared_id",
        )
        # Insert same id into second store
        tmp_store2.insert(
            content="shared", project_id=project_id, scope="global",
            id="shared_id",
        )

        atoms = filtered_atoms(
            "all", project_id,
            project_store=tmp_store,
            global_store=tmp_store2,
        )
        ids = [a.id for a in atoms]
        assert ids.count("shared_id") == 1

    def test_limit_applied_after_merge(self, tmp_store, project_id):
        for i in range(3):
            tmp_store.insert(content=f"p{i}", project_id=project_id, scope="project")
        for i in range(3):
            tmp_store.insert(content=f"g{i}", project_id=project_id, scope="global")

        atoms = filtered_atoms("all", project_id, project_store=tmp_store, limit=4)
        assert len(atoms) == 4

    def test_sorted_created_at_desc(self, tmp_store, project_id):
        tmp_store.insert(content="proj", project_id=project_id, scope="project")
        tmp_store.insert(content="glob", project_id=project_id, scope="global")

        atoms = filtered_atoms("all", project_id, project_store=tmp_store)
        dates = [a.created_at for a in atoms]
        assert dates == sorted(dates, reverse=True)


# ---------------------------------------------------------------------------
# filtered_search_results — scope tests
# ---------------------------------------------------------------------------


class TestFilteredSearchResults:
    def test_project_scope_uses_project_store(self, tmp_store, project_id, fake_engine):
        # R0 = exact match; insert content that exactly equals the query.
        tmp_store.insert(content="hello", project_id=project_id, scope="project")

        results = filtered_search_results(
            "project", project_id, "hello",
            project_store=tmp_store,
            tier="R0",
        )
        assert len(results) >= 1
        assert all(r.atom.project_id == project_id for r in results)

    def test_global_scope_uses_global_store(self, tmp_store, tmp_store2, project_id, fake_engine):
        tmp_store2.insert(content="global result", project_id=project_id, scope="global")

        results = filtered_search_results(
            "global", project_id, "global result",
            project_store=tmp_store,
            global_store=tmp_store2,
            tier="R0",
        )
        assert any(r.atom.content == "global result" for r in results)

    def test_all_scope_merges_stores(self, tmp_store, tmp_store2, project_id, fake_engine):
        """scope='all' returns atoms from both stores; global leg filtered to scope=='global'.

        R0 (exact match) is used so atoms are returned without embeddings.
        Both stores contain an atom whose content equals the query so both
        legs contribute matches.  A project-scoped atom in global_store must
        NOT surface (bug-fix regression for the global-leg scope leak).
        """
        # project_store: project-scoped atom that matches the query
        tmp_store.insert(content="target", project_id=project_id, scope="project")
        # global_store: global-scoped atom that matches + a project-scoped atom
        # that also matches but must be suppressed by the scope filter.
        tmp_store2.insert(content="target", project_id=project_id, scope="global", id="global_target")
        tmp_store2.insert(content="target", project_id=project_id, scope="project", id="proj_in_global")

        results = filtered_search_results(
            "all", project_id, "target",
            project_store=tmp_store,
            global_store=tmp_store2,
            tier="R0",
            top_k=10,
        )
        ids = {r.atom.id for r in results}
        scopes = {r.atom.scope for r in results}

        # global-scoped atom from global_store must appear
        assert "global_target" in ids
        # project-scoped atom from global_store must NOT appear (bug-fix regression)
        assert "proj_in_global" not in ids
        # all returned results have valid scopes
        assert scopes <= {"project", "global"}

    def test_global_scope_excludes_project_scoped_atoms(self, tmp_store, tmp_store2, project_id, fake_engine):
        """scope='global' must not return project-scoped atoms from the global store (bug-fix regression).

        Uses R0 (exact match) — atoms are returned without embeddings.
        """
        tmp_store2.insert(content="real-global", project_id=project_id, scope="global")
        tmp_store2.insert(content="real-global", project_id=project_id, scope="project", id="wrong_scope_id")

        results = filtered_search_results(
            "global", project_id, "real-global",
            project_store=tmp_store,
            global_store=tmp_store2,
            tier="R0",
            top_k=10,
        )
        ids = {r.atom.id for r in results}
        assert "wrong_scope_id" not in ids
        assert all(r.atom.scope == "global" for r in results)

    def test_all_scope_sorted_by_score_desc(self, tmp_store, project_id, fake_engine):
        """R1 results from a single store should be sorted by score DESC."""
        tmp_store.insert(content="alpha", project_id=project_id, scope="project")
        tmp_store.insert(content="beta", project_id=project_id, scope="project")

        results = filtered_search_results(
            "project", project_id, "alpha",
            project_store=tmp_store,
            tier="R1",
            top_k=10,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_all_scope_dedup_by_atom_id(self, tmp_store, tmp_store2, project_id, fake_engine):
        """Same atom id from both stores appears only once in results."""
        tmp_store.insert(
            content="shared", project_id=project_id, scope="project",
            id="dup_id",
        )
        tmp_store2.insert(
            content="shared", project_id=project_id, scope="global",
            id="dup_id",
        )

        results = filtered_search_results(
            "all", project_id, "shared",
            project_store=tmp_store,
            global_store=tmp_store2,
            tier="R0",
        )
        ids = [r.atom.id for r in results]
        assert ids.count("dup_id") == 1

    def test_stores_not_closed_after_search(self, tmp_store, project_id, fake_engine):
        tmp_store.insert(content="data", project_id=project_id, scope="project")
        filtered_search_results(
            "project", project_id, "data",
            project_store=tmp_store,
            tier="R0",
        )
        # Store still usable
        tmp_store.list_by_project(project_id)
