"""Tests for quipu.config: get_project_id, get_mode, get_project_root, resolve_mode_db_path."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Ensure repo root is on path
_repo_root = Path(__file__).resolve().parents[1]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.config import (
    get_mode,
    get_project_id,
    get_project_root,
    resolve_mode_db_path,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all QUIPU_* vars before each test to avoid cross-test pollution."""
    for var in ("QUIPU_MODE", "QUIPU_PROJECT_ROOT", "QUIPU_DB_PATH"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# get_project_id
# ---------------------------------------------------------------------------


class TestGetProjectId:
    def test_same_path_twice_same_id(self, tmp_path):
        a = get_project_id(str(tmp_path))
        b = get_project_id(str(tmp_path))
        assert a == b

    def test_returns_16_hex_chars(self, tmp_path):
        pid = get_project_id(str(tmp_path))
        assert len(pid) == 16
        assert all(c in "0123456789abcdef" for c in pid)

    def test_different_paths_different_id(self, tmp_path):
        path_a = tmp_path / "alpha"
        path_a.mkdir()
        path_b = tmp_path / "beta"
        path_b.mkdir()
        assert get_project_id(str(path_a)) != get_project_id(str(path_b))

    def test_trailing_slash_same_id(self, tmp_path):
        without = get_project_id(str(tmp_path))
        with_slash = get_project_id(str(tmp_path) + "/")
        assert without == with_slash

    @pytest.mark.skipif(os.name != "nt", reason="Windows-only case normalization")
    def test_windows_case_insensitive(self, tmp_path):
        lower = get_project_id(str(tmp_path).lower())
        upper = get_project_id(str(tmp_path).upper())
        assert lower == upper

    def test_none_uses_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        from_none = get_project_id(None)
        from_cwd = get_project_id(str(tmp_path))
        assert from_none == from_cwd


# ---------------------------------------------------------------------------
# get_mode
# ---------------------------------------------------------------------------


class TestGetMode:
    def test_default_is_project(self):
        assert get_mode() == "project"

    def test_project_explicit(self, monkeypatch):
        monkeypatch.setenv("QUIPU_MODE", "project")
        assert get_mode() == "project"

    def test_global_explicit(self, monkeypatch):
        monkeypatch.setenv("QUIPU_MODE", "global")
        assert get_mode() == "global"

    def test_unrecognized_falls_back_to_project(self, monkeypatch):
        monkeypatch.setenv("QUIPU_MODE", "other")
        assert get_mode() == "project"

    def test_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("QUIPU_MODE", "GLOBAL")
        assert get_mode() == "global"


# ---------------------------------------------------------------------------
# get_project_root
# ---------------------------------------------------------------------------


class TestGetProjectRoot:
    def test_default_is_cwd(self, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        assert get_project_root() == tmp_path.resolve()

    def test_respects_quipu_project_root(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        assert get_project_root() == tmp_path.resolve()

    def test_returns_resolved_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        result = get_project_root()
        assert isinstance(result, Path)
        assert result.is_absolute()


# ---------------------------------------------------------------------------
# resolve_mode_db_path
# ---------------------------------------------------------------------------


class TestResolveModeDbPath:
    def test_unset_returns_home_quipu_db(self):
        # QUIPU_MODE not set -> compat path
        result = resolve_mode_db_path()
        assert result == Path.home() / ".quipu" / "quipu.db"

    def test_global_mode_returns_global_db(self, monkeypatch):
        monkeypatch.setenv("QUIPU_MODE", "global")
        result = resolve_mode_db_path()
        assert result == Path.home() / ".quipu" / "global.db"

    def test_project_mode_returns_project_local_db(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        result = resolve_mode_db_path()
        assert result == tmp_path.resolve() / ".quipu" / "quipu.db"

    def test_project_mode_no_root_uses_cwd(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.chdir(tmp_path)
        result = resolve_mode_db_path()
        assert result == tmp_path.resolve() / ".quipu" / "quipu.db"
