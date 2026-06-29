"""Tests for quipu.storage.paths.resolve_db_path mode-routing (E6 additions).

Verifies acceptance criteria:
  - QUIPU_PROJECT_ROOT changes the project DB path.
  - QUIPU_MODE=global routes to global.db.
  - Explicit override and QUIPU_DB_PATH still win over mode routing.

NOTE: tests/storage/test_paths.py covers the original 9 cases (unchanged).
This file covers the NEW mode-routing branch exclusively.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.storage.paths import resolve_db_path


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Strip all QUIPU_* vars before each test."""
    for var in ("QUIPU_MODE", "QUIPU_PROJECT_ROOT", "QUIPU_DB_PATH"):
        monkeypatch.delenv(var, raising=False)


class TestModeRouting:
    def test_project_root_changes_db_path(self, tmp_path, monkeypatch):
        """AC: Changing QUIPU_PROJECT_ROOT changes the project DB path."""
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        result = resolve_db_path()
        assert result == tmp_path.resolve() / ".quipu" / "quipu.db"

    def test_global_mode_uses_global_db(self, monkeypatch):
        """QUIPU_MODE=global -> ~/.quipu/global.db."""
        monkeypatch.setenv("QUIPU_MODE", "global")
        result = resolve_db_path()
        assert result == Path.home() / ".quipu" / "global.db"

    def test_explicit_override_wins_over_mode(self, tmp_path, monkeypatch):
        """Explicit path arg wins regardless of QUIPU_MODE."""
        monkeypatch.setenv("QUIPU_MODE", "global")
        target = tmp_path / "override.db"
        result = resolve_db_path(str(target))
        assert result == target

    def test_quipu_db_path_wins_over_mode(self, tmp_path, monkeypatch):
        """QUIPU_DB_PATH env wins over QUIPU_MODE routing."""
        monkeypatch.setenv("QUIPU_MODE", "global")
        env_db = tmp_path / "env.db"
        monkeypatch.setenv("QUIPU_DB_PATH", str(env_db))
        result = resolve_db_path()
        assert result == env_db

    def test_unset_mode_compat_path(self, monkeypatch):
        """QUIPU_MODE unset -> ~/.quipu/quipu.db (compat, keeps old tests green)."""
        result = resolve_db_path()
        assert result == Path.home() / ".quipu" / "quipu.db"

    def test_parent_directory_created_for_project_mode(self, tmp_path, monkeypatch):
        """resolve_db_path creates parent dir even for mode-routed paths."""
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        result = resolve_db_path()
        assert result.parent.exists()
