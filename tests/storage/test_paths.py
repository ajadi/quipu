"""Unit tests for quipu.storage.paths.resolve_db_path.

Covers path-resolution precedence: explicit arg > QUIPU_DB_PATH env > default.
Also covers parent-directory auto-creation.
"""

import os
from pathlib import Path

import pytest

from quipu.storage.paths import resolve_db_path


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Remove QUIPU_DB_PATH from environment so tests start clean."""
    monkeypatch.delenv("QUIPU_DB_PATH", raising=False)


class TestResolveDbPath:
    def test_explicit_override_is_returned_as_path(self, tmp_path):
        target = tmp_path / "explicit.db"
        result = resolve_db_path(str(target))
        assert result == target

    def test_explicit_override_takes_precedence_over_env(self, tmp_path, monkeypatch):
        target = tmp_path / "explicit.db"
        env_path = tmp_path / "env.db"
        monkeypatch.setenv("QUIPU_DB_PATH", str(env_path))
        result = resolve_db_path(str(target))
        assert result == target

    def test_env_var_used_when_no_override(self, tmp_path, monkeypatch):
        env_path = tmp_path / "from_env.db"
        monkeypatch.setenv("QUIPU_DB_PATH", str(env_path))
        result = resolve_db_path()
        assert result == env_path

    def test_default_path_is_under_home_when_no_override_no_env(self, tmp_path, monkeypatch):
        monkeypatch.delenv("QUIPU_DB_PATH", raising=False)
        result = resolve_db_path()
        home = Path.home()
        assert result.is_relative_to(home)

    def test_default_path_ends_with_quipu_db(self, monkeypatch):
        monkeypatch.delenv("QUIPU_DB_PATH", raising=False)
        result = resolve_db_path()
        assert result.name == "quipu.db"

    def test_returns_path_object(self, tmp_path):
        result = resolve_db_path(str(tmp_path / "x.db"))
        assert isinstance(result, Path)

    def test_parent_directory_created_if_absent(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "quipu.db"
        assert not nested.parent.exists()
        resolve_db_path(str(nested))
        assert nested.parent.exists()

    def test_existing_parent_directory_no_error(self, tmp_path):
        target = tmp_path / "quipu.db"
        # tmp_path already exists — calling again must not raise
        result = resolve_db_path(str(target))
        assert result == target

    def test_none_override_falls_through_to_env(self, tmp_path, monkeypatch):
        env_path = tmp_path / "via_env.db"
        monkeypatch.setenv("QUIPU_DB_PATH", str(env_path))
        result = resolve_db_path(None)
        assert result == env_path
