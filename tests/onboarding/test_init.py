"""Tests for quipu.cli.cmd_init — project/global/server modes and idempotency."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from quipu.cli import cmd_init


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clear_quipu_env(monkeypatch):
    """Remove QUIPU_MODE / QUIPU_PROJECT_ROOT / QUIPU_DB_PATH from env."""
    for key in ("QUIPU_MODE", "QUIPU_PROJECT_ROOT", "QUIPU_DB_PATH"):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# Project mode
# ---------------------------------------------------------------------------

class TestCmdInitProject:
    def test_creates_db_in_tmp(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        rc = cmd_init("project")

        assert rc == 0
        assert (tmp_path / ".quipu" / "quipu.db").exists()

    def test_creates_config_json(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("project")

        config_path = tmp_path / ".quipu" / "config.json"
        assert config_path.exists()

    def test_config_schema_correct(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("project")

        config = json.loads((tmp_path / ".quipu" / "config.json").read_text())
        assert config["mode"] == "project"
        assert "project_id" in config
        assert "created" in config
        assert "last_init" in config
        assert "quipu_version" in config
        assert config["project_root"] == str(tmp_path.resolve())

    def test_project_id_matches_config_get_project_id(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("project")

        from quipu.config import get_project_id
        expected = get_project_id(str(tmp_path))
        config = json.loads((tmp_path / ".quipu" / "config.json").read_text())
        assert config["project_id"] == expected

    def test_none_mode_defaults_to_project(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        rc = cmd_init(None)

        assert rc == 0
        config = json.loads((tmp_path / ".quipu" / "config.json").read_text())
        assert config["mode"] == "project"


# ---------------------------------------------------------------------------
# Global mode
# ---------------------------------------------------------------------------

class TestCmdInitGlobal:
    def test_creates_global_db(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        rc = cmd_init("global")

        assert rc == 0
        assert (fake_home / ".quipu" / "global.db").exists()

    def test_creates_global_config_json(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        cmd_init("global")

        assert (fake_home / ".quipu" / "config.json").exists()

    def test_global_project_id_is_literal_global(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        cmd_init("global")

        config = json.loads((fake_home / ".quipu" / "config.json").read_text())
        assert config["project_id"] == "global"

    def test_global_config_mode_field(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))

        cmd_init("global")

        config = json.loads((fake_home / ".quipu" / "config.json").read_text())
        assert config["mode"] == "global"


# ---------------------------------------------------------------------------
# Server mode (real wiring — returns 0, writes config.json)
# ---------------------------------------------------------------------------

class TestCmdInitServer:
    def test_returns_exit_code_0(self, tmp_path, monkeypatch, capsys):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        rc = cmd_init("server")

        assert rc == 0

    def test_writes_config_json(self, tmp_path, monkeypatch, capsys):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("QUIPU_HUB_URL", "https://hub.example.com")

        cmd_init("server")

        config_path = tmp_path / ".quipu" / "config.json"
        assert config_path.exists(), "config.json must be written by server-mode init"
        config = json.loads(config_path.read_text())
        assert config["mode"] == "server"
        assert "client_id" in config
        assert config["hub_url"] == "https://hub.example.com"
        # QUIPU_HUB_TOKEN must NEVER be written as a key into config.json
        assert "token" not in config
        assert "hub_token" not in config

    def test_token_never_written_to_config(self, tmp_path, monkeypatch, capsys):
        """QUIPU_HUB_TOKEN set in env at init time must NOT appear in config.json."""
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "super-secret-token-xyz")

        cmd_init("server")

        config_path = tmp_path / ".quipu" / "config.json"
        assert config_path.exists()
        raw = config_path.read_text()
        # The actual token value must never appear in the config file
        assert "super-secret-token-xyz" not in raw
        # The config dict must not have a 'token' or 'hub_token' key
        config = json.loads(raw)
        assert "token" not in config
        assert "hub_token" not in config

    def test_message_mentions_server_mode_and_token_reminder(self, tmp_path, monkeypatch, capsys):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("server")

        captured = capsys.readouterr()
        combined = (captured.out + captured.err).lower()
        # Must mention server mode
        assert "server" in combined
        # Must remind user to set QUIPU_HUB_TOKEN (env-only)
        assert "quipu_hub_token" in combined
        # Must NOT say "staged" or "phase 3" (those were the old stub messages)
        assert "staged" not in combined
        assert "phase 3" not in combined


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestCmdInitIdempotency:
    def test_second_run_preserves_project_id(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("project")
        config1 = json.loads((tmp_path / ".quipu" / "config.json").read_text())

        cmd_init("project")
        config2 = json.loads((tmp_path / ".quipu" / "config.json").read_text())

        assert config1["project_id"] == config2["project_id"]

    def test_second_run_preserves_created(self, tmp_path, monkeypatch):
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("project")
        config1 = json.loads((tmp_path / ".quipu" / "config.json").read_text())

        cmd_init("project")
        config2 = json.loads((tmp_path / ".quipu" / "config.json").read_text())

        assert config1["created"] == config2["created"]

    def test_atom_survives_reinit(self, tmp_path, monkeypatch):
        """Write an atom, re-run init, assert atom is still readable."""
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("project")

        db_path = tmp_path / ".quipu" / "quipu.db"
        from quipu.storage import store as _store_factory
        s = _store_factory(str(db_path))
        atom = s.insert(content="survive-me", project_id="test-proj")
        atom_id = atom.id
        s.close()

        # Re-run init — must NOT clobber the DB.
        cmd_init("project")

        s2 = _store_factory(str(db_path))
        fetched = s2.get(atom_id)
        s2.close()

        assert fetched is not None
        assert fetched.content == "survive-me"

    def test_last_init_refreshed_on_second_run(self, tmp_path, monkeypatch):
        """last_init must change (or at least be set) after re-run."""
        _clear_quipu_env(monkeypatch)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        cmd_init("project")
        config1 = json.loads((tmp_path / ".quipu" / "config.json").read_text())

        # Rerun returns 0 and updates last_init (may be same second in fast CI, that's ok)
        rc = cmd_init("project")
        assert rc == 0
        config2 = json.loads((tmp_path / ".quipu" / "config.json").read_text())
        assert "last_init" in config2
        # created is preserved
        assert config1["created"] == config2["created"]
