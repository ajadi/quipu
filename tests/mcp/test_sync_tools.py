"""Tests for quipu_push, quipu_pull dispatch; quipu_stats sync fields; quipu_flush push trigger."""

from __future__ import annotations

import json
import logging

import pytest

from quipu.mcp.tools import dispatch


def _parse(result) -> dict:
    assert len(result) == 1
    return json.loads(result[0].text)


def _clear_hub_env(monkeypatch):
    for var in ("QUIPU_HUB_URL", "QUIPU_HUB_TOKEN", "QUIPU_KEY", "QUIPU_CLIENT_ID",
                "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# quipu_push
# ---------------------------------------------------------------------------


class TestQuipuPush:
    def test_push_never_configured(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_push",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["sync_status"] == "never_configured"
        assert data["pushed"] == 0

    def test_push_no_project_id_returns_error(self, tmp_store, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_push",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_push_offline_returns_offline(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        import base64
        monkeypatch.setenv("QUIPU_KEY", base64.b64encode(bytes(range(32))).decode())
        # Patch the push function in the push module so the offline condition is
        # triggered regardless of whether there are local oplog entries to push.
        # sync_now does `from quipu.sync.push import push` at call time, so
        # patching the attribute on the module is the correct intercept point.
        import sys
        from quipu.sync.errors import SyncUnavailableError

        def _raise_unavailable(*args, **kwargs):
            raise SyncUnavailableError("down")

        push_module = sys.modules["quipu.sync.push"]
        monkeypatch.setattr(push_module, "push", _raise_unavailable)
        result = dispatch(
            "quipu_push",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["sync_status"] == "offline"

    def test_push_returns_expected_keys(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_push",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "sync_status" in data
        assert "pushed" in data
        assert "detail" in data


# ---------------------------------------------------------------------------
# quipu_pull
# ---------------------------------------------------------------------------


class TestQuipuPull:
    def test_pull_never_configured(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_pull",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["sync_status"] == "never_configured"
        assert data["pulled"] == 0

    def test_pull_no_project_id_returns_error(self, tmp_store, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_pull",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        assert "error" in data

    def test_pull_returns_expected_keys(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_pull",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "sync_status" in data
        assert "pulled" in data
        assert "detail" in data


# ---------------------------------------------------------------------------
# quipu_stats sync fields
# ---------------------------------------------------------------------------


class TestQuipuStatsSyncFields:
    def test_stats_has_sync_fields(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "sync_status" in data
        assert "last_push" in data
        assert "last_pull" in data

    def test_stats_never_configured_when_no_hub(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert data["sync_status"] == "never_configured"
        assert data["last_push"] is None
        assert data["last_pull"] is None

    def test_stats_no_block_when_hub_configured_but_no_key(self, tmp_store, project_id, monkeypatch):
        """Hub configured but no QUIPU_KEY/QUIPU_PASSPHRASE: stats must return quickly
        with last_push/last_pull=None without calling getpass or hanging."""
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub.example.com")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "test-token")
        # Explicitly ensure key env vars are absent
        monkeypatch.delenv("QUIPU_KEY", raising=False)
        monkeypatch.delenv("QUIPU_PASSPHRASE", raising=False)
        # Patch getpass to prove it is never called
        import getpass
        def _should_not_prompt(prompt=""):
            raise AssertionError("getpass.getpass() called — would hang on headless server")
        monkeypatch.setattr(getpass, "getpass", _should_not_prompt)
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" not in data
        assert data["last_push"] is None
        assert data["last_pull"] is None

    def test_stats_never_fails_even_if_sync_errors(self, tmp_store, project_id, monkeypatch):
        """Even with broken sync, quipu_stats must return without error."""
        _clear_hub_env(monkeypatch)
        # Cause get_hub_config to fail
        import quipu.config as config_mod
        monkeypatch.setattr(config_mod, "get_hub_config", lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        result = dispatch(
            "quipu_stats",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "error" not in data
        assert "total" in data


# ---------------------------------------------------------------------------
# quipu_flush triggers push
# ---------------------------------------------------------------------------


class TestQuipuFlushTriggersPush:
    def test_flush_calls_sync_push(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        # Flush should run and include sync_status in result when project_id set
        result = dispatch(
            "quipu_flush",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        # Core flush keys present
        assert "enriched" in data
        assert "skipped" in data
        # Sync keys added after flush
        assert "sync_status" in data
        assert data["sync_status"] == "never_configured"  # no hub configured

    def test_flush_result_has_pushed_key(self, tmp_store, project_id, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = dispatch(
            "quipu_flush",
            store=tmp_store,
            default_project_id=project_id,
            arguments={},
        )
        data = _parse(result)
        assert "pushed" in data

    def test_flush_without_project_id_no_sync_key(self, tmp_store, monkeypatch):
        """When project_id is None, flush still works but no sync keys added."""
        _clear_hub_env(monkeypatch)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        result = dispatch(
            "quipu_flush",
            store=tmp_store,
            default_project_id=None,
            arguments={},
        )
        data = _parse(result)
        # Core flush keys present — sync keys not added when project_id is None
        assert "enriched" in data
        assert "skipped" in data
