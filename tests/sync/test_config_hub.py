"""Tests for get_hub_config / get_client_id in quipu/config.py."""

from __future__ import annotations

import json
import os

import pytest

from quipu.config import get_hub_config, get_client_id, HubConfig


def _clear_hub_env(monkeypatch):
    for var in ("QUIPU_HUB_URL", "QUIPU_HUB_TOKEN", "QUIPU_HUB_CA", "QUIPU_CLIENT_ID",
                "QUIPU_MODE", "QUIPU_PROJECT_ROOT"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# get_hub_config
# ---------------------------------------------------------------------------


class TestGetHubConfig:
    def test_returns_none_when_no_url_and_no_token(self, monkeypatch):
        _clear_hub_env(monkeypatch)
        assert get_hub_config() is None

    def test_returns_none_when_url_but_no_token(self, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        assert get_hub_config() is None

    def test_returns_none_when_token_but_no_url(self, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        assert get_hub_config() is None

    def test_returns_config_when_both_set(self, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_URL", "https://hub.example.com")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "secret")
        cfg = get_hub_config()
        assert cfg is not None
        assert cfg.url == "https://hub.example.com"
        assert cfg.token == "secret"
        assert cfg.verify is None

    def test_verify_set_from_quipu_hub_ca(self, monkeypatch, tmp_path):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_URL", "https://hub.example.com")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        ca = str(tmp_path / "ca.crt")
        monkeypatch.setenv("QUIPU_HUB_CA", ca)
        cfg = get_hub_config()
        assert cfg is not None
        assert cfg.verify == ca

    def test_url_from_config_json(self, monkeypatch, tmp_path):
        _clear_hub_env(monkeypatch)
        # Write config.json with hub_url
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        config_dir = tmp_path / ".quipu"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"hub_url": "http://from-file"}), encoding="utf-8"
        )
        cfg = get_hub_config()
        assert cfg is not None
        assert cfg.url == "http://from-file"

    def test_env_url_overrides_config_json(self, monkeypatch, tmp_path):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("QUIPU_HUB_URL", "http://from-env")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        config_dir = tmp_path / ".quipu"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"hub_url": "http://from-file"}), encoding="utf-8"
        )
        cfg = get_hub_config()
        assert cfg is not None
        assert cfg.url == "http://from-env"

    def test_token_not_read_from_config_json(self, monkeypatch, tmp_path):
        """Token must come from env only — never from config.json."""
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        config_dir = tmp_path / ".quipu"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"hub_url": "http://hub", "hub_token": "token-on-disk"}),
            encoding="utf-8",
        )
        # No QUIPU_HUB_TOKEN env -> should be None (token not read from disk)
        cfg = get_hub_config()
        assert cfg is None


# ---------------------------------------------------------------------------
# get_client_id
# ---------------------------------------------------------------------------


class TestGetClientId:
    def test_returns_env_if_set(self, monkeypatch):
        monkeypatch.setenv("QUIPU_CLIENT_ID", "env-client-xyz")
        assert get_client_id() == "env-client-xyz"

    def test_reads_from_config_json(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUIPU_CLIENT_ID", raising=False)
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        config_dir = tmp_path / ".quipu"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"client_id": "from-file-id"}), encoding="utf-8"
        )
        assert get_client_id() == "from-file-id"

    def test_generates_and_persists_if_absent(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUIPU_CLIENT_ID", raising=False)
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        # No config.json
        cid1 = get_client_id()
        assert isinstance(cid1, str)
        assert len(cid1) == 32  # uuid4().hex

        # Second call should return the same persisted value
        cid2 = get_client_id()
        assert cid2 == cid1

    def test_persisted_to_config_json(self, monkeypatch, tmp_path):
        monkeypatch.delenv("QUIPU_CLIENT_ID", raising=False)
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        cid = get_client_id()
        config_path = tmp_path / ".quipu" / "config.json"
        assert config_path.exists()
        data = json.loads(config_path.read_text())
        assert data["client_id"] == cid

    def test_env_takes_precedence_over_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("QUIPU_CLIENT_ID", "env-wins")
        monkeypatch.setenv("QUIPU_MODE", "project")
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))
        config_dir = tmp_path / ".quipu"
        config_dir.mkdir()
        (config_dir / "config.json").write_text(
            json.dumps({"client_id": "file-value"}), encoding="utf-8"
        )
        assert get_client_id() == "env-wins"


# ---------------------------------------------------------------------------
# H1 — HubConfig.token must NOT appear in repr()
# ---------------------------------------------------------------------------


class TestHubConfigReprRedactsToken:
    def test_token_absent_from_repr(self):
        """H1 security: repr(HubConfig) must NOT expose the bearer token value."""
        cfg = HubConfig(url="https://hub.example.com", token="SECRET_BEARER_VALUE", verify=None)
        r = repr(cfg)
        assert "SECRET_BEARER_VALUE" not in r, (
            f"Token leaked in repr: {r!r}"
        )

    def test_url_still_visible_in_repr(self):
        """url field (non-secret) should still appear in repr for debuggability."""
        cfg = HubConfig(url="https://hub.example.com", token="TOPSECRET", verify=None)
        r = repr(cfg)
        assert "https://hub.example.com" in r

    def test_repr_does_not_contain_token_keyword_value(self):
        """Even if field name 'token' appears in repr, its value must not."""
        cfg = HubConfig(url="http://hub", token="MY_UNIQUE_SECRET_42", verify=None)
        r = repr(cfg)
        assert "MY_UNIQUE_SECRET_42" not in r
