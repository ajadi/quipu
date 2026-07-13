"""Tests for sync_now: never_configured, offline degrade, key failure, ok path."""

from __future__ import annotations

import logging
import os

import pytest
from tests._semantic import TEST_EMBED_DIM

from quipu.sync.client import sync_now, _set_last_sync_status, reset_last_sync_status
from quipu.sync.errors import SyncUnavailableError, SyncProtocolError
from quipu.sync.transport import InMemoryTransport


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clear_hub_env(monkeypatch):
    """Remove all hub-related env vars."""
    for var in ("QUIPU_HUB_URL", "QUIPU_HUB_TOKEN", "QUIPU_HUB_CA",
                "QUIPU_KEY", "QUIPU_CLIENT_ID"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------
# never_configured: no hub env vars
# ---------------------------------------------------------------------------


class TestNeverConfigured:
    def test_no_env_returns_never_configured(self, tmp_store, monkeypatch):
        _clear_hub_env(monkeypatch)
        result = sync_now("proj-x", store=tmp_store)
        assert result.status == "never_configured"

    def test_no_env_no_network_calls(self, tmp_store, monkeypatch):
        """With no hub config, sync_now must not call any network function."""
        _clear_hub_env(monkeypatch)
        import quipu.sync.client as client_mod
        calls = []
        original = client_mod.HttpTransport.__init__
        def _spy_init(self, *args, **kwargs):
            calls.append(args)
            original(self, *args, **kwargs)
        monkeypatch.setattr(client_mod.HttpTransport, "__init__", _spy_init)
        sync_now("proj-x", store=tmp_store)
        assert calls == [], "HttpTransport should not be instantiated if hub not configured"

    def test_no_token_returns_never_configured(self, tmp_store, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        # No token -> None
        result = sync_now("proj-x", store=tmp_store)
        assert result.status == "never_configured"

    def test_only_token_no_url_returns_never_configured(self, tmp_store, monkeypatch):
        _clear_hub_env(monkeypatch)
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        result = sync_now("proj-x", store=tmp_store)
        assert result.status == "never_configured"


# ---------------------------------------------------------------------------
# Offline degrade: transport raises SyncUnavailableError
# ---------------------------------------------------------------------------


class TestOfflineDegrade:
    def _patch_hub_and_transport(self, monkeypatch, tmp_store, exc):
        """Set up hub env + monkeypatch sync_now's internals to raise exc."""
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        monkeypatch.setenv("QUIPU_KEY", "A" * 44)  # valid base64 32-byte key

        import quipu.sync.client as client_mod
        def _bad_request(self, method, path, body, params=None):
            raise exc
        monkeypatch.setattr(client_mod.HttpTransport, "_request", _bad_request)

    def test_unavailable_returns_offline(self, tmp_store, monkeypatch):
        self._patch_hub_and_transport(monkeypatch, tmp_store, SyncUnavailableError("down"))
        result = sync_now("proj-x", store=tmp_store)
        assert result.status == "offline"

    def test_offline_does_not_raise(self, tmp_store, monkeypatch):
        self._patch_hub_and_transport(monkeypatch, tmp_store, SyncUnavailableError("down"))
        # Must not raise
        sync_now("proj-x", store=tmp_store)

    def test_offline_local_store_untouched(self, tmp_store, monkeypatch, fake_engine):
        """After offline sync, local write/read still works."""
        self._patch_hub_and_transport(monkeypatch, tmp_store, SyncUnavailableError("down"))
        sync_now("proj-x", store=tmp_store)
        # Local store ops still work
        atom = tmp_store.insert(content="local data", project_id="proj-x", metadata={})
        assert tmp_store.get(atom.id) is not None

    def test_protocol_error_returns_offline(self, tmp_store, monkeypatch):
        self._patch_hub_and_transport(monkeypatch, tmp_store, SyncProtocolError("bad"))
        result = sync_now("proj-x", store=tmp_store)
        assert result.status == "offline"


# ---------------------------------------------------------------------------
# Key derivation failure -> offline, never prompt
# ---------------------------------------------------------------------------


class TestKeyDerivationFailure:
    def test_key_failure_returns_offline(self, tmp_store, monkeypatch):
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        # Remove QUIPU_KEY so it tries keyring; patch get_or_derive_key to raise
        monkeypatch.delenv("QUIPU_KEY", raising=False)
        import quipu.sync.client as client_mod
        import quipu.keystore._backend as kb
        monkeypatch.setattr(kb, "get_or_derive_key", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("no key")))
        result = sync_now("proj-x", store=tmp_store)
        assert result.status == "offline"

    def test_key_failure_no_raise(self, tmp_store, monkeypatch):
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        monkeypatch.delenv("QUIPU_KEY", raising=False)
        import quipu.keystore._backend as kb
        monkeypatch.setattr(kb, "get_or_derive_key", lambda *a, **kw: (_ for _ in ()).throw(ValueError("prompt?")))
        # Must not raise or prompt
        sync_now("proj-x", store=tmp_store)


# ---------------------------------------------------------------------------
# Ok path with InMemoryTransport-style fake
# ---------------------------------------------------------------------------


class TestOkPath:
    def test_ok_with_in_memory_transport(self, tmp_store, monkeypatch):
        """patch the internal transport with InMemoryTransport to get ok status."""
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        # Use a valid 32-byte base64 key
        import base64
        test_key = base64.b64encode(bytes(range(32))).decode()
        monkeypatch.setenv("QUIPU_KEY", test_key)
        monkeypatch.setenv("QUIPU_CLIENT_ID", "test-client-ok")

        mem = InMemoryTransport()
        import quipu.sync.client as client_mod

        def _fake_init(self, base_url, token, *, timeout=30.0, verify=None):
            self._mem = mem
            self._base_url = base_url
            self._token = token
            self._timeout = timeout
            self._ssl_ctx = None

        def _fake_push(self, bpid, entries):
            return mem.push(bpid, entries)

        def _fake_pull(self, bpid, cursor):
            return mem.pull(bpid, cursor)

        monkeypatch.setattr(client_mod.HttpTransport, "__init__", _fake_init)
        monkeypatch.setattr(client_mod.HttpTransport, "push", _fake_push)
        monkeypatch.setattr(client_mod.HttpTransport, "pull", _fake_pull)

        result = sync_now("proj-x", store=tmp_store, directions=("pull", "push"))
        assert result.status == "ok"

    def test_directions_pull_only(self, tmp_store, monkeypatch):
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "tok")
        import base64
        test_key = base64.b64encode(bytes(range(32))).decode()
        monkeypatch.setenv("QUIPU_KEY", test_key)
        monkeypatch.setenv("QUIPU_CLIENT_ID", "test-client-dir")

        mem = InMemoryTransport()
        import quipu.sync.client as client_mod

        def _fake_init(self, base_url, token, *, timeout=30.0, verify=None):
            self._mem = mem
            self._base_url = base_url
            self._token = token
            self._timeout = timeout
            self._ssl_ctx = None

        monkeypatch.setattr(client_mod.HttpTransport, "__init__", _fake_init)
        monkeypatch.setattr(client_mod.HttpTransport, "push",
                            lambda self, bpid, entries: mem.push(bpid, entries))
        monkeypatch.setattr(client_mod.HttpTransport, "pull",
                            lambda self, bpid, cursor: mem.pull(bpid, cursor))

        result = sync_now("proj-x", store=tmp_store, directions=("pull",))
        assert result.status == "ok"
        assert result.pushed == 0  # push not requested


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

import pytest

from quipu.storage import store as open_store
from quipu.embeddings.engine import _reset, set_engine, _Engine


class _FakeTokenizerEncoding:
    def __init__(self, ids, mask):
        self.ids = ids
        self.attention_mask = mask


class _FakeTokenizer:
    def __init__(self, seq_len: int = 8) -> None:
        self._seq_len = seq_len

    def encode_batch(self, texts):
        return [
            _FakeTokenizerEncoding(ids=[1] * self._seq_len, mask=[1] * self._seq_len)
            for _ in texts
        ]


class _N:
    def __init__(self, name: str) -> None:
        self.name = name
        self.type = "tensor(int64)"


class _FakeSession:
    def __init__(self, value: float = 1.0, seq_len: int = 8) -> None:
        self._value = value
        self._seq_len = seq_len

    def get_inputs(self):
        return [_N("input_ids"), _N("attention_mask")]

    def get_outputs(self):
        return [_N("sentence_embedding")]

    def run(self, output_names, feeds):
        import numpy as np
        n = feeds["input_ids"].shape[0]
        arr = np.full((n, TEST_EMBED_DIM), self._value, dtype=np.float32)
        return [arr]


@pytest.fixture(autouse=True)
def reset_embedding_engine():
    yield
    _reset()


@pytest.fixture()
def fake_engine(semantic_model):
    engine = _Engine(session=_FakeSession(value=1.0), tokenizer=_FakeTokenizer())
    set_engine(engine)
    return engine


@pytest.fixture()
def tmp_store(tmp_path):
    db_path = str(tmp_path / "t.db")
    s = open_store(db_path)
    yield s
    s.close()


@pytest.fixture(autouse=True)
def reset_sync_status():
    """Ensure _last_sync_status is reset between tests to prevent order-dependence."""
    reset_last_sync_status()
    yield
    reset_last_sync_status()
