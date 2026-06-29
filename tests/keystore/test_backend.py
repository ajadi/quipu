"""Unit tests for quipu.keystore._backend (KeyStore, InMemoryBackend, get_or_derive_key, fallback)."""

from __future__ import annotations

import base64
import os
import secrets

import pytest

from quipu.keystore._backend import (
    InMemoryBackend,
    KeyStore,
    get_or_derive_key,
)
from quipu.keystore.errors import KeystoreUnavailable


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_key() -> bytes:
    return os.urandom(32)


def _b64_key(key: bytes) -> str:
    return base64.b64encode(key).decode()


def _in_memory_store() -> KeyStore:
    return KeyStore(backend=InMemoryBackend())


# ---------------------------------------------------------------------------
# KeyStore — round-trip with InMemoryBackend
# ---------------------------------------------------------------------------

def test_keystore_set_then_get_returns_identical_bytes():
    store = _in_memory_store()
    value = _random_key()
    store.set("key:proj1", value)
    assert store.get("key:proj1") == value


def test_keystore_get_absent_username_returns_none():
    store = _in_memory_store()
    assert store.get("key:does-not-exist") is None


def test_keystore_get_corrupt_base64_returns_none():
    """Corrupt base64 stored directly in the backend must surface as None, not an exception."""
    backend = InMemoryBackend()
    # Write invalid base64 directly, bypassing KeyStore.set()
    backend.set_password("quipu-crypto", "key:proj", "not-valid-base64!!!")
    store = KeyStore(backend=backend)
    assert store.get("key:proj") is None


def test_keystore_is_fallback_false_when_explicit_backend():
    store = _in_memory_store()
    assert store.is_fallback is False


# ---------------------------------------------------------------------------
# get_or_derive_key — QUIPU_KEY escape hatch
# ---------------------------------------------------------------------------

def test_quipu_key_env_returns_key_directly(monkeypatch):
    key = _random_key()
    monkeypatch.setenv("QUIPU_KEY", _b64_key(key))
    result = get_or_derive_key("proj-1", store=_in_memory_store(), passphrase=secrets.token_hex(16))
    assert result == key


def test_quipu_key_env_never_calls_derive_key(monkeypatch):
    """When QUIPU_KEY is set the key is returned immediately; derive_key must never run."""
    key = _random_key()
    monkeypatch.setenv("QUIPU_KEY", _b64_key(key))

    call_count = []

    import quipu.crypto._kdf as _kdf_mod

    def fake_derive(passphrase, salt):  # noqa: ARG001
        call_count.append(1)
        return os.urandom(32)

    # Patch at the source so the lazy import inside get_or_derive_key picks it up.
    monkeypatch.setattr(_kdf_mod, "derive_key", fake_derive)

    get_or_derive_key("proj-1", store=_in_memory_store(), passphrase=secrets.token_hex(16))
    assert call_count == [], "derive_key must not be called when QUIPU_KEY is set"


def test_quipu_key_invalid_base64_raises_runtime_error(monkeypatch):
    monkeypatch.setenv("QUIPU_KEY", "not-valid-base64!!!")
    with pytest.raises(RuntimeError, match="not valid base64"):
        get_or_derive_key("proj-1", store=_in_memory_store(), passphrase=secrets.token_hex(16))


def test_quipu_key_wrong_length_raises_runtime_error(monkeypatch):
    short_key = os.urandom(16)  # 16 bytes, not 32
    monkeypatch.setenv("QUIPU_KEY", _b64_key(short_key))
    with pytest.raises(RuntimeError, match="32 bytes"):
        get_or_derive_key("proj-1", store=_in_memory_store(), passphrase=secrets.token_hex(16))


# ---------------------------------------------------------------------------
# get_or_derive_key — cache miss then hit
# ---------------------------------------------------------------------------

def test_cache_miss_then_hit_does_not_call_derive_twice(monkeypatch):
    """Second call must use the cached key — derive_key called exactly once.

    get_or_derive_key imports derive_key with `from quipu.crypto._kdf import derive_key`
    inside the function body, so we patch it at the source module (quipu.crypto._kdf).
    """
    store = _in_memory_store()
    passphrase = secrets.token_hex(16)
    call_count = []

    import quipu.crypto._kdf as _kdf_mod

    original_derive = _kdf_mod.derive_key

    def counting_derive(pp, salt):
        call_count.append(1)
        return original_derive(pp, salt)

    monkeypatch.setattr(_kdf_mod, "derive_key", counting_derive)

    # First call: cache miss → derive
    key1 = get_or_derive_key("proj-cache", store=store, passphrase=passphrase)
    # Second call: cache hit → no derive
    key2 = get_or_derive_key("proj-cache", store=store, passphrase=passphrase)

    assert key1 == key2
    assert len(call_count) == 1, f"derive_key called {len(call_count)} times (expected 1)"


def test_get_or_derive_key_passphrase_env(monkeypatch):
    """QUIPU_PASSPHRASE env var is used when no explicit passphrase arg."""
    passphrase = secrets.token_hex(16)
    monkeypatch.setenv("QUIPU_PASSPHRASE", passphrase)
    store = _in_memory_store()
    key = get_or_derive_key("proj-env", store=store)
    assert len(key) == 32


def test_get_or_derive_key_explicit_passphrase():
    """Explicit passphrase arg must produce a 32-byte key without env vars."""
    store = _in_memory_store()
    key = get_or_derive_key("proj-explicit", store=store, passphrase=secrets.token_hex(16))
    assert len(key) == 32


# ---------------------------------------------------------------------------
# Fallback / no-op backend detection
# ---------------------------------------------------------------------------

class _FakeFailKeyring:
    """Simulates keyring.backends.fail.Keyring (silent no-op backend)."""

    def get_password(self, service, username):
        return None

    def set_password(self, service, username, password):
        pass


_FakeFailKeyring.__module__ = "keyring.backends.fail"
_FakeFailKeyring.__qualname__ = "Keyring"


class _FakeNullKeyring:
    """Simulates keyring.backends.null.Keyring."""

    def get_password(self, service, username):
        return None

    def set_password(self, service, username, password):
        pass


_FakeNullKeyring.__module__ = "keyring.backends.null"


def test_resolve_real_backend_fail_keyring_raises_keystore_unavailable(monkeypatch):
    """A fail.Keyring backend must cause _resolve_real_backend to raise KeystoreUnavailable."""
    import quipu.keystore._backend as _mod

    fake_instance = _FakeFailKeyring()

    class _FakeKeyring:
        def get_keyring(self):
            return fake_instance

    monkeypatch.setattr(_mod, "keyring", _FakeKeyring(), raising=False)

    # Patch the import inside _resolve_real_backend
    import sys
    import types

    fake_keyring_mod = types.ModuleType("keyring")
    fake_keyring_mod.get_keyring = lambda: fake_instance  # type: ignore[attr-defined]
    fake_keyring_errors = types.ModuleType("keyring.errors")

    old_keyring = sys.modules.get("keyring")
    old_keyring_errors = sys.modules.get("keyring.errors")
    sys.modules["keyring"] = fake_keyring_mod
    sys.modules["keyring.errors"] = fake_keyring_errors

    try:
        from quipu.keystore._backend import _resolve_real_backend
        with pytest.raises(KeystoreUnavailable):
            _resolve_real_backend()
    finally:
        if old_keyring is not None:
            sys.modules["keyring"] = old_keyring
        else:
            sys.modules.pop("keyring", None)
        if old_keyring_errors is not None:
            sys.modules["keyring.errors"] = old_keyring_errors
        else:
            sys.modules.pop("keyring.errors", None)


# ---------------------------------------------------------------------------
# Regression: real OS backends named "Keyring" must NOT be rejected (GH-fix)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module_name", [
    "keyring.backends.SecretService",
    "keyring.backends.libsecret",
    "keyring.backends.macOS",
])
def test_resolve_real_backend_os_keyring_named_Keyring_is_not_rejected(monkeypatch, module_name):
    """Real OS backends whose class __name__ == 'Keyring' must pass through.

    Regression for the bug where _resolve_real_backend rejected ANY backend
    with __name__ == 'Keyring' under keyring.backends.*, which incorrectly
    blocked SecretService, libsecret, and macOS keyrings.
    """
    import sys
    import types

    from quipu.keystore._backend import _resolve_real_backend

    # Build a stub class with the same __name__ and __module__ as a real OS backend.
    stub_cls = type("Keyring", (), {
        "get_password": lambda self, s, u: None,
        "set_password": lambda self, s, u, p: None,
    })
    stub_cls.__module__ = module_name
    stub_instance = stub_cls()

    fake_keyring_mod = types.ModuleType("keyring")
    fake_keyring_mod.get_keyring = lambda: stub_instance  # type: ignore[attr-defined]
    fake_keyring_errors = types.ModuleType("keyring.errors")

    old_keyring = sys.modules.get("keyring")
    old_keyring_errors = sys.modules.get("keyring.errors")
    sys.modules["keyring"] = fake_keyring_mod
    sys.modules["keyring.errors"] = fake_keyring_errors

    try:
        # Must NOT raise KeystoreUnavailable and must return the stub backend.
        result = _resolve_real_backend()
        assert result is stub_instance
    finally:
        if old_keyring is not None:
            sys.modules["keyring"] = old_keyring
        else:
            sys.modules.pop("keyring", None)
        if old_keyring_errors is not None:
            sys.modules["keyring.errors"] = old_keyring_errors
        else:
            sys.modules.pop("keyring.errors", None)


def test_keystore_none_backend_falls_back_to_in_memory_and_warns(monkeypatch, caplog):
    """KeyStore(backend=None) on a headless system uses InMemoryBackend + WARNING."""
    import quipu.keystore._backend as _mod
    from quipu.keystore._backend import _resolve_real_backend

    def _fail_resolve():
        raise KeystoreUnavailable("no keyring available in this environment")

    monkeypatch.setattr(_mod, "_resolve_real_backend", _fail_resolve)

    import logging
    with caplog.at_level(logging.WARNING, logger="quipu.keystore._backend"):
        store = KeyStore(backend=None)

    assert store.is_fallback is True
    assert isinstance(store._backend, InMemoryBackend)
    # Warning must mention the unavailability without leaking key/passphrase values
    warning_text = " ".join(r.message for r in caplog.records if r.levelno >= logging.WARNING)
    assert "keyring" in warning_text.lower() or "unavailable" in warning_text.lower() or "fallback" in warning_text.lower()
    assert "QUIPU_KEY" not in warning_text
    assert "QUIPU_PASSPHRASE" not in warning_text
