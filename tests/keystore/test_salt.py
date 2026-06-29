"""Unit tests for quipu.keystore._salt (get_or_create_salt)."""

from __future__ import annotations

from quipu.keystore._backend import InMemoryBackend, KeyStore
from quipu.keystore._salt import get_or_create_salt


def _in_memory_store() -> KeyStore:
    return KeyStore(backend=InMemoryBackend())


# ---------------------------------------------------------------------------
# get_or_create_salt — shape
# ---------------------------------------------------------------------------

def test_get_or_create_salt_returns_16_bytes():
    store = _in_memory_store()
    salt = get_or_create_salt(store)
    assert len(salt) == 16


# ---------------------------------------------------------------------------
# get_or_create_salt — idempotency
# ---------------------------------------------------------------------------

def test_get_or_create_salt_idempotent_same_store_same_namespace():
    store = _in_memory_store()
    salt1 = get_or_create_salt(store, namespace="ember")
    salt2 = get_or_create_salt(store, namespace="ember")
    assert salt1 == salt2


def test_get_or_create_salt_returns_bytes_type():
    store = _in_memory_store()
    salt = get_or_create_salt(store)
    assert isinstance(salt, bytes)


# ---------------------------------------------------------------------------
# get_or_create_salt — namespace isolation
# ---------------------------------------------------------------------------

def test_get_or_create_salt_different_namespaces_different_salts():
    store = _in_memory_store()
    salt_a = get_or_create_salt(store, namespace="ns-a")
    salt_b = get_or_create_salt(store, namespace="ns-b")
    assert salt_a != salt_b


# ---------------------------------------------------------------------------
# get_or_create_salt — persistence via the injected store
# ---------------------------------------------------------------------------

def test_get_or_create_salt_persisted_via_store():
    """Salt must be stored in the backend so a second KeyStore wrapping the same
    backend returns the same salt (simulates process restart with shared backend)."""
    backend = InMemoryBackend()
    store1 = KeyStore(backend=backend)
    salt1 = get_or_create_salt(store1, namespace="persist-test")

    # Second KeyStore wrapping the SAME backend — simulates reload
    store2 = KeyStore(backend=backend)
    salt2 = get_or_create_salt(store2, namespace="persist-test")

    assert salt1 == salt2


def test_get_or_create_salt_default_namespace_is_ember():
    """Explicit call and call without namespace= produce the same result."""
    store = _in_memory_store()
    salt_default = get_or_create_salt(store)
    salt_explicit = get_or_create_salt(store, namespace="ember")
    assert salt_default == salt_explicit
