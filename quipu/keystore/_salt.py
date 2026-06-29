"""quipu.keystore._salt — per-project random salt lifecycle.

The salt is created once with os.urandom(16), persisted in the keystore,
and reused on every subsequent call. It is NEVER hardcoded.

The salt is stored in the keystore (OS keyring or InMemoryBackend fallback),
NOT on the hub, so the hub remains zero-knowledge even about the salt.
"""

from __future__ import annotations

import os

from quipu.keystore._backend import KeyStore

_SALT_LEN = 16


def get_or_create_salt(store: KeyStore, *, namespace: str = "ember") -> bytes:  # wire-value: salt namespace kept as "ember" — renaming orphans all existing keystore salts/keys
    """Return the persisted 16-byte random salt, creating it on first call.

    The salt username in the keystore is "salt:<namespace>" (default "salt:ember").
    Created once with os.urandom(16); subsequent calls return the same bytes.

    Args:
        store: KeyStore instance for persistence.
        namespace: Logical scope for the salt (default "ember").
                   Use distinct namespaces for isolated test environments.

    Returns:
        16 random bytes (stable across calls for the same store + namespace).
    """
    username = f"salt:{namespace}"
    existing = store.get(username)
    if existing is not None and len(existing) == _SALT_LEN:
        return existing

    salt = os.urandom(_SALT_LEN)
    store.set(username, salt)
    return salt
