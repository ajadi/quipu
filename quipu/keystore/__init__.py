"""quipu.keystore — OS keystore integration for Quipu crypto keys.

Public API:
    from quipu.keystore import (
        KeyStore, KeyringBackend, InMemoryBackend,
        get_or_derive_key, get_or_create_salt,
        KeystoreUnavailable,
    )
"""

from quipu.keystore._backend import (
    InMemoryBackend,
    KeyringBackend,
    KeyStore,
    get_or_derive_key,
)
from quipu.keystore._salt import get_or_create_salt
from quipu.keystore.errors import KeystoreUnavailable

__all__ = [
    "KeyStore",
    "KeyringBackend",
    "InMemoryBackend",
    "get_or_derive_key",
    "get_or_create_salt",
    "KeystoreUnavailable",
]
