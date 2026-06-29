"""quipu.keystore._backend — KeyStore facade, backend Protocol, and get_or_derive_key.

Fallback chain
--------------
1. Try keyring.get_keyring(). Accept it if it is NOT a fail.Keyring instance
   (silent no-op backend that Linux headless environments return).
2. On NoKeyringError, RuntimeError (D-Bus), or silent fail.Keyring -> fall back
   to InMemoryBackend, log a WARNING.
3. QUIPU_KEY escape hatch: if the env var is set (base64 32-byte key), return it
   directly without derive/prompt, regardless of backend state.

Security notes
--------------
- The DERIVED KEY is cached in the keystore, never the passphrase.
- QUIPU_PASSPHRASE and QUIPU_KEY are never logged (only existence is checked).
- InMemoryBackend is session-only (process lifetime).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Protocol, runtime_checkable

from quipu.keystore.errors import KeystoreUnavailable

log = logging.getLogger(__name__)

KEYRING_SERVICE = "quipu-crypto"
# keyring username conventions:
#   "key:<project_id>"   -> base64-encoded 32-byte derived key
#   "salt:<scope>"       -> base64-encoded 16-byte random salt


# ---------------------------------------------------------------------------
# Backend Protocol (injectable for tests)
# ---------------------------------------------------------------------------

@runtime_checkable
class KeyringBackend(Protocol):
    """Minimal keyring-compatible interface required by KeyStore."""

    def get_password(self, service: str, username: str) -> str | None:
        ...

    def set_password(self, service: str, username: str, password: str) -> None:
        ...


# ---------------------------------------------------------------------------
# In-memory backend (test injection + headless fallback)
# ---------------------------------------------------------------------------

class InMemoryBackend:
    """Session-only key store. No persistence across process restarts.

    Used:
      1. In tests via KeyStore(backend=InMemoryBackend()).
      2. At runtime when no usable OS keyring is found (fallback path).
    """

    def __init__(self) -> None:
        self._store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self._store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self._store[(service, username)] = password


# ---------------------------------------------------------------------------
# Real-keyring backend resolver
# ---------------------------------------------------------------------------

def _resolve_real_backend() -> KeyringBackend:
    """Attempt to obtain a real OS keyring backend.

    Raises KeystoreUnavailable if:
      - keyring raises NoKeyringError
      - keyring raises RuntimeError (D-Bus / Wayland unavailable)
      - the returned backend is a silent fail.Keyring (no-op)
    """
    try:
        import keyring
        import keyring.errors

        backend = keyring.get_keyring()

        # Reject silent no-op backends (keyring.backends.fail.Keyring /
        # keyring.backends.null.Keyring) by module path. Do NOT match on the
        # class name "Keyring" — real OS backends (SecretService/libsecret/macOS)
        # also use that class name and must pass through.
        backend_type = type(backend)
        module = (getattr(backend_type, "__module__", "") or "").lower()
        if "fail" in module or "null" in module:
            raise KeystoreUnavailable(
                f"keyring returned silent no-op backend: {backend_type!r}"
            )

        return backend  # type: ignore[return-value]

    except KeystoreUnavailable:
        raise
    except Exception as exc:
        # Covers: keyring.errors.NoKeyringError, RuntimeError (D-Bus),
        # ImportError, and any other keyring internal failure.
        raise KeystoreUnavailable(f"keyring unavailable: {exc}") from exc


# ---------------------------------------------------------------------------
# KeyStore facade
# ---------------------------------------------------------------------------

class KeyStore:
    """Thin facade over a KeyringBackend with base64 serialization.

    Values stored as base64 strings (keyring stores strings, not bytes).

    Args:
        backend: Explicit backend to use. Pass an InMemoryBackend for tests.
                 If None, resolves the real OS keyring with InMemoryBackend
                 fallback on failure.
    """

    def __init__(self, backend: KeyringBackend | None = None) -> None:
        if backend is not None:
            self._backend: KeyringBackend = backend
            self._fallback = False
        else:
            try:
                self._backend = _resolve_real_backend()
                self._fallback = False
            except KeystoreUnavailable as exc:
                log.warning(
                    "quipu-keystore: OS keyring unavailable (%s). "
                    "Falling back to in-memory session store. "
                    "Keys will NOT persist across restarts.",
                    exc,
                )
                self._backend = InMemoryBackend()
                self._fallback = True

    @property
    def is_fallback(self) -> bool:
        """True if using the in-memory fallback (no OS keyring)."""
        return self._fallback

    def get(self, username: str) -> bytes | None:
        """Retrieve bytes stored under *username* in KEYRING_SERVICE, or None."""
        raw = self._backend.get_password(KEYRING_SERVICE, username)
        if raw is None:
            return None
        try:
            return base64.b64decode(raw)
        except Exception:
            return None

    def set(self, username: str, value: bytes) -> None:
        """Store *value* bytes under *username* in KEYRING_SERVICE."""
        self._backend.set_password(KEYRING_SERVICE, username, base64.b64encode(value).decode())


# ---------------------------------------------------------------------------
# get_or_derive_key
# ---------------------------------------------------------------------------

def get_or_derive_key(
    project_id: str,
    *,
    store: KeyStore | None = None,
    passphrase: str | None = None,
) -> bytes:
    """Return the 32-byte AES-256 key for *project_id*, deriving if needed.

    Lookup order:
    1. QUIPU_KEY env var (base64 32-byte) -> return directly, no derive/prompt.
    2. KeyStore cache hit -> return cached DERIVED KEY.
    3. Cache miss:
       a. Passphrase from *passphrase* arg, else QUIPU_PASSPHRASE env, else prompt.
       b. Salt via get_or_create_salt(store).
       c. derive_key(passphrase, salt) -> cache -> return.

    Security:
      - Caches the DERIVED KEY, never the passphrase.
      - QUIPU_PASSPHRASE and QUIPU_KEY are never logged.

    Args:
        project_id: Project identifier (from quipu.config.get_project_id()).
        store: KeyStore instance. Defaults to KeyStore() (real OS keyring or fallback).
        passphrase: Explicit passphrase override (bypasses env/prompt). For tests.

    Returns:
        32 raw bytes suitable for AES-256-GCM.

    Raises:
        ValueError: If no passphrase can be obtained and no cached key exists.
        RuntimeError: If QUIPU_KEY is set but is not valid base64 or not 32 bytes.
    """
    # Escape hatch: QUIPU_KEY short-circuits everything (CI/headless environments).
    quipu_key_b64 = os.environ.get("QUIPU_KEY")
    if quipu_key_b64:
        try:
            key = base64.b64decode(quipu_key_b64)
        except Exception as exc:
            raise RuntimeError("QUIPU_KEY is not valid base64") from exc
        if len(key) != 32:
            raise RuntimeError(
                f"QUIPU_KEY must decode to exactly 32 bytes, got {len(key)}"
            )
        return key

    if store is None:
        store = KeyStore()

    cache_username = f"key:{project_id}"

    # Cache hit
    cached = store.get(cache_username)
    if cached is not None:
        if len(cached) == 32:
            return cached
        log.warning(
            "quipu-keystore: cached key for %s has invalid length %d; re-deriving",
            cache_username,
            len(cached),
        )

    # Cache miss — derive
    _passphrase = _resolve_passphrase(passphrase)

    # Import here to avoid circular imports (crypto -> keystore would be a cycle).
    from quipu.crypto._kdf import derive_key
    from quipu.keystore._salt import get_or_create_salt

    salt = get_or_create_salt(store)
    key = derive_key(_passphrase, salt)

    store.set(cache_username, key)
    return key


def _resolve_passphrase(explicit: str | None) -> str:
    """Resolve passphrase from explicit arg, QUIPU_PASSPHRASE env, or stdin prompt.

    QUIPU_PASSPHRASE is never logged.
    """
    if explicit is not None:
        return explicit
    env = os.environ.get("QUIPU_PASSPHRASE")
    if env:
        return env
    # Prompt as last resort (interactive sessions only).
    import getpass
    return getpass.getpass("Quipu passphrase: ")
