"""Unit tests for quipu.crypto._kdf (derive_key, ARGON2_PARAMS)."""

from __future__ import annotations

import os
import secrets

import pytest

from quipu.crypto._kdf import ARGON2_PARAMS, derive_key
from quipu.crypto.errors import KdfError


# ---------------------------------------------------------------------------
# ARGON2_PARAMS constant values
# ---------------------------------------------------------------------------

def test_argon2_params_time_cost():
    assert ARGON2_PARAMS["time_cost"] == 3


def test_argon2_params_memory_cost():
    assert ARGON2_PARAMS["memory_cost"] == 65536


def test_argon2_params_parallelism():
    assert ARGON2_PARAMS["parallelism"] == 4


def test_argon2_params_hash_len():
    assert ARGON2_PARAMS["hash_len"] == 32


# ---------------------------------------------------------------------------
# derive_key — determinism
# ---------------------------------------------------------------------------

def test_derive_key_returns_32_bytes():
    salt = os.urandom(16)
    passphrase = secrets.token_hex(16)
    key = derive_key(passphrase, salt)
    assert len(key) == 32


def test_derive_key_same_inputs_same_output():
    salt = os.urandom(16)
    passphrase = secrets.token_hex(16)
    key1 = derive_key(passphrase, salt)
    key2 = derive_key(passphrase, salt)
    assert key1 == key2


def test_derive_key_different_salt_different_output():
    passphrase = secrets.token_hex(16)
    salt1 = os.urandom(16)
    salt2 = os.urandom(16)
    assert derive_key(passphrase, salt1) != derive_key(passphrase, salt2)


def test_derive_key_different_passphrase_different_output():
    salt = os.urandom(16)
    key1 = derive_key(secrets.token_hex(16), salt)
    key2 = derive_key(secrets.token_hex(16), salt)
    assert key1 != key2


# ---------------------------------------------------------------------------
# derive_key — error paths
# ---------------------------------------------------------------------------

def test_derive_key_empty_passphrase_raises_value_error():
    with pytest.raises(ValueError, match="passphrase must not be empty"):
        derive_key("", os.urandom(16))


def test_derive_key_salt_shorter_than_16_raises_value_error():
    with pytest.raises(ValueError, match="at least 16 bytes"):
        derive_key(secrets.token_hex(16), os.urandom(15))


def test_derive_key_exactly_16_byte_salt_accepted():
    """Salt at the floor (16 bytes) must NOT raise."""
    key = derive_key(secrets.token_hex(16), os.urandom(16))
    assert len(key) == 32


def test_derive_key_8_byte_salt_raises_value_error():
    """Explicit regression: the developer-fix raised the floor from 8 to 16."""
    with pytest.raises(ValueError, match="at least 16 bytes"):
        derive_key(secrets.token_hex(16), os.urandom(8))
