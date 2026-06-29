"""Unit tests for quipu.crypto._cipher (encrypt_record, decrypt_record, serialize/deserialize_blob)."""

from __future__ import annotations

import os

import pytest

from quipu.crypto._cipher import (
    BLOB_VERSION,
    decrypt_record,
    deserialize_blob,
    encrypt_record,
    serialize_blob,
)
from quipu.crypto.errors import DecryptError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_key() -> bytes:
    return os.urandom(32)


# ---------------------------------------------------------------------------
# encrypt_record / decrypt_record — round-trip
# ---------------------------------------------------------------------------

def test_round_trip_basic_plaintext():
    key = _random_key()
    plaintext = b"hello quipu crypto"
    blob = encrypt_record(plaintext, key)
    assert decrypt_record(blob, key) == plaintext


def test_round_trip_empty_plaintext():
    key = _random_key()
    blob = encrypt_record(b"", key)
    assert decrypt_record(blob, key) == b""


def test_round_trip_large_plaintext():
    key = _random_key()
    plaintext = os.urandom(1024 * 64)  # 64 KiB
    blob = encrypt_record(plaintext, key)
    assert decrypt_record(blob, key) == plaintext


def test_round_trip_with_aad():
    key = _random_key()
    plaintext = b"aad-bound record"
    aad = b"quipu:project:abc123"
    blob = encrypt_record(plaintext, key, aad=aad)
    assert decrypt_record(blob, key, aad=aad) == plaintext


# ---------------------------------------------------------------------------
# encrypt_record — nonce randomness
# ---------------------------------------------------------------------------

def test_two_encrypts_of_same_plaintext_produce_different_blobs():
    key = _random_key()
    plaintext = b"same plaintext"
    blob1 = encrypt_record(plaintext, key)
    blob2 = encrypt_record(plaintext, key)
    assert blob1 != blob2


# ---------------------------------------------------------------------------
# decrypt_record — tamper / wrong-key / aad-mismatch → DecryptError
# ---------------------------------------------------------------------------

def test_tamper_ciphertext_raises_decrypt_error():
    key = _random_key()
    blob = bytearray(encrypt_record(b"sensitive", key))
    # Flip a byte in the ciphertext body (after 14-byte header)
    blob[20] ^= 0xFF
    with pytest.raises(DecryptError):
        decrypt_record(bytes(blob), key)


def test_wrong_key_raises_decrypt_error():
    key = _random_key()
    wrong_key = _random_key()
    blob = encrypt_record(b"secret", key)
    with pytest.raises(DecryptError):
        decrypt_record(blob, wrong_key)


def test_aad_mismatch_encrypt_with_aad_decrypt_without_raises():
    key = _random_key()
    blob = encrypt_record(b"payload", key, aad=b"x")
    with pytest.raises(DecryptError):
        decrypt_record(blob, key, aad=None)


def test_aad_mismatch_different_aad_raises():
    key = _random_key()
    blob = encrypt_record(b"payload", key, aad=b"x")
    with pytest.raises(DecryptError):
        decrypt_record(blob, key, aad=b"y")


# ---------------------------------------------------------------------------
# encrypt_record — wrong-key-length ValueError
# ---------------------------------------------------------------------------

def test_encrypt_non_32_byte_key_raises_value_error():
    with pytest.raises(ValueError, match="32 bytes"):
        encrypt_record(b"data", os.urandom(16))


# ---------------------------------------------------------------------------
# decrypt_record on truncated / too-short blob → DecryptError (not IndexError)
# ---------------------------------------------------------------------------

def test_decrypt_empty_blob_raises_decrypt_error():
    key = _random_key()
    with pytest.raises(DecryptError):
        decrypt_record(b"", key)


def test_decrypt_truncated_blob_raises_decrypt_error():
    key = _random_key()
    blob = encrypt_record(b"data", key)
    truncated = blob[:10]  # less than the 14-byte minimum header+nonce
    with pytest.raises(DecryptError):
        decrypt_record(truncated, key)


# ---------------------------------------------------------------------------
# serialize_blob / deserialize_blob — round-trip and layout
# ---------------------------------------------------------------------------

def test_serialize_deserialize_round_trip():
    nonce = os.urandom(12)
    ct_and_tag = os.urandom(32)
    blob = serialize_blob(nonce, ct_and_tag)
    out_nonce, out_ct = deserialize_blob(blob)
    assert out_nonce == nonce
    assert out_ct == ct_and_tag


def test_blob_layout_version_byte():
    key = _random_key()
    blob = encrypt_record(b"layout check", key)
    assert blob[0] == BLOB_VERSION  # byte 0 = version = 1


def test_blob_layout_key_version_byte():
    key = _random_key()
    blob = encrypt_record(b"layout check", key)
    assert blob[1] == 0  # byte 1 = key_version = 0 (reserved)


def test_blob_layout_nonce_is_bytes_2_to_14():
    """Nonce occupies bytes [2:14] (12 bytes)."""
    key = _random_key()
    blob = encrypt_record(b"layout check", key)
    nonce_slice = blob[2:14]
    assert len(nonce_slice) == 12


def test_deserialize_blob_too_short_raises_value_error():
    with pytest.raises(ValueError):
        deserialize_blob(b"\x01\x00" + b"\x00" * 5)  # only 7 bytes, min is 30
