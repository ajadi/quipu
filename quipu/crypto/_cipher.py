"""quipu.crypto._cipher — AES-256-GCM encrypt/decrypt with versioned blob layout.

Blob layout (per record):
    | 1B version | 1B key_version | 12B nonce | AESGCM(ciphertext + 16B tag) |

The outer length-prefix frame is the OPLOG/sync layer's concern (TASK-012).
key_version byte is reserved for future key-rotation; always 0 in V1.
Nonce: 12 bytes os.urandom per record. 2^32 record limit per key documented;
rotation deferred to TASK-012 (key_version byte reserved for that purpose).
"""

from __future__ import annotations

import os
import struct

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

from quipu.crypto.errors import DecryptError

BLOB_VERSION: int = 1
_KEY_VERSION: int = 0  # reserved; increment on key rotation (TASK-012+)
_NONCE_LEN: int = 12
_HEADER_LEN: int = 2  # version(1) + key_version(1)


def encrypt_record(
    plaintext: bytes,
    key: bytes,
    *,
    aad: bytes | None = None,
) -> bytes:
    """Encrypt *plaintext* with AES-256-GCM and return a versioned blob.

    A fresh 12-byte nonce is generated from os.urandom on every call.

    Args:
        plaintext: Raw bytes to encrypt.
        key: 32-byte AES-256 key (from derive_key or QUIPU_KEY escape hatch).
        aad: Optional additional authenticated data. Recommended production
             binding: blind_project_id(project_id, key).encode(). Prevents
             hub cross-project replay. Default None (no AAD binding).

    Returns:
        Versioned blob: | 1B version | 1B key_version | 12B nonce | ct+tag |
    """
    if len(key) != 32:
        raise ValueError(f"key must be 32 bytes, got {len(key)}")

    nonce = os.urandom(_NONCE_LEN)
    aesgcm = AESGCM(key)
    ct_and_tag = aesgcm.encrypt(nonce, plaintext, aad)
    return serialize_blob(nonce, ct_and_tag)


def decrypt_record(
    blob: bytes,
    key: bytes,
    *,
    aad: bytes | None = None,
) -> bytes:
    """Decrypt a versioned blob produced by encrypt_record.

    Args:
        blob: Versioned blob from encrypt_record.
        key: 32-byte AES-256 key.
        aad: Must match the AAD supplied at encrypt time (including None).

    Returns:
        Original plaintext bytes.

    Raises:
        DecryptError: On tag mismatch, wrong key, AAD mismatch, or malformed blob.
    """
    try:
        nonce, ct_and_tag = deserialize_blob(blob)
    except (ValueError, struct.error) as exc:
        raise DecryptError(f"malformed blob: {exc}") from exc

    if len(key) != 32:
        raise DecryptError("key must be 32 bytes")

    aesgcm = AESGCM(key)
    try:
        return aesgcm.decrypt(nonce, ct_and_tag, aad)
    except InvalidTag as exc:
        raise DecryptError("decryption failed: tag mismatch, wrong key, or AAD mismatch") from exc
    except Exception as exc:
        raise DecryptError("decryption failed: internal error") from exc


def serialize_blob(nonce: bytes, ct_and_tag: bytes) -> bytes:
    """Pack nonce + ciphertext+tag into a versioned blob.

    Layout: | 1B version=1 | 1B key_version=0 | 12B nonce | ct+tag |
    """
    if len(nonce) != _NONCE_LEN:
        raise ValueError(f"nonce must be {_NONCE_LEN} bytes, got {len(nonce)}")
    header = struct.pack("BB", BLOB_VERSION, _KEY_VERSION)
    return header + nonce + ct_and_tag


def deserialize_blob(blob: bytes) -> tuple[bytes, bytes]:
    """Unpack a versioned blob into (nonce, ct_and_tag).

    Raises:
        ValueError: If blob is too short, version unsupported, or malformed.
    """
    min_len = _HEADER_LEN + _NONCE_LEN + 16  # 16 = GCM tag minimum
    if len(blob) < min_len:
        raise ValueError(
            f"blob too short: {len(blob)} bytes (minimum {min_len})"
        )

    version, _key_ver = struct.unpack_from("BB", blob, 0)
    if version != BLOB_VERSION:
        raise ValueError(f"unsupported blob version {version!r} (expected {BLOB_VERSION})")

    nonce = blob[_HEADER_LEN : _HEADER_LEN + _NONCE_LEN]
    ct_and_tag = blob[_HEADER_LEN + _NONCE_LEN :]
    return nonce, ct_and_tag
