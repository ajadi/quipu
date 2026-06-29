"""Tests for quipu.oplog.codec — encrypt/decrypt round-trip, AAD binding, tamper,
wrong-key, cross-project rejection, and the outer length-prefix frame.
"""

import os

import pytest

from quipu.crypto import blind_project_id
from quipu.crypto.errors import DecryptError
from quipu.oplog.codec import (
    decode_entry,
    encode_entry,
    frame_blobs,
    unframe_blobs,
)

KEY_A = bytes(range(32))
KEY_B = bytes(range(32, 64))

OP = {
    "op": "upsert",
    "record_id": "rec-1",
    "ts": "2026-06-21T00:00:00Z",
    "content": "hello world",
    "type": "diary",
    "scope": "project",
    "metadata": {"k": "v"},
    "refs": [],
    "project_id": "proj-1",
}


def _blinded(project_id, key):
    return blind_project_id(project_id, key)


# ---------------------------------------------------------------------------
# encode -> decode round-trip
# ---------------------------------------------------------------------------

def test_encode_decode_roundtrip():
    bp = _blinded("proj-1", KEY_A)
    blob = encode_entry(OP, KEY_A, bp)
    out = decode_entry(blob, KEY_A, bp)
    assert out == OP


def test_aad_roundtrip_binds_blinded_pid():
    """Decrypt with the same blinded pid (AAD) succeeds."""
    bp = _blinded("proj-1", KEY_A)
    blob = encode_entry(OP, KEY_A, bp)
    # round-trips only because AAD matches
    assert decode_entry(blob, KEY_A, bp)["record_id"] == "rec-1"


# ---------------------------------------------------------------------------
# tamper / wrong-key / cross-project -> DecryptError
# ---------------------------------------------------------------------------

def test_tamper_payload_byte_raises():
    bp = _blinded("proj-1", KEY_A)
    blob = bytearray(encode_entry(OP, KEY_A, bp))
    blob[-1] ^= 0x01  # flip a ciphertext/tag byte
    with pytest.raises(DecryptError):
        decode_entry(bytes(blob), KEY_A, bp)


def test_wrong_key_raises():
    bp_a = _blinded("proj-1", KEY_A)
    blob = encode_entry(OP, KEY_A, bp_a)
    bp_b = _blinded("proj-1", KEY_B)
    with pytest.raises(DecryptError):
        decode_entry(blob, KEY_B, bp_b)


def test_cross_project_blinded_pid_as_aad_raises():
    """A blob encrypted for project A cannot be decrypted with project B's AAD."""
    bp_a = _blinded("proj-1", KEY_A)
    blob = encode_entry(OP, KEY_A, bp_a)
    bp_other = _blinded("proj-OTHER", KEY_A)  # same key, different blinded pid
    with pytest.raises(DecryptError):
        decode_entry(blob, KEY_A, bp_other)


# ---------------------------------------------------------------------------
# frame / unframe
# ---------------------------------------------------------------------------

def test_frame_unframe_roundtrip():
    blobs = [os.urandom(n) for n in (1, 17, 250, 0, 4096)]
    framed = frame_blobs(blobs)
    assert unframe_blobs(framed) == blobs


def test_unframe_truncated_length_prefix_raises():
    framed = frame_blobs([b"abcdef"])
    with pytest.raises(ValueError):
        unframe_blobs(framed[:2])  # cut mid length-prefix


def test_unframe_truncated_body_raises():
    framed = frame_blobs([b"abcdef"])
    with pytest.raises(ValueError):
        unframe_blobs(framed[:-2])  # declared length longer than remaining
