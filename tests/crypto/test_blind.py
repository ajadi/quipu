"""Unit tests for quipu.crypto._blind (blind_project_id)."""

from __future__ import annotations

import hashlib
import hmac
import os

import pytest

from quipu.crypto._blind import blind_project_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _random_key() -> bytes:
    return os.urandom(32)


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------

def test_blind_project_id_returns_64_hex_chars():
    result = blind_project_id("proj-abc", _random_key())
    assert len(result) == 64
    assert result == result.lower()
    assert all(c in "0123456789abcdef" for c in result)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_blind_project_id_deterministic():
    key = _random_key()
    pid = "project-X"
    assert blind_project_id(pid, key) == blind_project_id(pid, key)


# ---------------------------------------------------------------------------
# Input sensitivity
# ---------------------------------------------------------------------------

def test_blind_project_id_different_project_id_different_output():
    key = _random_key()
    out1 = blind_project_id("project-A", key)
    out2 = blind_project_id("project-B", key)
    assert out1 != out2


def test_blind_project_id_different_key_different_output():
    pid = "same-project"
    out1 = blind_project_id(pid, _random_key())
    out2 = blind_project_id(pid, _random_key())
    assert out1 != out2


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_blind_project_id_non_32_byte_key_raises_value_error():
    with pytest.raises(ValueError, match="32 bytes"):
        blind_project_id("proj", os.urandom(16))


def test_blind_project_id_31_byte_key_raises_value_error():
    with pytest.raises(ValueError, match="32 bytes"):
        blind_project_id("proj", os.urandom(31))


# ---------------------------------------------------------------------------
# Domain separation — HKDF sub-key, not the raw master key
# ---------------------------------------------------------------------------

def test_blind_project_id_differs_from_plain_hmac_sha256_on_raw_key():
    """The implementation uses HKDF to derive a sub-key before HMAC.
    The output must differ from a naive HMAC-SHA256(key, project_id).
    This asserts that the HKDF domain-separation step is actually applied.
    """
    key = _random_key()
    pid = "test-project"

    blinded = blind_project_id(pid, key)

    # Plain HMAC-SHA256 directly on the master key (no HKDF sub-key).
    plain_hmac = hmac.new(key, pid.encode("utf-8"), hashlib.sha256).hexdigest()

    assert blinded != plain_hmac, (
        "blind_project_id must use an HKDF-derived sub-key, not the raw master key"
    )
