"""quipu.crypto._blind — HMAC-SHA256 project-id blinding with HKDF sub-key.

blind_project_id() produces a hub-safe pseudonym for a project that the hub
cannot reverse without the encryption key. Domain-separated from the AES
content key via HKDF to prevent cross-protocol key reuse.

Non-reversibility:
    The output is HMAC-SHA256(mac_key, project_id.encode('utf-8')) where
    mac_key is derived via HKDF-SHA256 with info=b"ember:project-id-mac:v1".
    Reversing the output requires either:
      (a) brute-forcing HMAC-SHA256 (preimage-resistant), or
      (b) recovering the 32-byte mac_key (which requires the master key).
    The hub stores only the 64-hex output and cannot recover project_id without
    the master key. The HKDF info string provides domain separation so the
    mac_key is cryptographically independent of the AES content key even though
    both are derived from the same master key.
"""

from __future__ import annotations

import hashlib
import hmac

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

_HKDF_INFO = b"ember:project-id-mac:v1"  # wire-value: HKDF info string kept as "ember" — renaming breaks all existing encrypted stores / hub sync
_MAC_KEY_LEN = 32


def _derive_mac_key(key: bytes) -> bytes:
    """Derive a MAC sub-key from the master key via HKDF-SHA256.

    info=b"ember:project-id-mac:v1" domain-separates this sub-key from
    the AES content key so there is no cross-protocol key interaction.
    """
    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_MAC_KEY_LEN,
        salt=None,  # Intentional: RFC 5869 §3.1 — salt=None is valid when IKM is
                    # already high-entropy (32-byte Argon2id output). An explicit
                    # HKDF salt would add no security here.
        info=_HKDF_INFO,
    )
    return hkdf.derive(key)


def blind_project_id(project_id: str, key: bytes) -> str:
    """Return a 64-hex pseudonym for *project_id* that the hub cannot reverse.

    Deterministic: same (project_id, key) always returns the same 64-hex string.

    Algorithm:
        mac_key = HKDF-SHA256(ikm=key, salt=None, info=b"ember:project-id-mac:v1", length=32)
        return HMAC-SHA256(mac_key, project_id.encode("utf-8")).hexdigest()

    Non-reversibility: See module docstring.

    Args:
        project_id: Plain project identifier (e.g. from quipu.config.get_project_id()).
        key: 32-byte master key (from derive_key or QUIPU_KEY escape hatch).

    Returns:
        64-character lowercase hex string (256-bit MAC output).
    """
    if len(key) != 32:
        raise ValueError(f"key must be 32 bytes, got {len(key)}")

    mac_key = _derive_mac_key(key)
    return hmac.new(mac_key, project_id.encode("utf-8"), hashlib.sha256).hexdigest()
