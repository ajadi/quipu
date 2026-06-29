"""quipu.oplog.codec — the encryption boundary for oplog payloads.

This module and quipu/sync/_aad.py are the ONLY two modules in quipu/oplog or
quipu/sync allowed to import encrypt_record / decrypt_record. Every encrypt AND
decrypt routes its AAD through quipu.sync._aad.aad_for_blinded(blinded_pid),
which equals blind_project_id(project_id, key).encode().

The cipher (quipu/crypto/_cipher) owns the per-record blob layout
(version/key_version/nonce/ct+tag) with a fresh os.urandom nonce per call. The
OUTER length-prefix frame that batches many blobs for one transport call is THIS
layer's job: frame_blobs / unframe_blobs.

Payload plaintext is a JSON object describing the operation:
    {"op": "upsert"|"invalidate", "record_id": str, "ts": str, "content": str|None,
     "type": str, "scope": str, "metadata": dict, "refs": list, "project_id": str|None}
For 'invalidate', content/embedding are not required.
"""

from __future__ import annotations

import json
import struct

from quipu.crypto import decrypt_record, encrypt_record
from quipu.crypto.errors import DecryptError


def encode_entry(op_fields: dict, key: bytes, blinded_project_id: str) -> bytes:
    """Serialize *op_fields* to JSON and encrypt with AAD = blinded_project_id.

    Args:
        op_fields: plaintext operation dict (op, record_id, ts, content, ...).
        key: 32-byte project key (caller-derived once; never per-entry KDF).
        blinded_project_id: 64-hex pseudonym; bound as AAD.

    Returns:
        Opaque encrypted blob (versioned; safe to store on a zero-knowledge hub).
    """
    from quipu.sync._aad import aad_for_blinded  # local import breaks oplog<->sync cycle

    plaintext = json.dumps(op_fields, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return encrypt_record(plaintext, key, aad=aad_for_blinded(blinded_project_id))


def decode_entry(payload: bytes, key: bytes, blinded_project_id: str) -> dict:
    """Decrypt *payload* (AAD = blinded_project_id) and deserialize the JSON op dict.

    Raises:
        DecryptError: on wrong key, tampered blob, or AAD (cross-project) mismatch.
    """
    from quipu.sync._aad import aad_for_blinded  # local import breaks oplog<->sync cycle

    plaintext = decrypt_record(payload, key, aad=aad_for_blinded(blinded_project_id))
    try:
        return json.loads(plaintext.decode("utf-8"))
    except (ValueError, UnicodeDecodeError) as exc:
        raise DecryptError(f"malformed decrypted payload: {exc}") from exc


# ---------------------------------------------------------------------------
# Outer length-prefix frame (batches blobs for one transport call)
# ---------------------------------------------------------------------------

def frame_blobs(blobs: list[bytes]) -> bytes:
    """Concatenate blobs as | 4B big-endian length | blob | repeated."""
    out = bytearray()
    for blob in blobs:
        out += struct.pack(">I", len(blob))
        out += blob
    return bytes(out)


def unframe_blobs(framed: bytes) -> list[bytes]:
    """Reverse frame_blobs. Raises ValueError on truncation / malformed frame."""
    blobs: list[bytes] = []
    offset = 0
    total = len(framed)
    while offset < total:
        if offset + 4 > total:
            raise ValueError("truncated frame: incomplete length prefix")
        (length,) = struct.unpack_from(">I", framed, offset)
        offset += 4
        if offset + length > total:
            raise ValueError("truncated frame: blob shorter than declared length")
        blobs.append(framed[offset : offset + length])
        offset += length
    return blobs
