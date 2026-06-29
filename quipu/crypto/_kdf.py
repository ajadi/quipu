"""quipu.crypto._kdf — Argon2id key derivation.

Parameters come from the S1 spike (TASK-010, rapid-prototyper section).
DO NOT change without re-running the spike and updating the recommendation.
"""

from __future__ import annotations

from argon2.low_level import Type, hash_secret_raw

from quipu.crypto.errors import KdfError

# S1 spike recommended params (TASK-010): time_cost=3, memory_cost=65536 KiB,
# parallelism=4, hash_len=32. Measured ~105 ms on reference hardware.
# STARTUP-ONLY: derive_key is intentionally slow. Call once; cache the result.
# Never call on a hot path (encrypt/decrypt loop).
ARGON2_PARAMS: dict[str, int] = {
    "time_cost": 3,
    "memory_cost": 65536,  # 64 MiB
    "parallelism": 4,
    "hash_len": 32,
}


def derive_key(passphrase: str, salt: bytes) -> bytes:
    """Derive a 32-byte AES-256 key from *passphrase* and *salt* using Argon2id.

    Deterministic: same (passphrase, salt) always produces the same 32-byte key.

    Args:
        passphrase: User passphrase. Must be a non-empty string.
        salt: Random salt bytes (minimum 16 bytes recommended). NEVER hardcode.
              Use get_or_create_salt() to obtain a persisted random salt.

    Returns:
        32 raw bytes suitable for AES-256-GCM.

    Raises:
        KdfError: If argon2 raises an internal error.
        ValueError: If passphrase is empty or salt is too short.
    """
    if not passphrase:
        raise ValueError("passphrase must not be empty")
    if len(salt) < 16:
        raise ValueError("salt must be at least 16 bytes")

    try:
        return hash_secret_raw(
            secret=passphrase.encode("utf-8"),
            salt=salt,
            time_cost=ARGON2_PARAMS["time_cost"],
            memory_cost=ARGON2_PARAMS["memory_cost"],
            parallelism=ARGON2_PARAMS["parallelism"],
            hash_len=ARGON2_PARAMS["hash_len"],
            type=Type.ID,
        )
    except Exception as exc:
        raise KdfError(f"Argon2id derivation failed: {exc}") from exc
