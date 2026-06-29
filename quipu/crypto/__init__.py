"""quipu.crypto — end-to-end encryption primitives.

Public API:
    from quipu.crypto import (
        derive_key, ARGON2_PARAMS,
        encrypt_record, decrypt_record, serialize_blob, deserialize_blob, BLOB_VERSION,
        blind_project_id,
        CryptoError, DecryptError, KdfError,
    )
"""

from quipu.crypto._kdf import ARGON2_PARAMS, derive_key
from quipu.crypto._cipher import (
    BLOB_VERSION,
    decrypt_record,
    deserialize_blob,
    encrypt_record,
    serialize_blob,
)
from quipu.crypto._blind import blind_project_id
from quipu.crypto.errors import CryptoError, DecryptError, KdfError

__all__ = [
    "ARGON2_PARAMS",
    "derive_key",
    "BLOB_VERSION",
    "encrypt_record",
    "decrypt_record",
    "serialize_blob",
    "deserialize_blob",
    "blind_project_id",
    "CryptoError",
    "DecryptError",
    "KdfError",
]
