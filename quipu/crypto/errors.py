"""quipu.crypto.errors — crypto exception hierarchy."""


class CryptoError(Exception):
    """Base class for all quipu crypto errors."""


class DecryptError(CryptoError):
    """Raised when decryption fails: wrong key, tampered ciphertext, or AAD mismatch."""


class KdfError(CryptoError):
    """Raised when key derivation fails (e.g. argon2 internal error)."""
