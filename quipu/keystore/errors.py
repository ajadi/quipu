"""quipu.keystore.errors — keystore exception hierarchy."""


class KeystoreUnavailable(Exception):
    """Raised when no usable OS keyring backend can be found.

    In production this triggers the InMemoryBackend fallback path; callers
    should not catch this directly unless overriding the fallback behaviour.
    """
