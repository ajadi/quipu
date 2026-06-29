"""quipu.sync.errors — typed error taxonomy for the sync layer."""

from __future__ import annotations


class SyncError(Exception):
    """Base class for all sync errors."""


class SyncUnavailableError(SyncError):
    """Hub unreachable: network failure, timeout, connection reset, 5xx, 429.

    Triggers offline-degrade: local operation continues, retry on next trigger.
    """


class SyncAuthError(SyncError):
    """HTTP 401 — bad or missing bearer token."""


class SyncProtocolError(SyncError):
    """Protocol violation: 400/413/422, malformed response body, bad cursor."""
