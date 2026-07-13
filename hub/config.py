"""hub.config — env-only settings. No secrets in code."""

from __future__ import annotations

import hashlib
import os


def _parse_tokens(raw: str) -> frozenset[str]:
    """Parse HUB_TOKENS: comma-separated raw tokens.

    Each token is SHA-256 hashed at startup. All comparisons use the hash set.
    The raw value is never stored beyond this function call.
    """
    tokens = [t.strip() for t in raw.split(",") if t.strip()]
    return frozenset(hashlib.sha256(t.encode()).hexdigest() for t in tokens)


class Config:
    """All configuration read from environment at construction time."""

    def __init__(self) -> None:
        raw_tokens = os.environ.get("HUB_TOKENS", "")
        self.allowed_token_hashes: frozenset[str] = (
            _parse_tokens(raw_tokens) if raw_tokens else frozenset()
        )
        if not self.allowed_token_hashes:
            raise ValueError(
                "HUB_TOKENS must be set to at least one token; "
                "refusing to start with no auth configured"
            )

        self.db_path: str = os.environ.get("HUB_DB_PATH", "hub/hub.db")
        self.audit_path: str = os.environ.get("HUB_AUDIT_PATH", "hub/audit.log")

        self.rate_limit: int = int(os.environ.get("HUB_RATE_LIMIT", "1000"))
        self.rate_window: int = int(os.environ.get("HUB_RATE_WINDOW", "3600"))

        # Max body bytes before 413 (default 10 MB)
        self.max_body_bytes: int = int(
            os.environ.get("HUB_MAX_BODY_BYTES", str(10 * 1024 * 1024))
        )
        # Max entries per POST batch before 413
        self.max_entries: int = int(os.environ.get("HUB_MAX_ENTRIES", "1000"))
        # Max entries returned per GET /oplog pull (page size); has_more=True
        # in the response signals the client to pull again with the new cursor.
        self.max_pull: int = int(os.environ.get("HUB_MAX_PULL", "500"))
        # Max decoded payload bytes per entry before 422 (default 1 MB)
        self.max_payload_bytes: int = int(
            os.environ.get("HUB_MAX_PAYLOAD_BYTES", str(1024 * 1024))
        )

        # Optional TLS (pass to uvicorn via CLI; documented in RUNBOOK)
        self.tls_cert: str | None = os.environ.get("HUB_TLS_CERT")
        self.tls_key: str | None = os.environ.get("HUB_TLS_KEY")
