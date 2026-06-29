"""hub.models — Pydantic models + parse_cursor defensive helper."""

from __future__ import annotations

import base64
import os
import re

from fastapi import HTTPException
from pydantic import BaseModel, field_validator

# Per-entry decoded payload size cap.  Read once at import time from the same
# env var that Config uses so the limit is consistent without a circular import.
# Read at IMPORT time — a test needing a different cap must set HUB_MAX_PAYLOAD_BYTES
# in the environment BEFORE importing hub.models (or reload the module).
_MAX_PAYLOAD_BYTES: int = int(os.environ.get("HUB_MAX_PAYLOAD_BYTES", str(1024 * 1024)))

# Strict cursor regex: digits only, max 19 chars (covers max int64)
_CURSOR_RE = re.compile(r"^\d+$")
_CURSOR_MAX_LEN = 19

# Valid blinded_project_id pattern: exactly 64 hex chars
_BPID_RE = re.compile(r"^[0-9a-f]{64}$")


def validate_blinded_project_id(bpid: str) -> None:
    """Raise HTTP 400 if bpid does not match ^[0-9a-f]{64}$."""
    if not _BPID_RE.match(bpid):
        raise HTTPException(
            status_code=400,
            detail="blinded_project_id must be 64 lowercase hex chars",
        )


def parse_cursor(raw: str | None) -> str | None:
    r"""Defensive cursor parser.

    - None or '' -> None (first pull from start)
    - Strict ^\d+$ AND len <= 19 AND non-negative -> return raw string as-is
    - Anything else -> HTTP 400 (never 500, never crashes, never unbounded int())
    """
    if raw is None or raw == "":
        return None
    if len(raw) > _CURSOR_MAX_LEN:
        raise HTTPException(
            status_code=400,
            detail="cursor too long (max 19 digits)",
        )
    if not _CURSOR_RE.match(raw):
        raise HTTPException(
            status_code=400,
            detail="cursor must be a non-negative integer string",
        )
    # Now safe to parse (bounded length, digits-only); ^\d+$ guarantees non-negative.
    _ = int(raw)
    return raw


class PushEntry(BaseModel):
    """One hub-visible entry dict on the wire. payload is base64."""

    entry_id: str
    client_id: str
    sequence_no: int
    op: str
    record_id: str
    blinded_project_id: str
    ts: str
    payload: str  # base64 string on the wire

    @field_validator("op")
    @classmethod
    def op_must_be_valid(cls, v: str) -> str:
        if v not in ("upsert", "invalidate"):
            raise ValueError("op must be 'upsert' or 'invalidate'")
        return v

    @field_validator("payload")
    @classmethod
    def payload_must_be_base64(cls, v: str) -> str:
        try:
            decoded = base64.b64decode(v, validate=True)
        except Exception:
            raise ValueError("payload must be valid base64")
        # Per-entry decoded size cap (HUB_MAX_PAYLOAD_BYTES, default 1 MB).
        # Reuse the already-decoded value — do not decode twice.
        if len(decoded) > _MAX_PAYLOAD_BYTES:
            raise ValueError(
                f"payload decoded size {len(decoded)} bytes exceeds "
                f"per-entry limit of {_MAX_PAYLOAD_BYTES} bytes"
            )
        return v


class PushBody(BaseModel):
    """POST /oplog/{bpid} request body."""

    entries: list[PushEntry]
