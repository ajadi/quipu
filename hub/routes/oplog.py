"""hub.routes.oplog — POST /oplog/{bpid} and GET /oplog/{bpid}.

Enforces:
- blinded_project_id path param: ^[0-9a-f]{64}$ -> 400 on mismatch
- POST: each entry.blinded_project_id must match path bpid -> 400
- POST: max entries per batch (HUB_MAX_ENTRIES) -> 413
- GET: parse_cursor defensive validation -> 400 on bad cursor; 200+empty on out-of-range
- GET: response is paginated to cfg.max_pull entries per page; "has_more": true
  signals the client to pull again with the returned "cursor" as the next since=
- payload stored/returned verbatim (base64 wire -> bytes BLOB -> base64 wire)
- ingest_seq NEVER exposed in responses
"""

from __future__ import annotations

import base64

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from hub.models import PushBody, parse_cursor, validate_blinded_project_id
from hub.store import append, get_cursor, read_since

router = APIRouter()


@router.post("/oplog/{blinded_project_id}")
async def push_entries(
    blinded_project_id: str,
    body: PushBody,
    request: Request,
):
    """Append encrypted entries for a blinded project. Idempotent by entry_id."""
    validate_blinded_project_id(blinded_project_id)

    cfg = request.app.state.config
    conn = request.app.state.db_conn

    # Entry count limit
    if len(body.entries) > cfg.max_entries:
        return JSONResponse(
            {"detail": f"Batch exceeds max_entries limit of {cfg.max_entries}"},
            status_code=413,
        )

    # Validate each entry's blinded_project_id matches the path
    for entry in body.entries:
        if entry.blinded_project_id != blinded_project_id:
            return JSONResponse(
                {"detail": "entry.blinded_project_id does not match path"},
                status_code=400,
            )

    # Convert Pydantic models to plain dicts for store
    entry_dicts = [e.model_dump() for e in body.entries]

    # Byte count for audit (payload bytes decoded size)
    byte_count = sum(
        len(base64.b64decode(e["payload"])) for e in entry_dicts
    )

    append(conn, blinded_project_id, entry_dicts)

    # Set audit metadata on request state for AuditMiddleware
    request.state.audit_entry_count = len(entry_dicts)
    request.state.audit_byte_count = byte_count

    # Return cursor = MAX(ingest_seq) for this project after ingest.
    # O(1) SELECT MAX query — does not fetch or re-encode any payload rows.
    cursor = get_cursor(conn, blinded_project_id)

    return {"cursor": cursor}


@router.get("/oplog/{blinded_project_id}")
async def pull_entries(
    blinded_project_id: str,
    request: Request,
    since: str | None = None,
):
    """Return a page of entries since cursor. Empty result if cursor is ahead of store."""
    validate_blinded_project_id(blinded_project_id)

    cfg = request.app.state.config
    conn = request.app.state.db_conn

    cursor = parse_cursor(since)
    entries, next_cursor, has_more = read_since(
        conn, blinded_project_id, cursor, cfg.max_pull
    )

    # Set audit metadata
    byte_count = sum(
        len(base64.b64decode(e["payload"])) for e in entries
    )
    request.state.audit_entry_count = len(entries)
    request.state.audit_byte_count = byte_count

    return {"entries": entries, "cursor": next_cursor, "has_more": has_more}
