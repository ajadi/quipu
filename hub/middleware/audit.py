"""hub.middleware.audit — append-only JSON-line audit log.

One line per request. Fields ONLY: ts, token_hash, blinded_project_id, op,
entry_count, byte_count, status.

NEVER logs: payload, plaintext, raw token, real project_id, or any content.
Runs AFTER the route resolves so it captures the final status code.
"""

from __future__ import annotations

import json
import os
import sys
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

_HEALTH_PATH = "/health"


class AuditMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, audit_path: str) -> None:
        super().__init__(app)
        self._audit_path = audit_path
        # Ensure parent directory exists
        parent = os.path.dirname(audit_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    async def dispatch(self, request: Request, call_next) -> Response:
        if request.url.path == _HEALTH_PATH:
            return await call_next(request)

        response = await call_next(request)

        token_hash = getattr(request.state, "token_hash", None)
        bpid = _extract_bpid(request)
        op = _extract_op(request)
        entry_count = getattr(request.state, "audit_entry_count", 0)
        byte_count = getattr(request.state, "audit_byte_count", 0)

        line = json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "token_hash": token_hash,
            "blinded_project_id": bpid,
            "op": op,
            "entry_count": entry_count,
            "byte_count": byte_count,
            "status": response.status_code,
        }, separators=(",", ":"))

        try:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError as exc:
            # Non-fatal: do not fail the request if audit write fails
            print(f"audit: failed to write audit log: {exc}", file=sys.stderr)

        return response


def _extract_bpid(request: Request) -> str | None:
    parts = request.url.path.split("/")
    if len(parts) >= 3 and parts[1] == "oplog":
        return parts[2] or None
    return None


def _extract_op(request: Request) -> str:
    method = request.method.upper()
    if method == "POST":
        return "push"
    if method == "GET":
        return "pull"
    return method.lower()
