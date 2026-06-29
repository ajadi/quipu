"""hub.middleware.size_limit — reject oversized request bodies before store.

HUB_MAX_BODY_BYTES (default 10 MB) -> 413.
Entry-count limit enforced in route handler (size_limit reads Content-Length
for the byte check; entry count requires parsing the body).

Two-phase cap:
  1. Content-Length header fast-path: reject before buffering when header is present.
  2. Raw body cap: read the full body and reject if it exceeds the limit, regardless
     of whether Content-Length was sent.  This prevents attackers from omitting the
     header and streaming an unbounded body to Uvicorn's buffer.
     After reading, the body is cached back onto request._body so downstream
     handlers (Pydantic/FastAPI body parsing) still receive it.
"""

from __future__ import annotations

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_HEALTH_PATH = "/health"


class SizeLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_bytes: int) -> None:
        super().__init__(app)
        self._max = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        if request.url.path == _HEALTH_PATH:
            return await call_next(request)

        # Fast-path: reject early if Content-Length header announces an oversize body.
        content_length = request.headers.get("Content-Length")
        if content_length is not None:
            try:
                if int(content_length) > self._max:
                    return JSONResponse(
                        {"detail": f"Request body exceeds limit of {self._max} bytes"},
                        status_code=413,
                    )
            except ValueError:
                pass  # Malformed Content-Length; fall through to raw-body cap

        # Real cap: read the body unconditionally and reject if too large.
        # An attacker who omits Content-Length would otherwise bypass the fast-path
        # and stream an unbounded body to Uvicorn before the entry-count gate fires.
        body = await request.body()
        if len(body) > self._max:
            return JSONResponse(
                {"detail": f"Request body exceeds limit of {self._max} bytes"},
                status_code=413,
            )
        # Cache the body so downstream (FastAPI body parsing) can still read it.
        request._body = body  # type: ignore[attr-defined]

        return await call_next(request)
