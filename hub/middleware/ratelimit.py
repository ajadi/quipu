"""hub.middleware.ratelimit — per-(token_hash, blinded_project_id) fixed-window.

State lives in-process (resets on restart — acceptable V1, documented).
Default 1000 req / HUB_RATE_WINDOW (3600 s). Exceed -> 429 + Retry-After.
/health is exempt (no auth, no rate-limit key).
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_HEALTH_PATH = "/health"


@dataclass
class _Window:
    count: int = 0
    window_start: float = field(default_factory=time.monotonic)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, rate_limit: int, rate_window: int) -> None:
        super().__init__(app)
        self._limit = rate_limit
        self._window = rate_window
        # (token_hash, blinded_project_id) -> _Window
        # V1 limitation: _buckets grows unbounded — one entry per unique
        # (token_hash, blinded_project_id) pair, never evicted.  Acceptable for
        # single-token self-hosted V1; a future version should add TTL/LRU eviction.
        # In-process state resets on restart (process restart clears all counters) —
        # acceptable V1 caveat (documented in module docstring above).
        self._buckets: dict[tuple[str, str], _Window] = defaultdict(_Window)

    async def dispatch(self, request: Request, call_next):
        if request.url.path == _HEALTH_PATH:
            return await call_next(request)

        token_hash = getattr(request.state, "token_hash", None)
        # Extract blinded_project_id from path (segment after /oplog/)
        bpid = _extract_bpid(request)
        if token_hash is None or bpid is None:
            # Auth middleware should have caught missing token; pass through
            return await call_next(request)

        key = (token_hash, bpid)
        now = time.monotonic()
        bucket = self._buckets[key]

        if now - bucket.window_start >= self._window:
            # New window: reset
            bucket.count = 0
            bucket.window_start = now

        bucket.count += 1
        if bucket.count > self._limit:
            retry_after = int(self._window - (now - bucket.window_start)) + 1
            return JSONResponse(
                {"detail": "Rate limit exceeded"},
                status_code=429,
                headers={"Retry-After": str(max(retry_after, 1))},
            )

        return await call_next(request)


def _extract_bpid(request: Request) -> str | None:
    """Extract blinded_project_id from path /oplog/{bpid}."""
    parts = request.url.path.split("/")
    # Expected: ['', 'oplog', '<bpid>']
    if len(parts) >= 3 and parts[1] == "oplog":
        return parts[2] or None
    return None
