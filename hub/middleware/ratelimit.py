"""hub.middleware.ratelimit — per-(token_hash, blinded_project_id) fixed-window.

State lives in-process (resets on restart — acceptable V1, documented).
Default 1000 req / HUB_RATE_WINDOW (3600 s). Exceed -> 429 + Retry-After.
/health is exempt (no auth, no rate-limit key).
"""

from __future__ import annotations

import time
from heapq import heappop, heappush
from dataclasses import dataclass, field

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_HEALTH_PATH = "/health"
_MAX_BUCKETS = 10_000


@dataclass
class _Window:
    count: int = 0
    window_start: float = field(default_factory=time.monotonic)


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        rate_limit: int,
        rate_window: int,
        max_buckets: int = _MAX_BUCKETS,
    ) -> None:
        super().__init__(app)
        self._limit = rate_limit
        self._window = rate_window
        self._max_buckets = max_buckets
        # (token_hash, blinded_project_id) -> _Window, capped at active windows.
        # In-process state resets on restart (process restart clears all counters).
        self._buckets: dict[tuple[str, str], _Window] = {}
        self._expires: list[tuple[float, tuple[str, str]]] = []

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
        bucket = self._bucket_for(key, now)
        if bucket is None:
            retry_after = int(self._expires[0][0] - now) + 1
            return JSONResponse(
                {"detail": "Rate limit bucket capacity exceeded"},
                status_code=429,
                headers={"Retry-After": str(max(retry_after, 1))},
            )

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

    def _bucket_for(self, key: tuple[str, str], now: float) -> _Window | None:
        """Return an active bucket without evicting any active rate-limit window."""
        self._evict_expired(now)

        bucket = self._buckets.get(key)
        if bucket is not None:
            return bucket
        if len(self._buckets) >= self._max_buckets:
            return None

        bucket = _Window(window_start=now)
        self._buckets[key] = bucket
        heappush(self._expires, (now + self._window, key))
        return bucket

    def _evict_expired(self, now: float) -> None:
        """Remove elapsed fixed windows from the expiry heap in O(log n) each."""
        while self._expires and self._expires[0][0] <= now:
            _, key = heappop(self._expires)
            del self._buckets[key]


def _extract_bpid(request: Request) -> str | None:
    """Extract blinded_project_id from path /oplog/{bpid}."""
    parts = request.url.path.split("/")
    # Expected: ['', 'oplog', '<bpid>']
    if len(parts) >= 3 and parts[1] == "oplog":
        return parts[2] or None
    return None
