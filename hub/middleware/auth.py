"""hub.middleware.auth — Bearer token authentication.

Algorithm:
  1. Extract token from Authorization: Bearer <token>.
  2. Compute SHA-256(presented_token).
  3. Compare CONSTANT-TIME via hmac.compare_digest against each allowed hash.
  4. Missing / malformed / wrong token -> 401.
  5. /health is exempt (checked in route, not here — middleware skips /health).
"""

from __future__ import annotations

import hashlib
import hmac

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

_HEALTH_PATH = "/health"


class AuthMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, allowed_token_hashes: frozenset[str]) -> None:
        super().__init__(app)
        self._hashes = allowed_token_hashes

    async def dispatch(self, request: Request, call_next):
        # /health is the only unauthenticated route
        if request.url.path == _HEALTH_PATH:
            return await call_next(request)

        token = _extract_bearer(request)
        if token is None:
            return JSONResponse(
                {"detail": "Missing or malformed Authorization header"},
                status_code=401,
            )

        token_hash = hashlib.sha256(token.encode()).hexdigest()
        if not _verify_token(token_hash, self._hashes):
            return JSONResponse(
                {"detail": "Invalid bearer token"},
                status_code=401,
            )

        # Attach token_hash for downstream middleware (audit, ratelimit)
        request.state.token_hash = token_hash
        return await call_next(request)


def _extract_bearer(request: Request) -> str | None:
    """Return token string from 'Authorization: Bearer <token>', or None."""
    header = request.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return None
    token = header[len("Bearer "):]
    return token if token else None


def _verify_token(token_hash: str, allowed: frozenset[str]) -> bool:
    """Constant-time membership check over all allowed hashes.

    Iterates all hashes to avoid early exit leaking hash set size.
    hmac.compare_digest prevents timing attacks on the comparison itself.
    """
    matched = False
    for allowed_hash in allowed:
        if hmac.compare_digest(token_hash, allowed_hash):
            matched = True
        # No break — always run all comparisons
    return matched
