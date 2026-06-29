"""hub.routes.health — liveness check. No auth, no DB write."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health():
    """Return 200 {"status":"ok"}. Unauthenticated liveness check."""
    return {"status": "ok"}
