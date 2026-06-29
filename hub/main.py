"""hub.main — FastAPI ASGI app factory.

Lifespan: open WAL store + init_db at startup, close on shutdown.
Middleware order (outer to inner): SizeLimit -> Auth -> RateLimit -> Audit -> routes.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from hub.config import Config
from hub.middleware.audit import AuditMiddleware
from hub.middleware.auth import AuthMiddleware
from hub.middleware.ratelimit import RateLimitMiddleware
from hub.middleware.size_limit import SizeLimitMiddleware
from hub.routes.health import router as health_router
from hub.routes.oplog import router as oplog_router
from hub.store import get_connection, init_db


def create_app(config: Config | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: open DB connection, init schema
        conn = get_connection(config.db_path)
        init_db(conn)
        app.state.db_conn = conn
        app.state.config = config
        yield
        # Shutdown: close DB connection
        conn.close()

    app = FastAPI(
        title="Quipu Hub",
        description="Zero-knowledge oplog relay service",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Middleware stacked: outermost runs first on request, last on response.
    # Order: SizeLimit -> Auth -> RateLimit -> Audit
    # (Starlette adds middleware in reverse — last added = outermost)
    app.add_middleware(AuditMiddleware, audit_path=config.audit_path)
    app.add_middleware(
        RateLimitMiddleware,
        rate_limit=config.rate_limit,
        rate_window=config.rate_window,
    )
    app.add_middleware(
        AuthMiddleware,
        allowed_token_hashes=config.allowed_token_hashes,
    )
    app.add_middleware(SizeLimitMiddleware, max_body_bytes=config.max_body_bytes)

    app.include_router(health_router)
    app.include_router(oplog_router)

    return app


app = create_app()
