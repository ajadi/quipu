"""Quipu MCP server: builds a configured mcp.server.Server and provides the
stdio run entrypoint.

Public API:
    build_server(store, default_project_id) -> mcp.server.Server
    async run_stdio(*, store=None, default_project_id=None) -> None

Test hook:
    If the environment variable QUIPU_TEST_FAKE_EMBED=1 is set, run_stdio()
    installs a deterministic fake embedding engine BEFORE opening the store.
    This hook is TEST-ONLY and is off by default (no production behavior change).
"""

from __future__ import annotations

import logging
import os

from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from quipu.mcp.tools import TOOLS, dispatch
from quipu.storage import Store

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TEST-ONLY fake-embedding engine (activated via QUIPU_TEST_FAKE_EMBED=1)
# Mirrors the _FakeSession/_FakeTokenizer pattern in tests/mcp/conftest.py
# but lives here so it can be used by a subprocess that can't import conftest.
# ---------------------------------------------------------------------------


def _install_fake_embed_engine() -> None:  # pragma: no cover — test-hook path
    """Inject a minimal fake embedding engine that requires no ONNX model.

    Called only when QUIPU_TEST_FAKE_EMBED=1 is set in the environment.
    """
    import numpy as np
    from quipu.embeddings.engine import set_engine, EMBED_DIM, _Engine

    class _FakeTokEnc:
        def __init__(self, seq_len: int = 8) -> None:
            self.ids = [1] * seq_len
            self.attention_mask = [1] * seq_len

    class _FakeTok:
        def encode_batch(self, texts):
            return [_FakeTokEnc() for _ in texts]

    class _N:
        def __init__(self, name: str) -> None:
            self.name = name
            self.type = "tensor(int64)"

    class _FakeSess:
        def get_inputs(self):
            return [_N("input_ids"), _N("attention_mask")]

        def get_outputs(self):
            return [_N("sentence_embedding")]

        def run(self, output_names, feeds):
            n = feeds["input_ids"].shape[0]
            return [np.full((n, EMBED_DIM), 1.0, dtype=np.float32)]

    set_engine(_Engine(session=_FakeSess(), tokenizer=_FakeTok()))


def build_server(store: Store, default_project_id: str | None) -> Server:
    """Build and return a configured MCP Server instance.

    Args:
        store: Long-lived Store to inject into every tool call.
        default_project_id: Fallback project_id when tool args omit it.

    Returns:
        Configured mcp.server.Server with list_tools and call_tool handlers.
    """
    server = Server("quipu", version="0.1.0")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return TOOLS

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[TextContent]:
        return dispatch(
            name,
            store=store,
            default_project_id=default_project_id,
            arguments=arguments or {},
        )

    return server


async def run_stdio(
    *,
    store: Store | None = None,
    default_project_id: str | None = None,
) -> None:
    """Run the Quipu MCP server over stdio until the client disconnects.

    Args:
        store: Injected store (not closed on exit). If None, opens via
               store(None) which auto-resolves QUIPU_DB_PATH, and owns/closes
               it on shutdown.
        default_project_id: Fallback project_id. If None, reads
                            QUIPU_PROJECT_ID env at call time.
    """
    from quipu.storage import store as open_store

    # TEST-ONLY hook: install fake embed engine before store opens so that
    # embedding calls in the subprocess don't require a real ONNX model.
    # Requires BOTH QUIPU_TEST_FAKE_EMBED=1 AND QUIPU_ALLOW_TEST_HOOKS=1 to
    # prevent silent activation in a production launch.
    if os.environ.get("QUIPU_TEST_FAKE_EMBED") == "1":
        if os.environ.get("QUIPU_ALLOW_TEST_HOOKS") == "1":
            logger.warning(
                "QUIPU_TEST_FAKE_EMBED active — using FAKE embeddings; NOT for production"
            )
            _install_fake_embed_engine()
        else:
            logger.warning(
                "QUIPU_TEST_FAKE_EMBED ignored: QUIPU_ALLOW_TEST_HOOKS not set"
            )

    if default_project_id is None:
        default_project_id = os.environ.get("QUIPU_PROJECT_ID")

    owns_store = store is None
    if owns_store:
        store = open_store(None)

    # Session-start pull: if project + hub are configured, pull before serving.
    if default_project_id is not None:
        try:
            from quipu.config import get_hub_config
            if get_hub_config() is not None:
                from quipu.sync.client import sync_now
                sync_now(default_project_id, store=store, directions=("pull",))
        except Exception:
            logger.warning("run_stdio: session-start pull failed", exc_info=True)

    try:
        srv = build_server(store, default_project_id)
        init_options = srv.create_initialization_options()
        async with stdio_server() as (read_stream, write_stream):
            await srv.run(read_stream, write_stream, init_options)
    finally:
        if owns_store:
            store.close()
