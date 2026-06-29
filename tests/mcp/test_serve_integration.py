"""Integration tests: start `python -m quipu serve` as a subprocess and drive
it via the MCP SDK stdio client.

The test-only env var QUIPU_TEST_FAKE_EMBED=1 activates the fake embedding
hook in quipu/server.py so the subprocess doesn't need a real ONNX model.

Skipped automatically if mcp.client is not importable.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

from pathlib import Path

import pytest

# Skip the whole module if mcp.client is unavailable.
pytest.importorskip("mcp.client.stdio", reason="mcp.client not available")

from mcp.client.session import ClientSession  # noqa: E402
from mcp.client.stdio import StdioServerParameters, stdio_client  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]

_EXPECTED_TOOLS = {
    "quipu_write",
    "quipu_search",
    "quipu_get",
    "quipu_list",
    "quipu_invalidate",
    "quipu_flush",
    "quipu_stats",
    "quipu_push",
    "quipu_pull",
    "quipu_prime",
    "quipu_receipts",
    "quipu_gc",
    "quipu_graph",
}

_TIMEOUT = 30  # seconds per async operation


def _make_params(tmp_path) -> StdioServerParameters:
    """Build StdioServerParameters for a fresh isolated server process."""
    db_path = str(tmp_path / "itest.db")
    # Build env: inherit current env, override quipu vars, remove API key.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    env["QUIPU_DB_PATH"] = db_path
    env["QUIPU_PROJECT_ID"] = "itest"
    env["QUIPU_TEST_FAKE_EMBED"] = "1"
    env["QUIPU_ALLOW_TEST_HOOKS"] = "1"
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "quipu", "serve"],
        env=env,
        cwd=str(_REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _wf(coro, timeout: float = _TIMEOUT):
    """Wrap a coroutine with asyncio.wait_for."""
    return await asyncio.wait_for(coro, timeout=timeout)


# ---------------------------------------------------------------------------
# Tests — each test spawns its own server subprocess for isolation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_initialize(tmp_path):
    """initialize() succeeds; serverInfo and capabilities are present."""
    params = _make_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            result = await _wf(session.initialize())
            assert result.serverInfo is not None, "serverInfo missing"
            assert result.serverInfo.name == "quipu"
            caps = result.capabilities
            assert caps is not None, "capabilities missing"
            assert caps.tools is not None, "tools capability missing"


@pytest.mark.asyncio
async def test_list_tools(tmp_path):
    """list_tools() returns exactly the 7 expected tool names, each with a
    non-empty inputSchema."""
    params = _make_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _wf(session.initialize())
            result = await _wf(session.list_tools())
            names = {t.name for t in result.tools}
            assert names == _EXPECTED_TOOLS, f"tool names mismatch: {names}"
            for tool in result.tools:
                assert tool.inputSchema, (
                    f"tool {tool.name!r} has empty inputSchema"
                )


@pytest.mark.asyncio
async def test_write_search_round_trip(tmp_path):
    """quipu_write -> quipu_search: the written id appears in search results."""
    params = _make_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _wf(session.initialize())

            # Write a record with a unique marker string.
            write_result = await _wf(
                session.call_tool(
                    "quipu_write",
                    {"content": "unique integration marker XYZ"},
                )
            )
            write_payload = json.loads(write_result.content[0].text)
            assert "id" in write_payload, f"write response missing 'id': {write_payload}"
            written_id = write_payload["id"]
            assert written_id, "write returned empty id"

            # Search — project_id defaults to QUIPU_PROJECT_ID=itest.
            search_result = await _wf(
                session.call_tool(
                    "quipu_search",
                    {"query": "integration marker XYZ"},
                )
            )
            search_payload = json.loads(search_result.content[0].text)
            assert "results" in search_payload, (
                f"search response missing 'results': {search_payload}"
            )
            result_ids = [r["id"] for r in search_payload["results"]]
            assert written_id in result_ids, (
                f"written id {written_id!r} not found in search results: {result_ids}"
            )


@pytest.mark.asyncio
async def test_stats(tmp_path):
    """quipu_stats returns total>=1 with active+invalidated==total after a write."""
    params = _make_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _wf(session.initialize())

            # Write something first so there's at least one record.
            await _wf(
                session.call_tool("quipu_write", {"content": "stats test record"})
            )

            stats_result = await _wf(session.call_tool("quipu_stats", {}))
            payload = json.loads(stats_result.content[0].text)
            assert "error" not in payload, f"quipu_stats returned error: {payload}"
            assert payload["total"] >= 1, f"expected total>=1: {payload}"
            assert payload["active"] + payload["invalidated"] == payload["total"], (
                f"active+invalidated != total: {payload}"
            )


@pytest.mark.asyncio
async def test_malformed_call_and_server_survival(tmp_path):
    """quipu_write with no content returns an error (not a crash);
    a follow-up call (quipu_stats) still succeeds -> server stayed alive.

    The MCP SDK may intercept schema-invalid calls before dispatch and return
    isError=True with a plain text message, or dispatch may return a JSON body
    with {"error": ...}.  We accept either form.
    """
    params = _make_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _wf(session.initialize())

            # Malformed call: missing required 'content'.
            bad_result = await _wf(session.call_tool("quipu_write", {}))
            assert bad_result.content, "expected non-empty content on error response"
            error_text = bad_result.content[0].text
            # Accept SDK-level isError=True OR a dispatched {"error": ...} JSON body.
            is_error = bad_result.isError
            if not is_error:
                try:
                    parsed = json.loads(error_text)
                    is_error = "error" in parsed
                except (json.JSONDecodeError, AttributeError):
                    pass
            assert is_error, (
                f"expected error response for missing content, got: {error_text!r}"
            )

            # Server must still be alive — follow-up call succeeds.
            # Write a record first so stats has a project to work with.
            await _wf(
                session.call_tool("quipu_write", {"content": "survival probe"})
            )
            stats_result = await _wf(session.call_tool("quipu_stats", {}))
            stats_payload = json.loads(stats_result.content[0].text)
            assert "error" not in stats_payload, (
                f"server died after malformed call; stats error: {stats_payload}"
            )
            assert stats_payload["total"] >= 1


@pytest.mark.asyncio
async def test_prime_round_trip(tmp_path):
    """quipu_write a marker atom, then quipu_prime with matching topic:
    the marker id must appear in prime results."""
    params = _make_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _wf(session.initialize())

            # Write a record with a unique marker string.
            write_result = await _wf(
                session.call_tool(
                    "quipu_write",
                    {"content": "prime integration marker ZETA unique"},
                )
            )
            write_payload = json.loads(write_result.content[0].text)
            assert "id" in write_payload, f"write response missing 'id': {write_payload}"
            written_id = write_payload["id"]

            # Prime with a topic that matches the marker.
            prime_result = await _wf(
                session.call_tool(
                    "quipu_prime",
                    {"topic": "prime integration marker ZETA"},
                )
            )
            prime_payload = json.loads(prime_result.content[0].text)
            assert "error" not in prime_payload, f"quipu_prime returned error: {prime_payload}"
            assert prime_payload.get("primed") is True, f"expected primed=true: {prime_payload}"
            assert "results" in prime_payload, f"missing 'results': {prime_payload}"
            result_ids = [r["id"] for r in prime_payload["results"]]
            assert written_id in result_ids, (
                f"written id {written_id!r} not found in prime results: {result_ids}"
            )


@pytest.mark.asyncio
async def test_prime_empty_store_no_error(tmp_path):
    """On a fresh db, quipu_prime with no prior writes must:
    - return no 'error' key
    - return results == []
    """
    params = _make_params(tmp_path)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await _wf(session.initialize())

            prime_result = await _wf(session.call_tool("quipu_prime", {}))
            prime_payload = json.loads(prime_result.content[0].text)
            assert "error" not in prime_payload, (
                f"quipu_prime returned error on empty store: {prime_payload}"
            )
            assert prime_payload.get("results") == [], (
                f"expected empty results on fresh db: {prime_payload}"
            )
