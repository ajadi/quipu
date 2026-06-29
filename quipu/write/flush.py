"""Flush pipeline: Haiku enrichment of un-enriched atoms.

Only network surface in the write track. All network calls go through
_http_post_json (urllib.request only) — monkeypatched in tests.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from quipu.storage.store import Atom, Store

logger = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
HAIKU_MODEL = "claude-haiku-4-5"
ENRICHED_FLAG = "enriched"

_ANTHROPIC_VERSION = "2023-06-01"
_HTTP_TIMEOUT_S = 30


def _http_post_json(url: str, headers: dict, payload: dict) -> dict:
    """Send a POST request with JSON body and return the parsed response.

    This is the single network indirection point. Tests monkeypatch this
    function to avoid real HTTP calls.

    Args:
        url: Target URL.
        headers: HTTP headers dict.
        payload: Request body as dict (will be JSON-encoded).

    Returns:
        Parsed JSON response as dict.
    """
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT_S) as resp:
        return json.loads(resp.read().decode("utf-8"))


def flush(
    project_id: str | None = None,
    *,
    store: "Store | None" = None,
) -> dict:
    """Enrich un-enriched atoms via Claude Haiku.

    Skips gracefully if ANTHROPIC_API_KEY is absent. Never raises on
    missing key or network errors.

    DI seam: if store is None, opens and closes its own Store. If injected,
    uses caller's store and does NOT close it.

    Args:
        project_id: Optional project scope. If None, enriches all projects.
        store: Optional injected Store.

    Returns:
        dict with keys:
          - 'enriched': int — number of atoms enriched.
          - 'skipped': bool — True if enrichment was skipped entirely.
          - 'reason': str | None — reason for skip, or None if not skipped.
    """
    from quipu.storage import store as open_store

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning(
            "ANTHROPIC_API_KEY not set — skipping Haiku enrichment"
        )
        return {"enriched": 0, "skipped": True, "reason": "no_api_key"}

    own_store = store is None
    if own_store:
        store = open_store()

    try:
        return _run_enrichment(store, api_key, project_id)
    finally:
        if own_store:
            store.close()


def _run_enrichment(
    store: "Store",
    api_key: str,
    project_id: str | None,
) -> dict:
    """Internal: enrich un-enriched atoms and write back metadata."""
    # Collect un-enriched atoms
    if project_id is not None:
        candidates = store.list_by_project(project_id, include_invalidated=False)
    else:
        # No project_id: fetch all atoms (query all via connection directly)
        rows = store._conn.execute(
            "SELECT * FROM atoms WHERE invalidated = 0 ORDER BY created_at DESC"
        ).fetchall()
        from quipu.storage.store import _row_to_atom
        candidates = [_row_to_atom(r) for r in rows]

    unenriched = [
        a for a in candidates
        if not a.metadata.get(ENRICHED_FLAG, False)
    ]

    enriched_count = 0
    for atom in unenriched:
        try:
            enriched_count += _enrich_atom(store, atom, api_key)
        except Exception:
            # Log but don't raise — leave atom un-enriched
            logger.warning(
                "Failed to enrich atom %s — skipping", atom.id, exc_info=True
            )

    return {"enriched": enriched_count, "skipped": False, "reason": None}


def _enrich_atom(store: "Store", atom: "Atom", api_key: str) -> int:
    """Enrich a single atom via Haiku. Returns 1 on success, 0 on failure."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
    }
    payload = {
        "model": HAIKU_MODEL,
        "max_tokens": 512,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Extract a brief summary, key entities, and keywords from "
                    "the following text. Respond with JSON only, with keys: "
                    "summary (string), entities (list of strings), "
                    "keywords (list of strings).\n\n"
                    f"Text:\n{atom.content}"
                ),
            }
        ],
    }

    try:
        response = _http_post_json(ANTHROPIC_URL, headers, payload)
    except Exception:
        logger.warning("Network error enriching atom %s", atom.id, exc_info=True)
        return 0

    # Parse response defensively
    try:
        content_blocks = response.get("content", [])
        if not content_blocks:
            return 0
        text_block = content_blocks[0].get("text", "")
        # Strip markdown fences if present
        text_block = text_block.strip()
        if text_block.startswith("```"):
            lines = text_block.split("\n")
            # Remove first and last fence lines
            text_block = "\n".join(lines[1:-1]) if len(lines) > 2 else text_block
        parsed = json.loads(text_block)
    except (KeyError, IndexError, json.JSONDecodeError, TypeError):
        logger.warning(
            "Could not parse Haiku response for atom %s — leaving un-enriched",
            atom.id,
        )
        return 0

    # Writeback metadata (type-guarded — drop non-conforming values silently)
    meta = dict(atom.metadata)
    summary = parsed.get("summary")
    if isinstance(summary, str):
        meta["summary"] = summary[:1000]
    entities = parsed.get("entities")
    if isinstance(entities, list) and all(isinstance(e, str) for e in entities):
        meta["entities"] = entities
    keywords = parsed.get("keywords")
    if isinstance(keywords, list) and all(isinstance(k, str) for k in keywords):
        meta["keywords"] = keywords
    meta[ENRICHED_FLAG] = True
    meta["enriched_at"] = datetime.now(timezone.utc).isoformat()

    store._conn.execute(
        "UPDATE atoms SET metadata = ? WHERE id = ?",
        (json.dumps(meta), atom.id),
    )
    store._conn.commit()

    return 1
