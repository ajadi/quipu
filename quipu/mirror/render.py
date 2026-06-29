"""Render Quipu records to memory/*.md files (one-way mirror)."""

from __future__ import annotations

import re
from pathlib import Path

_START_MARKER = "<!-- quipu:start -->"
_END_MARKER = "<!-- quipu:end -->"

_SAFE_CAT = re.compile(r"[^a-z0-9_-]")
_DEFAULT_CAT = "notes"


def _sanitize_category(raw: str) -> str:
    """Return a safe filename stem from a raw category string.

    Lowercases, strips whitespace, replaces disallowed chars, falls back to
    'notes' if nothing remains.  Capped at 64 chars to avoid OS filename limits.
    """
    s = raw.strip().lower()
    s = _SAFE_CAT.sub("", s)
    s = s[:64]
    return s if s else _DEFAULT_CAT


def _normalize_content(text: str) -> str:
    """Return a single-line, marker-safe version of atom content.

    Operations (in order, each idempotent):
    1. Flatten any newline sequence to a single space so the bullet stays
       on one line.
    2. Neutralize ``<!--`` so embedded managed-block markers can never be
       mistaken for real boundaries on re-read.  ``<!--`` -> ``<!- -``
    """
    # (a) flatten newlines
    content = text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    # (b) neutralize HTML comment opener
    content = content.replace("<!--", "<!- -")
    return content


def _render_block(atoms: list) -> str:
    """Return the content that goes *between* the markers (exclusive).

    Each atom is rendered as ``- {content}`` with a trailing newline.
    """
    lines = []
    for atom in atoms:
        lines.append(f"- {_normalize_content(atom.content)}")
    return "\n".join(lines) + "\n" if lines else ""


def _managed_block(atoms: list) -> str:
    """Return the full managed block including markers and trailing newline."""
    inner = _render_block(atoms)
    return f"{_START_MARKER}\n{inner}{_END_MARKER}\n"


def _write_file(path: Path, atoms: list) -> None:
    """Write managed block into *path*, preserving manual content."""
    block = _managed_block(atoms)

    if not path.exists():
        path.write_text(block, encoding="utf-8")
        return

    existing = path.read_text(encoding="utf-8")

    start_idx = existing.find(_START_MARKER)
    end_idx = existing.find(_END_MARKER)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        # Replace between markers (exclusive of marker lines).
        # We want to keep everything before _START_MARKER (inclusive) and
        # everything after _END_MARKER (inclusive).
        before = existing[: start_idx + len(_START_MARKER)]
        after = existing[end_idx:]  # includes _END_MARKER and beyond
        inner = _render_block(atoms)
        new_content = before + "\n" + inner + after
        path.write_text(new_content, encoding="utf-8")
        return

    # No markers found: append managed block, preserving existing content.
    if existing and not existing.endswith("\n"):
        existing += "\n"
    path.write_text(existing + block, encoding="utf-8")


def render_to_md(
    project_id: str,
    output_dir: str | Path,
    *,
    store=None,
) -> dict[str, int]:
    """Render active Quipu records for *project_id* into *output_dir*/<cat>.md.

    Args:
        project_id: Quipu project identifier.
        output_dir: Directory where ``<category>.md`` files are written.
        store: Optional injected Store instance.  If None the default store is
               opened and closed in a finally block.

    Returns:
        Mapping ``{category: count_written}`` for each file touched.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    _own_store = store is None
    if _own_store:
        from quipu.storage import store as _store_factory
        store = _store_factory()

    try:
        atoms = store.list_by_project(project_id, include_invalidated=False)
    finally:
        if _own_store:
            store.close()

    # Defensive filter (store should already exclude, but be safe).
    atoms = [a for a in atoms if not a.invalidated]

    # Group by sanitized category; stable sort within each group.
    groups: dict[str, list] = {}
    for atom in atoms:
        raw_cat = atom.metadata.get("category") or _DEFAULT_CAT
        cat = _sanitize_category(raw_cat)
        groups.setdefault(cat, []).append(atom)

    # Sort atoms within each category by created_at then id (ascending).
    for cat in groups:
        groups[cat].sort(key=lambda a: (a.created_at, a.id))

    result: dict[str, int] = {}
    for cat, cat_atoms in groups.items():
        if not cat_atoms:
            continue
        file_path = output_path / f"{cat}.md"
        _write_file(file_path, cat_atoms)
        result[cat] = len(cat_atoms)

    return result
