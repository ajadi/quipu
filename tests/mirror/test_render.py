"""Unit tests for quipu.mirror.render — render_to_md correctness and idempotency."""

import pytest

from quipu.storage import store as _store_factory
from quipu.mirror import render_to_md
from quipu.mirror.render import _normalize_content

_START = "<!-- quipu:start -->"
_END = "<!-- quipu:end -->"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _open_store(tmp_path):
    db = tmp_path / "test.db"
    return _store_factory(str(db))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBasicRender:
    def test_writes_category_file(self, tmp_path):
        """Records for category 'decisions' go to decisions.md."""
        s = _open_store(tmp_path)
        try:
            pid = "proj1"
            s.insert(content="Use SQLite", project_id=pid, metadata={"category": "decisions"})
            s.insert(content="Use argparse", project_id=pid, metadata={"category": "decisions"})

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert result == {"decisions": 2}
        md = (out / "decisions.md").read_text(encoding="utf-8")
        assert _START in md
        assert _END in md
        assert "- Use SQLite" in md
        assert "- Use argparse" in md

    def test_default_category_is_notes(self, tmp_path):
        """Atom with no 'category' metadata goes to notes.md."""
        s = _open_store(tmp_path)
        try:
            pid = "proj2"
            s.insert(content="misc note", project_id=pid, metadata={})

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert "notes" in result
        md = (out / "notes.md").read_text(encoding="utf-8")
        assert "- misc note" in md

    def test_multiple_categories(self, tmp_path):
        """Records in different categories produce separate files."""
        s = _open_store(tmp_path)
        try:
            pid = "proj3"
            s.insert(content="decision 1", project_id=pid, metadata={"category": "decisions"})
            s.insert(content="pattern 1", project_id=pid, metadata={"category": "patterns"})

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert set(result.keys()) == {"decisions", "patterns"}
        assert (out / "decisions.md").exists()
        assert (out / "patterns.md").exists()


class TestIdempotency:
    def test_idempotent_on_new_file(self, tmp_path):
        """Calling render_to_md twice on a new file produces byte-identical output."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-idem"
            s.insert(content="stable content", project_id=pid, metadata={"category": "stack"})

            out = tmp_path / "out"
            render_to_md(pid, out, store=s)
            first = (out / "stack.md").read_bytes()
            render_to_md(pid, out, store=s)
            second = (out / "stack.md").read_bytes()
        finally:
            s.close()

        assert first == second

    def test_idempotent_on_existing_managed_file(self, tmp_path):
        """Calling render_to_md twice on an already-managed file is idempotent."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-idem2"
            s.insert(content="item A", project_id=pid, metadata={"category": "notes"})
            s.insert(content="item B", project_id=pid, metadata={"category": "notes"})

            out = tmp_path / "out"
            render_to_md(pid, out, store=s)
            render_to_md(pid, out, store=s)
            render_to_md(pid, out, store=s)
            first = (out / "notes.md").read_bytes()
            render_to_md(pid, out, store=s)
            second = (out / "notes.md").read_bytes()
        finally:
            s.close()

        assert first == second


class TestInvalidatedExclusion:
    def test_invalidated_atoms_excluded(self, tmp_path):
        """Invalidated atoms must not appear in the mirror output."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-inv"
            a1 = s.insert(content="active item", project_id=pid, metadata={"category": "decisions"})
            a2 = s.insert(content="stale item", project_id=pid, metadata={"category": "decisions"})
            s.update_invalidated(a2.id)

            out = tmp_path / "out"
            render_to_md(pid, out, store=s)
        finally:
            s.close()

        md = (out / "decisions.md").read_text(encoding="utf-8")
        assert "- active item" in md
        assert "stale item" not in md

    def test_all_invalidated_no_file_created(self, tmp_path):
        """If all atoms for a category are invalidated, no file is created."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-all-inv"
            a = s.insert(content="going away", project_id=pid, metadata={"category": "patterns"})
            s.update_invalidated(a.id)

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert result == {}
        assert not (out / "patterns.md").exists()


class TestNonClobber:
    def test_manual_file_untouched_for_unrelated_category(self, tmp_path):
        """A manual file for a category with no Quipu records is not modified."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-nc"
            # Only decisions records; no 'manual' category records.
            s.insert(content="a decision", project_id=pid, metadata={"category": "decisions"})

            out = tmp_path / "out"
            out.mkdir(parents=True, exist_ok=True)
            manual_file = out / "manual.md"
            manual_content = "# Manual notes\n\nThis was written by a human.\n"
            manual_file.write_text(manual_content, encoding="utf-8")

            render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert manual_file.read_text(encoding="utf-8") == manual_content

    def test_manual_content_above_block_preserved(self, tmp_path):
        """Manual content above the Quipu block is preserved after render."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-nc2"
            s.insert(content="decision X", project_id=pid, metadata={"category": "decisions"})

            out = tmp_path / "out"
            out.mkdir(parents=True, exist_ok=True)
            dec_file = out / "decisions.md"
            # Pre-write a file with manual content and a managed block.
            pre_content = (
                "# Decisions\n\nManual intro paragraph.\n\n"
                f"{_START}\n"
                "- old item\n"
                f"{_END}\n"
            )
            dec_file.write_text(pre_content, encoding="utf-8")

            render_to_md(pid, out, store=s)
        finally:
            s.close()

        result = dec_file.read_text(encoding="utf-8")
        # Manual header preserved.
        assert "# Decisions" in result
        assert "Manual intro paragraph." in result
        # New content replaces old.
        assert "- decision X" in result
        assert "old item" not in result
        # Markers present.
        assert _START in result
        assert _END in result

    def test_append_block_when_no_markers(self, tmp_path):
        """When an existing file has no markers, the block is appended."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-append"
            s.insert(content="appended item", project_id=pid, metadata={"category": "stack"})

            out = tmp_path / "out"
            out.mkdir(parents=True, exist_ok=True)
            stack_file = out / "stack.md"
            manual_content = "# Stack\n\nHuman-written content here.\n"
            stack_file.write_text(manual_content, encoding="utf-8")

            render_to_md(pid, out, store=s)
        finally:
            s.close()

        result = stack_file.read_text(encoding="utf-8")
        assert "Human-written content here." in result
        assert "- appended item" in result
        assert _START in result
        assert _END in result
        # Manual content must come before the managed block.
        assert result.index("Human-written content here.") < result.index(_START)


class TestEdgeCases:
    def test_empty_project_no_files_created(self, tmp_path):
        """An empty project produces no files and no crash."""
        s = _open_store(tmp_path)
        try:
            out = tmp_path / "out"
            result = render_to_md("empty-project", out, store=s)
        finally:
            s.close()

        assert result == {}
        assert not list(out.glob("*.md"))

    def test_output_dir_created_if_missing(self, tmp_path):
        """render_to_md creates output_dir if it does not exist."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-mkdir"
            s.insert(content="x", project_id=pid, metadata={"category": "notes"})
            out = tmp_path / "deep" / "nested" / "out"
            render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert out.is_dir()
        assert (out / "notes.md").exists()

    def test_category_sanitization(self, tmp_path):
        """Category names with special chars are sanitized to safe filename stems."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-san"
            s.insert(content="item", project_id=pid, metadata={"category": "Known Issues!"})

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        # "Known Issues!" -> lowercase -> "known issues!" -> strip [^a-z0-9_-] -> "knownissues"
        assert "knownissues" in result
        assert (out / "knownissues.md").exists()

    def test_category_all_special_chars_falls_back_to_notes(self, tmp_path):
        """A category that reduces to empty string after sanitization uses 'notes'."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-fallback"
            s.insert(content="item", project_id=pid, metadata={"category": "!!!"})

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert "notes" in result

    def test_di_store_not_closed(self, tmp_path):
        """When a store is injected, render_to_md does NOT close it."""
        s = _open_store(tmp_path)
        pid = "proj-di"
        s.insert(content="item", project_id=pid, metadata={"category": "notes"})

        out = tmp_path / "out"
        render_to_md(pid, out, store=s)

        # If store was closed this would raise; instead it must still work.
        atoms = s.list_by_project(pid)
        assert len(atoms) == 1
        s.close()

    def test_none_category_metadata_uses_notes(self, tmp_path):
        """atom.metadata['category'] = None falls back to 'notes'."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-none-cat"
            s.insert(content="fallback item", project_id=pid, metadata={"category": None})

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert "notes" in result


class TestSecurityFixes:
    def test_marker_injection_in_content_does_not_corrupt(self, tmp_path):
        """Atom content containing managed-block markers is neutralized.

        Pre-write a decisions.md with a real managed block and manual text
        BELOW the end marker.  Insert atoms whose content contains the literal
        end/start markers.  Run render_to_md twice and verify:
        (a) both runs are byte-identical (idempotent),
        (b) manual text below the end marker is still present,
        (c) exactly ONE real <!-- quipu:start --> and ONE <!-- quipu:end -->
            appear (the injected ones are neutralized, not literal).
        """
        s = _open_store(tmp_path)
        try:
            pid = "proj-inject"
            s.insert(
                content=f"safe text {_END} more text",
                project_id=pid,
                metadata={"category": "decisions"},
            )
            s.insert(
                content=f"another {_START} embedded",
                project_id=pid,
                metadata={"category": "decisions"},
            )

            out = tmp_path / "out"
            out.mkdir(parents=True, exist_ok=True)
            dec_file = out / "decisions.md"
            # Pre-write file with a managed block and manual footer below it.
            pre_content = (
                f"{_START}\n"
                "- old item\n"
                f"{_END}\n"
                "\nManual footer that must survive.\n"
            )
            dec_file.write_text(pre_content, encoding="utf-8")

            render_to_md(pid, out, store=s)
            after_first = dec_file.read_bytes()

            render_to_md(pid, out, store=s)
            after_second = dec_file.read_bytes()
        finally:
            s.close()

        # (a) idempotent
        assert after_first == after_second

        text = after_first.decode("utf-8")

        # (b) manual footer preserved
        assert "Manual footer that must survive." in text

        # (c) exactly one real start and one real end marker
        assert text.count(_START) == 1
        assert text.count(_END) == 1

        # The injected markers must NOT appear literally inside the bullet lines
        lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
        for line in lines:
            assert _END not in line, f"real end marker found in bullet: {line!r}"
            assert _START not in line, f"real start marker found in bullet: {line!r}"

        # The neutralized form MUST appear in the bullet lines that originally
        # contained a marker — proves content was neutralized, not silently dropped.
        neutralized = "<!- -"
        assert any(neutralized in line for line in lines), (
            f"expected neutralized marker {neutralized!r} in at least one bullet line; "
            f"bullet lines: {lines!r}"
        )

    def test_multiline_content_renders_single_line(self, tmp_path):
        """An atom with newlines in content produces a single-line bullet."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-multiline"
            s.insert(
                content="line1\nline2\r\nline3",
                project_id=pid,
                metadata={"category": "notes"},
            )

            out = tmp_path / "out"
            render_to_md(pid, out, store=s)
        finally:
            s.close()

        text = (out / "notes.md").read_text(encoding="utf-8")
        # Find the bullet line(s) - there should be exactly one starting with "- "
        bullet_lines = [ln for ln in text.splitlines() if ln.startswith("- ")]
        assert len(bullet_lines) == 1
        # The content should be on a single line with spaces replacing newlines
        assert "line1" in bullet_lines[0]
        assert "line2" in bullet_lines[0]
        assert "line3" in bullet_lines[0]

    def test_long_category_capped(self, tmp_path):
        """A 200-char category is truncated to 64 chars; no OSError."""
        s = _open_store(tmp_path)
        try:
            pid = "proj-longcat"
            long_cat = "a" * 200
            s.insert(content="item", project_id=pid, metadata={"category": long_cat})

            out = tmp_path / "out"
            result = render_to_md(pid, out, store=s)
        finally:
            s.close()

        assert len(result) == 1
        cat_key = next(iter(result))
        assert len(cat_key) <= 64
        assert (out / f"{cat_key}.md").exists()

    def test_normalize_content_idempotent(self):
        """_normalize_content applied twice equals applying once.

        Also verifies that:
        - the result contains no ``<!--`` (markers are neutralized)
        - the result contains no raw newline (\\n or \\r)
        """
        raw = "a\n<!-- x -->\r\nb"

        once = _normalize_content(raw)
        twice = _normalize_content(once)

        assert once == twice, "applying _normalize_content twice must equal applying once"
        assert "<!--" not in once, "result must not contain literal <!--"
        assert "\n" not in once, "result must not contain raw newline"
        assert "\r" not in once, "result must not contain raw carriage return"
