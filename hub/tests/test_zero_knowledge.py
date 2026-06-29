"""test_zero_knowledge — structural ZK checks.

1. No module under hub/ imports quipu (walk hub/ runtime .py, parse imports).
2. Stored+retrieved payload is byte-identical (hub round-trips opaque bytes).
3. DB row stores raw bytes; response re-encodes identically.
"""

from __future__ import annotations

import ast
import base64
import importlib
import os
import sqlite3

import pytest

from hub.tests.conftest import BPID, make_entry


# ---------------------------------------------------------------------------
# 1. Structural: no quipu import in hub runtime code
# ---------------------------------------------------------------------------

def _hub_runtime_files() -> list[str]:
    """Collect all .py files under hub/ excluding tests/."""
    hub_root = os.path.join(os.path.dirname(__file__), "..")
    result = []
    for dirpath, dirnames, filenames in os.walk(hub_root):
        # Exclude the tests directory
        dirnames[:] = [d for d in dirnames if d != "tests"]
        for fname in filenames:
            if fname.endswith(".py"):
                result.append(os.path.join(dirpath, fname))
    return result


def _get_imports(filepath: str) -> list[str]:
    """Return list of top-level module names imported by the file."""
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    try:
        tree = ast.parse(source, filename=filepath)
    except SyntaxError:
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module.split(".")[0])
    return imports


def test_no_quipu_import_in_hub_runtime():
    """hub/ runtime code must never import quipu.*."""
    violations = []
    for filepath in _hub_runtime_files():
        imports = _get_imports(filepath)
        if "quipu" in imports:
            violations.append(filepath)

    assert violations == [], (
        f"hub/ runtime files import quipu: {violations}\n"
        "The hub must be zero-knowledge and self-contained."
    )


def _strip_comments_and_docstrings(source: str, filepath: str) -> str:
    """Return source with string literals and comments replaced by whitespace.

    Uses tokenize so the character positions are preserved (line numbers stay
    intact for diagnostics).  Falls back to the raw source on parse error.
    """
    import io
    import tokenize

    result = list(source)
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except tokenize.TokenError:
        return source  # fail-open: return raw source

    for tok_type, tok_string, (srow, scol), (erow, ecol), _ in tokens:
        if tok_type not in (tokenize.STRING, tokenize.COMMENT):
            continue
        # Blank out every character of this token
        lines = source.splitlines(keepends=True)
        # Collect absolute char offsets for each line start
        line_starts = []
        offset = 0
        for line in lines:
            line_starts.append(offset)
            offset += len(line)

        # Replace with spaces
        for row in range(srow, erow + 1):
            line_start = line_starts[row - 1]
            col_start = scol if row == srow else 0
            col_end = ecol if row == erow else len(lines[row - 1])
            for idx in range(line_start + col_start, line_start + col_end):
                if idx < len(result):
                    result[idx] = " "

    return "".join(result)


def test_no_crypto_in_hub_runtime():
    """hub/ runtime code must not contain real crypto calls/identifiers.

    Forbidden identifiers (word-boundary matched, after stripping comments and
    docstrings so prose cannot trigger false positives):

      decrypt, decode_entry, derive_key, AESGCM, Fernet, argon2,
      blind_project_id  (the FUNCTION — underscore before 'project'),
      encrypt_record, encode_entry

    NOT forbidden:
      blinded_project_id  (the partition-key field — legitimate hub metadata)
      encrypt              (prose word in a comment/docstring — stripped)
    """
    import re

    # Forbidden as whole identifiers (word boundaries).
    # Note: "blind_project_id" matches the helper function; it does NOT match
    # "blinded_project_id" because \bblind_project_id\b cannot appear inside
    # "blinded_project_id" (the character before "b" is "d", not a word boundary
    # in the word "blinded_project_id").  Use explicit negative-lookbehind to be
    # extra safe.
    forbidden_patterns = [
        r"\bdecrypt\b",
        r"\bdecode_entry\b",
        r"\bderive_key\b",
        r"\bAESGCM\b",
        r"\bFernet\b",
        r"\bargon2\b",
        r"(?<!\w)blind_project_id\b",   # function; NOT "blinded_project_id"
        r"\bencrypt_record\b",
        r"\bencode_entry\b",
    ]
    combined = re.compile("|".join(forbidden_patterns))

    violations = []
    for filepath in _hub_runtime_files():
        with open(filepath, encoding="utf-8") as f:
            source = f.read()
        stripped = _strip_comments_and_docstrings(source, filepath)
        found = combined.findall(stripped)
        if found:
            violations.append((filepath, found))

    assert violations == [], (
        f"hub/ runtime files contain forbidden crypto identifiers: {violations}"
    )


def test_no_crypto_in_hub_runtime_catches_real_violations():
    """Verify the scanner is not a no-op: a fake decrypt( call IS caught.

    This test injects a synthetic source string (never written to disk) and
    asserts the forbidden-pattern logic flags it.  Proves the whitelist did not
    accidentally swallow all patterns.
    """
    import re

    fake_source = "result = decrypt(ciphertext, key)\n"

    forbidden_patterns = [
        r"\bdecrypt\b",
        r"\bdecode_entry\b",
        r"\bderive_key\b",
        r"\bAESGCM\b",
        r"\bFernet\b",
        r"\bargon2\b",
        r"(?<!\w)blind_project_id\b",
        r"\bencrypt_record\b",
        r"\bencode_entry\b",
    ]
    combined = re.compile("|".join(forbidden_patterns))
    assert combined.search(fake_source), (
        "Scanner must catch 'decrypt(' — pattern list is broken"
    )


# ---------------------------------------------------------------------------
# 2. Payload byte-identical round-trip
# ---------------------------------------------------------------------------

def test_payload_round_trip_byte_identical(client, auth_headers):
    """Push opaque bytes; pull back; verify byte-for-byte identity."""
    raw_payload = bytes(range(256))  # all byte values
    entry = make_entry(payload=raw_payload)

    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)
    resp = client.get(f"/oplog/{BPID}", headers=auth_headers)

    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert len(entries) == 1

    returned = base64.b64decode(entries[0]["payload"])
    assert returned == raw_payload


# ---------------------------------------------------------------------------
# 3. DB row stores raw bytes; re-encoded identically
# ---------------------------------------------------------------------------

def test_db_stores_raw_bytes(client, auth_headers, tmp_db):
    """Verify the DB BLOB contains the raw decoded bytes, not base64."""
    raw_payload = b"\xDE\xAD\xBE\xEF\x00\x01\x02\x03"
    entry = make_entry(payload=raw_payload)

    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)

    # Direct DB inspection
    conn = sqlite3.connect(tmp_db)
    row = conn.execute("SELECT payload FROM hub_oplog LIMIT 1").fetchone()
    conn.close()

    assert row is not None
    assert isinstance(row[0], bytes), "payload column must be BLOB (bytes)"
    assert row[0] == raw_payload


def test_response_base64_matches_original(client, auth_headers):
    """base64 in response is identical to what was pushed."""
    raw_payload = b"test-payload-abc"
    b64_in = base64.b64encode(raw_payload).decode("ascii")
    entry = make_entry(payload=raw_payload)

    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)
    resp = client.get(f"/oplog/{BPID}", headers=auth_headers)

    b64_out = resp.json()["entries"][0]["payload"]
    assert b64_out == b64_in
