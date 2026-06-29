"""test_audit — audit log written per push and per pull; metadata only."""

from __future__ import annotations

import json

from hub.tests.conftest import BPID, VALID_TOKEN, VALID_TOKEN_HASH, make_entry


def _read_audit(audit_path: str) -> list[dict]:
    with open(audit_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def test_push_writes_audit_line(client, auth_headers, tmp_audit):
    entry = make_entry()
    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)

    lines = _read_audit(tmp_audit)
    assert len(lines) == 1
    line = lines[0]
    assert line["op"] == "push"
    assert line["blinded_project_id"] == BPID
    assert line["token_hash"] == VALID_TOKEN_HASH
    assert line["status"] == 200
    assert line["entry_count"] == 1
    assert "byte_count" in line


def test_pull_writes_audit_line(client, auth_headers, tmp_audit):
    client.get(f"/oplog/{BPID}", headers=auth_headers)

    lines = _read_audit(tmp_audit)
    assert len(lines) == 1
    line = lines[0]
    assert line["op"] == "pull"
    assert line["blinded_project_id"] == BPID


def test_push_and_pull_write_separate_lines(client, auth_headers, tmp_audit):
    entry = make_entry()
    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)
    client.get(f"/oplog/{BPID}", headers=auth_headers)

    lines = _read_audit(tmp_audit)
    assert len(lines) == 2
    ops = {l["op"] for l in lines}
    assert ops == {"push", "pull"}


def test_audit_does_not_contain_raw_token(client, auth_headers, tmp_audit):
    """Raw token must never appear in the audit log."""
    entry = make_entry()
    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)

    with open(tmp_audit, encoding="utf-8") as f:
        content = f.read()

    assert VALID_TOKEN not in content


def test_audit_does_not_contain_payload(client, auth_headers, tmp_audit):
    """Payload bytes must never appear in the audit log."""
    import base64
    payload_bytes = b"secret-ciphertext-xyz"
    entry = make_entry(payload=payload_bytes)
    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)

    with open(tmp_audit, encoding="utf-8") as f:
        content = f.read()

    # Neither the raw bytes nor the base64 encoding
    assert base64.b64encode(payload_bytes).decode() not in content
    assert "secret-ciphertext" not in content


def test_audit_does_not_contain_plaintext(client, auth_headers, tmp_audit):
    """No plaintext content in audit — only metadata fields."""
    entry = make_entry(record_id="test-record-001")
    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)

    lines = _read_audit(tmp_audit)
    for line in lines:
        # Audit fields: ts, token_hash, blinded_project_id, op, entry_count, byte_count, status
        allowed = {"ts", "token_hash", "blinded_project_id", "op", "entry_count", "byte_count", "status"}
        assert set(line.keys()) == allowed, f"Unexpected audit fields: {set(line.keys()) - allowed}"
