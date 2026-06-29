"""test_oplog_roundtrip — POST entries then GET them back."""

from __future__ import annotations

import base64

from hub.tests.conftest import BPID, make_entry


def test_push_and_pull_roundtrip(client, auth_headers):
    """Push entries then pull them back; payload byte-identical; cursor advances."""
    payload_bytes = b"opaque-ciphertext-abc"
    entry = make_entry(payload=payload_bytes)

    push_resp = client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers=auth_headers,
    )
    assert push_resp.status_code == 200
    cursor_after_push = push_resp.json()["cursor"]
    assert cursor_after_push != "0"

    pull_resp = client.get(f"/oplog/{BPID}", headers=auth_headers)
    assert pull_resp.status_code == 200
    data = pull_resp.json()
    assert len(data["entries"]) == 1

    got = data["entries"][0]
    assert got["entry_id"] == entry["entry_id"]
    assert got["client_id"] == entry["client_id"]
    assert got["sequence_no"] == entry["sequence_no"]
    assert got["op"] == entry["op"]
    assert got["record_id"] == entry["record_id"]
    assert got["blinded_project_id"] == BPID
    assert got["ts"] == entry["ts"]

    # payload byte-identical
    returned_bytes = base64.b64decode(got["payload"])
    assert returned_bytes == payload_bytes


def test_cursor_advances(client, auth_headers):
    """cursor in push response matches cursor in pull response after consuming all."""
    entry = make_entry()
    push_resp = client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers=auth_headers,
    )
    push_cursor = push_resp.json()["cursor"]

    pull_resp = client.get(f"/oplog/{BPID}", headers=auth_headers)
    pull_cursor = pull_resp.json()["cursor"]

    assert push_cursor == pull_cursor


def test_repush_same_entries_is_noop(client, auth_headers):
    """Re-POST same entries (dedup) is a no-op; count doesn't increase."""
    entry = make_entry()
    client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers=auth_headers,
    )

    # Re-push same entry
    resp2 = client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers=auth_headers,
    )
    assert resp2.status_code == 200

    pull_resp = client.get(f"/oplog/{BPID}", headers=auth_headers)
    # Only 1 entry despite 2 pushes
    assert len(pull_resp.json()["entries"]) == 1


def test_pull_from_cursor_returns_only_new(client, auth_headers):
    """Pull from returned cursor yields empty + unchanged cursor."""
    entry = make_entry()
    client.post(
        f"/oplog/{BPID}",
        json={"entries": [entry]},
        headers=auth_headers,
    )

    # First full pull
    pull1 = client.get(f"/oplog/{BPID}", headers=auth_headers)
    cursor1 = pull1.json()["cursor"]

    # Pull from cursor -> should be empty, cursor unchanged
    pull2 = client.get(f"/oplog/{BPID}", params={"since": cursor1}, headers=auth_headers)
    assert pull2.status_code == 200
    data2 = pull2.json()
    assert data2["entries"] == []
    assert data2["cursor"] == cursor1


def test_push_multiple_entries(client, auth_headers):
    """Push multiple entries; all come back in ingest order."""
    entries = [
        make_entry(entry_id="e" + str(i) + "0" * 62, sequence_no=i, record_id=f"rec-{i}")
        for i in range(1, 4)
    ]
    client.post(f"/oplog/{BPID}", json={"entries": entries}, headers=auth_headers)

    pull = client.get(f"/oplog/{BPID}", headers=auth_headers)
    got = pull.json()["entries"]
    assert len(got) == 3
    for i, e in enumerate(got):
        assert e["entry_id"] == entries[i]["entry_id"]


def test_ingest_seq_not_exposed(client, auth_headers):
    """ingest_seq must never appear in pull response entries."""
    entry = make_entry()
    client.post(f"/oplog/{BPID}", json={"entries": [entry]}, headers=auth_headers)
    pull = client.get(f"/oplog/{BPID}", headers=auth_headers)
    for e in pull.json()["entries"]:
        assert "ingest_seq" not in e
