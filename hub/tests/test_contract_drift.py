"""test_contract_drift — integration contract-drift guard.

Build entry dicts the same shape the real producer emits (quipu.oplog
to_transport_dict), but WITHOUT importing quipu — hand-construct the 8 fields.

Asserts:
  (a) Every field round-trips verbatim including payload bytes.
  (b) next_cursor is monotonically increasing and non-overlapping across two
      sequential pulls.
  (c) Entry order preserved (ingest order).
"""

from __future__ import annotations

import base64

from hub.tests.conftest import BPID, make_entry


def _producer_shaped_entry(
    *,
    idx: int,
    client_id: str = "device-abc123",
    payload_bytes: bytes | None = None,
) -> dict:
    """Hand-construct an entry with the same 8 wire fields as OplogEntry.

    field shape from quipu/oplog/entry.py (read for reference, not imported):
      entry_id: str (64-hex SHA-256)
      client_id: str
      sequence_no: int
      op: str ('upsert' | 'invalidate')
      record_id: str
      blinded_project_id: str (64-hex)
      ts: str (ISO-8601 UTC)
      payload: bytes -> base64 on wire
    """
    import hashlib
    # Replicate entry_id = SHA-256(f"{client_id}:{sequence_no}") without importing quipu
    entry_id = hashlib.sha256(f"{client_id}:{idx}".encode()).hexdigest()

    if payload_bytes is None:
        # Realistic opaque ciphertext bytes (not real encrypted data)
        payload_bytes = bytes([idx % 256, 0x01, 0x02, 0x03, 0xAB, 0xCD]) * 4

    return {
        "entry_id": entry_id,
        "client_id": client_id,
        "sequence_no": idx,
        "op": "upsert",
        "record_id": f"atom-{idx:04d}",
        "blinded_project_id": BPID,
        "ts": f"2026-01-{idx:02d}T12:00:00Z",
        "payload": base64.b64encode(payload_bytes).decode("ascii"),
    }


def test_all_fields_roundtrip_verbatim(client, auth_headers):
    """(a) Every field round-trips verbatim including payload bytes."""
    entries = [_producer_shaped_entry(idx=i) for i in range(1, 4)]

    push_resp = client.post(
        f"/oplog/{BPID}",
        json={"entries": entries},
        headers=auth_headers,
    )
    assert push_resp.status_code == 200

    pull_resp = client.get(f"/oplog/{BPID}", headers=auth_headers)
    assert pull_resp.status_code == 200
    got = pull_resp.json()["entries"]
    assert len(got) == 3

    for orig, returned in zip(entries, got):
        assert returned["entry_id"] == orig["entry_id"]
        assert returned["client_id"] == orig["client_id"]
        assert returned["sequence_no"] == orig["sequence_no"]
        assert returned["op"] == orig["op"]
        assert returned["record_id"] == orig["record_id"]
        assert returned["blinded_project_id"] == BPID
        assert returned["ts"] == orig["ts"]
        # payload byte-identical
        assert base64.b64decode(returned["payload"]) == base64.b64decode(orig["payload"])


def test_cursor_monotonically_increasing_non_overlapping(client, auth_headers):
    """(b) cursor is monotonically increasing; second pull from cursor returns only new entries."""
    # Push first batch
    batch1 = [_producer_shaped_entry(idx=i) for i in range(1, 4)]
    client.post(f"/oplog/{BPID}", json={"entries": batch1}, headers=auth_headers)

    # First pull — returns all 3
    pull1 = client.get(f"/oplog/{BPID}", headers=auth_headers)
    data1 = pull1.json()
    assert len(data1["entries"]) == 3
    cursor1 = data1["cursor"]

    # Push second batch
    batch2 = [_producer_shaped_entry(idx=i) for i in range(4, 7)]
    client.post(f"/oplog/{BPID}", json={"entries": batch2}, headers=auth_headers)

    # Pull from cursor1 -> returns ONLY the new 3
    pull2 = client.get(f"/oplog/{BPID}", params={"since": cursor1}, headers=auth_headers)
    data2 = pull2.json()
    assert len(data2["entries"]) == 3, "Second pull should return only new entries"
    cursor2 = data2["cursor"]

    # Cursors are monotonically increasing (format-agnostic check)
    assert cursor2 != cursor1, "cursor must advance after new entries"

    # Prove cursor2 is at/after the end — re-pull from cursor2 must be empty
    pull3 = client.get(f"/oplog/{BPID}", params={"since": cursor2}, headers=auth_headers)
    assert pull3.status_code == 200
    assert pull3.json()["entries"] == [], "re-pull from cursor2 must be empty (cursor2 at end of log)"

    # Verify no overlap: entry_ids in second pull are all from batch2
    batch2_ids = {e["entry_id"] for e in batch2}
    batch1_ids = {e["entry_id"] for e in batch1}
    for e in data2["entries"]:
        assert e["entry_id"] in batch2_ids, "Second pull contains entry from first batch"
        assert e["entry_id"] not in batch1_ids


def test_entry_order_preserved(client, auth_headers):
    """(c) Entries are returned in ingest order."""
    entries = [_producer_shaped_entry(idx=i) for i in range(1, 6)]
    client.post(f"/oplog/{BPID}", json={"entries": entries}, headers=auth_headers)

    pull = client.get(f"/oplog/{BPID}", headers=auth_headers)
    got = pull.json()["entries"]

    original_ids = [e["entry_id"] for e in entries]
    returned_ids = [e["entry_id"] for e in got]
    assert returned_ids == original_ids, "Entries must be returned in ingest order"


def test_cursor_unchanged_on_empty_pull(client, auth_headers):
    """After consuming all entries, pulling again returns empty with same cursor."""
    entries = [_producer_shaped_entry(idx=1)]
    client.post(f"/oplog/{BPID}", json={"entries": entries}, headers=auth_headers)

    pull1 = client.get(f"/oplog/{BPID}", headers=auth_headers)
    cursor1 = pull1.json()["cursor"]

    pull2 = client.get(f"/oplog/{BPID}", params={"since": cursor1}, headers=auth_headers)
    assert pull2.json()["entries"] == []
    assert pull2.json()["cursor"] == cursor1  # unchanged
