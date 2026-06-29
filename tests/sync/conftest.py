"""conftest.py for tests/sync — importability + shared sync fixtures.

Uses a fixed 32-byte test key + InMemoryBackend so no OS keyring and no Argon2id
run in the hot loop. The ONNX embedding model is never required (atoms inserted
with embedding=None).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.oplog.codec import encode_entry
from quipu.oplog.entry import OplogEntry
from quipu.storage import store as open_store
from quipu.sync._aad import aad_for
from quipu.sync.oplog_store import OplogStore
from quipu.sync.transport import InMemoryTransport

# Fixed test key — bypasses Argon2id / keyring entirely.
TEST_KEY = bytes(range(32))
PROJECT_ID = "proj-sync-test"


@pytest.fixture()
def key():
    return TEST_KEY


@pytest.fixture()
def project_id():
    return PROJECT_ID


@pytest.fixture()
def blinded(key, project_id):
    return aad_for(project_id, key).decode()


@pytest.fixture()
def make_store(tmp_path):
    """Factory: build N independent temp Stores (replicas) sharing one schema."""
    created = []

    def _make(name: str):
        s = open_store(str(tmp_path / f"{name}.db"))
        created.append(s)
        return s

    yield _make
    for s in created:
        s.close()


@pytest.fixture()
def transport():
    return InMemoryTransport()


def write_local(store, key, project_id, client_id, *, op, record_id, ts, content=""):
    """Full local-write path: next seq -> entry_id -> encode -> append_local.

    Returns the OplogEntry that was appended.
    """
    blinded = aad_for(project_id, key).decode()
    oplog = OplogStore(store._conn)
    seq = oplog.next_sequence_no(client_id)
    entry_id = OplogEntry.compute_entry_id(client_id, seq)
    op_fields = {
        "op": op,
        "record_id": record_id,
        "ts": ts,
        "content": content,
        "type": "diary",
        "scope": "project",
        "metadata": {},
        "refs": [],
        "project_id": project_id,
    }
    payload = encode_entry(op_fields, key, blinded)
    entry = OplogEntry(
        entry_id=entry_id,
        client_id=client_id,
        sequence_no=seq,
        op=op,
        record_id=record_id,
        blinded_project_id=blinded,
        ts=ts,
        payload=payload,
    )
    oplog.append_local(entry)
    return entry
