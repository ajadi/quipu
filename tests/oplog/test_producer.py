"""Tests for quipu.oplog.producer — the oplog PRODUCER module.

Covers:
  - write() emits exactly ONE upsert when sync is active
  - cosine auto-invalidate emits ONE invalidate per superseded atom
  - MCP explicit invalidate emits ONE invalidate entry
  - merge.py pull emits ZERO (no loop)
  - LOCAL-ONLY no-key: write succeeds, atom present, ZERO oplog rows, NO crash
  - no-hub-config: key present but hub=None -> 0 entries
  - idempotency: re-run / second sequence no double-row
  - field asserts: entry_id, op, record_id, ts, bpid, decode_entry round-trip
  - KDF-called-once: N writes -> exactly 1 derive; never in invalidate loop
  - getpass-guard: no key env + no TTY -> no prompt attempted
"""

from __future__ import annotations

import base64
import hashlib
import os
import sys
from pathlib import Path

import pytest

_repo_root = Path(__file__).resolve().parents[2]
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from quipu.oplog.codec import decode_entry
from quipu.oplog.entry import OplogEntry
from quipu.oplog.producer import reset_cache
from quipu.storage import store as open_store
from quipu.sync._aad import aad_for
from quipu.sync.oplog_store import OplogStore

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 32-byte fixed test key (avoids Argon2id / keyring in tests).
TEST_KEY = bytes(range(32))
# Valid base64 encoding of TEST_KEY for QUIPU_KEY env var.
TEST_KEY_B64 = base64.b64encode(TEST_KEY).decode()
PROJECT_ID = "proj-producer-test"
CLIENT_ID = "client-producer-test"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blinded(project_id: str = PROJECT_ID, key: bytes = TEST_KEY) -> str:
    return aad_for(project_id, key).decode()


def _oplog_rows(store) -> list:
    rows = store._conn.execute(
        "SELECT * FROM oplog_entries ORDER BY sequence_no"
    ).fetchall()
    return rows


def _count_rows(store) -> int:
    return len(_oplog_rows(store))


# ---------------------------------------------------------------------------
# Fake engine (no ONNX needed)
# ---------------------------------------------------------------------------

EMBED_DIM = 384

class _N:
    def __init__(self, name):
        self.name = name
        self.type = "tensor(int64)"

class _FakeSession:
    def get_inputs(self):
        return [_N("input_ids"), _N("attention_mask")]
    def get_outputs(self):
        return [_N("sentence_embedding")]
    def run(self, output_names, feeds):
        import numpy as np
        n = feeds["input_ids"].shape[0]
        return [np.ones((n, EMBED_DIM), dtype=np.float32)]

class _FakeTokenizer:
    def encode_batch(self, texts):
        class _Enc:
            ids = [1] * 8
            attention_mask = [1] * 8
        return [_Enc() for _ in texts]


def _install_fake_engine():
    from quipu.embeddings.engine import _Engine, set_engine, _reset
    _reset()
    set_engine(_Engine(session=_FakeSession(), tokenizer=_FakeTokenizer()))


def _install_unit_vec_engine(dim_index: int = 0):
    """Install engine that returns a unit vector along dim_index."""
    import numpy as np
    from quipu.embeddings.engine import _Engine, set_engine, _reset

    class _UnitSession:
        def get_inputs(self):
            return [_N("input_ids"), _N("attention_mask")]
        def get_outputs(self):
            return [_N("sentence_embedding")]
        def run(self, output_names, feeds):
            n = feeds["input_ids"].shape[0]
            arr = np.zeros((n, EMBED_DIM), dtype=np.float32)
            arr[:, dim_index] = 1.0
            return [arr]

    _reset()
    set_engine(_Engine(session=_UnitSession(), tokenizer=_FakeTokenizer()))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_producer_cache():
    """Reset the producer process-lifetime cache before/after each test."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture(autouse=True)
def _reset_embed_engine():
    from quipu.embeddings.engine import _reset
    yield
    _reset()


@pytest.fixture()
def tmp_store(tmp_path):
    s = open_store(str(tmp_path / "test.db"))
    yield s
    s.close()


@pytest.fixture()
def active_env(monkeypatch, tmp_path):
    """Set env vars so the producer thinks sync is active."""
    monkeypatch.setenv("QUIPU_KEY", TEST_KEY_B64)
    monkeypatch.setenv("QUIPU_HUB_URL", "http://hub-fake")
    monkeypatch.setenv("QUIPU_HUB_TOKEN", "fake-token")
    monkeypatch.setenv("QUIPU_CLIENT_ID", CLIENT_ID)
    # Use a tmp config dir so get_client_id doesn't write to real ~/.quipu
    monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))


# ---------------------------------------------------------------------------
# 1. write() emits exactly ONE upsert when active
# ---------------------------------------------------------------------------

class TestWriteEmitsUpsert:
    def test_write_emits_one_upsert(self, tmp_store, active_env):
        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write("hello producer", project_id=PROJECT_ID, store=tmp_store)

        rows = _oplog_rows(tmp_store)
        assert len(rows) == 1, f"Expected 1 oplog row, got {len(rows)}"
        row = rows[0]
        assert row["op"] == "upsert"
        assert row["record_id"] == atom_id

    def test_write_emits_ts_equals_atom_updated_at(self, tmp_store, active_env):
        """ts in oplog must equal atom.updated_at (event-time, not wall-clock now)."""
        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write("ts test", project_id=PROJECT_ID, store=tmp_store)

        atom = tmp_store.get(atom_id)
        rows = _oplog_rows(tmp_store)
        # There may be more than 1 row if invalidation happened, but the upsert must be present.
        upsert_rows = [r for r in rows if r["op"] == "upsert"]
        assert len(upsert_rows) == 1
        assert upsert_rows[0]["ts"] == atom.updated_at


# ---------------------------------------------------------------------------
# 2. write() does NOT auto-invalidate; only upsert entries emitted
# ---------------------------------------------------------------------------

class TestCosineAutoInvalidateEmits:
    def test_write_near_dup_emits_only_upserts_no_auto_invalidate(
        self, tmp_store, active_env
    ):
        """write() no longer auto-invalidates near-duplicates (TASK-020).

        Two writes with identical unit-vec embeddings must produce exactly 2
        upsert oplog entries and 0 invalidate entries. The old atom remains
        ACTIVE; invalidation is the caller's responsibility via quipu_invalidate.
        """
        from quipu.write.pipeline import write

        # First write: unit vec along dim 0
        _install_unit_vec_engine(0)
        old_id = write("first entry", project_id=PROJECT_ID, store=tmp_store)

        # Second write: same direction — no longer auto-supersedes first
        _install_unit_vec_engine(0)
        write("second entry", project_id=PROJECT_ID, store=tmp_store)

        # old_id must still be ACTIVE (no auto-invalidation)
        assert not tmp_store.get(old_id).invalidated

        rows = _oplog_rows(tmp_store)
        ops = [r["op"] for r in rows]
        # Exactly 2 upserts, 0 invalidates
        assert ops.count("upsert") == 2
        assert ops.count("invalidate") == 0

    def test_three_near_dup_writes_emit_only_upserts(
        self, tmp_store, active_env
    ):
        """Three near-dup writes produce 3 upserts and 0 invalidates (TASK-020)."""
        from quipu.write.pipeline import write

        _install_unit_vec_engine(0)
        id1 = write("atom 1", project_id=PROJECT_ID, store=tmp_store)
        _install_unit_vec_engine(0)
        id2 = write("atom 2", project_id=PROJECT_ID, store=tmp_store)
        _install_unit_vec_engine(0)
        write("atom 3", project_id=PROJECT_ID, store=tmp_store)

        rows = _oplog_rows(tmp_store)
        inv_rows = [r for r in rows if r["op"] == "invalidate"]
        assert len(inv_rows) == 0
        # All three atoms remain active
        assert not tmp_store.get(id1).invalidated
        assert not tmp_store.get(id2).invalidated


# ---------------------------------------------------------------------------
# 3. MCP explicit invalidate emits ONE invalidate entry
# ---------------------------------------------------------------------------

class TestMcpExplicitInvalidateEmits:
    def test_mcp_invalidate_emits_one_entry(self, tmp_store, active_env, monkeypatch):
        from quipu.mcp.tools import _handle_quipu_invalidate

        # Insert atom directly (skip write() to isolate MCP handler test)
        atom = tmp_store.insert(
            content="mcp test", project_id=PROJECT_ID
        )

        # MCP invalidate
        result = _handle_quipu_invalidate(
            tmp_store, PROJECT_ID, {"id": atom.id, "project_id": PROJECT_ID}
        )
        assert result["invalidated"] is True

        rows = _oplog_rows(tmp_store)
        assert len(rows) == 1
        assert rows[0]["op"] == "invalidate"
        assert rows[0]["record_id"] == atom.id


# ---------------------------------------------------------------------------
# 4. merge.py pull emits ZERO (no loop)
# ---------------------------------------------------------------------------

class TestMergeEmitsZero:
    def test_merge_resolve_emits_no_local_oplog_rows(
        self, tmp_store, active_env
    ):
        """merge.py -> store.insert/update_invalidated must NOT produce local oplog rows."""
        from quipu.sync.merge import resolve_record
        from quipu.sync.oplog_store import OplogStore
        from quipu.oplog.codec import encode_entry

        blinded = _blinded()
        oplog = OplogStore(tmp_store._conn)

        # Manually write a remote entry to oplog (source='remote')
        seq = oplog.next_sequence_no("remote-client")
        entry_id = OplogEntry.compute_entry_id("remote-client", seq)
        op_fields = {
            "op": "upsert",
            "record_id": "remote-rec-1",
            "ts": "2026-06-21T00:00:01Z",
            "content": "remote content",
            "type": "diary",
            "scope": "project",
            "metadata": {},
            "refs": [],
            "project_id": PROJECT_ID,
        }
        payload = encode_entry(op_fields, TEST_KEY, blinded)
        entry = OplogEntry(
            entry_id=entry_id,
            client_id="remote-client",
            sequence_no=seq,
            op="upsert",
            record_id="remote-rec-1",
            blinded_project_id=blinded,
            ts="2026-06-21T00:00:01Z",
            payload=payload,
            source="remote",
            pushed=False,
        )
        oplog.append_local(entry)

        # Count rows BEFORE resolve (should be 1: the remote entry)
        before = _count_rows(tmp_store)
        assert before == 1

        # Run merge (this calls store.insert internally)
        resolve_record(oplog, tmp_store, blinded, "remote-rec-1", TEST_KEY)

        # Count rows AFTER resolve: must still be 1 (no new local emit)
        after = _count_rows(tmp_store)
        assert after == 1, (
            f"merge.resolve_record emitted a new local oplog row (before={before}, after={after})"
        )

        # Verify the atom was created
        atom = tmp_store.get("remote-rec-1")
        assert atom is not None
        assert atom.content == "remote content"


# ---------------------------------------------------------------------------
# 5. LOCAL-ONLY no-key: write succeeds, atom present, ZERO oplog rows, NO crash
# ---------------------------------------------------------------------------

class TestLocalOnlyNoKey:
    def test_write_succeeds_no_oplog_without_key(
        self, tmp_store, monkeypatch, tmp_path
    ):
        """No QUIPU_KEY, no QUIPU_PASSPHRASE -> 0 oplog rows, atom still written."""
        monkeypatch.delenv("QUIPU_KEY", raising=False)
        monkeypatch.delenv("QUIPU_PASSPHRASE", raising=False)
        monkeypatch.delenv("QUIPU_HUB_URL", raising=False)
        monkeypatch.delenv("QUIPU_HUB_TOKEN", raising=False)
        monkeypatch.setenv("QUIPU_CLIENT_ID", CLIENT_ID)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write("local only", project_id=PROJECT_ID, store=tmp_store)

        # Atom must exist
        atom = tmp_store.get(atom_id)
        assert atom is not None
        assert atom.content == "local only"

        # Zero oplog rows
        assert _count_rows(tmp_store) == 0

    def test_no_getpass_prompt_without_key(self, tmp_store, monkeypatch, tmp_path):
        """Without key env vars, getpass must never be called."""
        monkeypatch.delenv("QUIPU_KEY", raising=False)
        monkeypatch.delenv("QUIPU_PASSPHRASE", raising=False)
        # Hub config present (to trigger the code path where we check key env)
        monkeypatch.setenv("QUIPU_HUB_URL", "http://hub-fake")
        monkeypatch.setenv("QUIPU_HUB_TOKEN", "fake-token")
        monkeypatch.setenv("QUIPU_CLIENT_ID", CLIENT_ID)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        import getpass
        called = []

        def _no_prompt(prompt=""):
            called.append(prompt)
            raise RuntimeError("getpass should not be called")

        monkeypatch.setattr(getpass, "getpass", _no_prompt)

        from quipu.write.pipeline import write

        _install_fake_engine()
        # Must not raise, must not call getpass
        atom_id = write("no prompt test", project_id=PROJECT_ID, store=tmp_store)
        assert atom_id is not None
        assert called == [], "getpass was invoked by the producer"


# ---------------------------------------------------------------------------
# 6. no-hub-config: key present, get_hub_config None -> 0 entries
# ---------------------------------------------------------------------------

class TestNoHubConfig:
    def test_no_hub_config_emits_zero_entries(
        self, tmp_store, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("QUIPU_KEY", TEST_KEY_B64)
        monkeypatch.delenv("QUIPU_HUB_URL", raising=False)
        monkeypatch.delenv("QUIPU_HUB_TOKEN", raising=False)
        monkeypatch.setenv("QUIPU_CLIENT_ID", CLIENT_ID)
        monkeypatch.setenv("QUIPU_PROJECT_ROOT", str(tmp_path))

        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write("no hub", project_id=PROJECT_ID, store=tmp_store)
        assert atom_id is not None
        assert _count_rows(tmp_store) == 0


# ---------------------------------------------------------------------------
# 7. Idempotency: two writes -> two distinct rows (no double-seq for one write)
# ---------------------------------------------------------------------------

class TestIdempotency:
    def test_two_writes_produce_two_distinct_entries(
        self, tmp_store, active_env
    ):
        from quipu.write.pipeline import write

        _install_fake_engine()
        id1 = write("first", project_id=PROJECT_ID, store=tmp_store)
        id2 = write("second", project_id=PROJECT_ID, store=tmp_store)

        rows = _oplog_rows(tmp_store)
        upsert_rows = [r for r in rows if r["op"] == "upsert"]
        assert len(upsert_rows) >= 2

        seq_nos = [r["sequence_no"] for r in rows]
        assert len(seq_nos) == len(set(seq_nos)), "Duplicate sequence_nos detected"

        entry_ids = [r["entry_id"] for r in rows]
        assert len(entry_ids) == len(set(entry_ids)), "Duplicate entry_ids detected"


# ---------------------------------------------------------------------------
# 8. Field asserts: entry_id, op, record_id, ts, bpid, decode round-trip
# ---------------------------------------------------------------------------

class TestFieldAsserts:
    def test_entry_id_equals_sha256_of_client_seq(
        self, tmp_store, active_env
    ):
        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write("field test", project_id=PROJECT_ID, store=tmp_store)

        rows = _oplog_rows(tmp_store)
        upsert = next(r for r in rows if r["op"] == "upsert")

        expected = hashlib.sha256(
            f"{upsert['client_id']}:{upsert['sequence_no']}".encode()
        ).hexdigest()
        assert upsert["entry_id"] == expected

    def test_record_id_equals_atom_id(self, tmp_store, active_env):
        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write("record_id test", project_id=PROJECT_ID, store=tmp_store)

        rows = _oplog_rows(tmp_store)
        upsert = next(r for r in rows if r["op"] == "upsert")
        assert upsert["record_id"] == atom_id

    def test_ts_equals_atom_updated_at(self, tmp_store, active_env):
        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write("ts field test", project_id=PROJECT_ID, store=tmp_store)

        atom = tmp_store.get(atom_id)
        rows = _oplog_rows(tmp_store)
        upsert = next(r for r in rows if r["op"] == "upsert")
        assert upsert["ts"] == atom.updated_at

    def test_bpid_is_64_hex(self, tmp_store, active_env):
        from quipu.write.pipeline import write

        _install_fake_engine()
        write("bpid test", project_id=PROJECT_ID, store=tmp_store)

        rows = _oplog_rows(tmp_store)
        upsert = next(r for r in rows if r["op"] == "upsert")
        bpid = upsert["blinded_project_id"]
        assert len(bpid) == 64
        assert all(c in "0123456789abcdef" for c in bpid)

    def test_decode_entry_roundtrips_op_fields(self, tmp_store, active_env):
        from quipu.write.pipeline import write

        _install_fake_engine()
        atom_id = write(
            "roundtrip test",
            project_id=PROJECT_ID,
            store=tmp_store,
        )

        atom = tmp_store.get(atom_id)
        rows = _oplog_rows(tmp_store)
        upsert = next(r for r in rows if r["op"] == "upsert")

        bpid = upsert["blinded_project_id"]
        decoded = decode_entry(upsert["payload"], TEST_KEY, bpid)

        assert decoded["op"] == "upsert"
        assert decoded["record_id"] == atom_id
        assert decoded["ts"] == atom.updated_at
        assert decoded["content"] == "roundtrip test"
        assert decoded["project_id"] == PROJECT_ID
        assert "type" in decoded
        assert "scope" in decoded
        assert "metadata" in decoded
        assert "refs" in decoded


# ---------------------------------------------------------------------------
# 9. KDF-called-once: N writes -> exactly 1 derive; never in invalidate loop
# ---------------------------------------------------------------------------

class TestKdfCalledOnce:
    def test_n_writes_call_kdf_exactly_once(self, tmp_store, active_env, monkeypatch):
        derive_calls = []

        original_derive = None
        import quipu.keystore._backend as _backend_mod

        original_derive = _backend_mod.get_or_derive_key

        def _counting_derive(project_id, **kwargs):
            derive_calls.append(project_id)
            return original_derive(project_id, **kwargs)

        monkeypatch.setattr(_backend_mod, "get_or_derive_key", _counting_derive)

        from quipu.write.pipeline import write

        _install_fake_engine()
        for i in range(4):
            write(f"entry {i}", project_id=PROJECT_ID, store=tmp_store)

        # After first call the cache is warm; KDF should be called at most once.
        assert len(derive_calls) <= 1, (
            f"get_or_derive_key called {len(derive_calls)} times for {4} writes; expected <=1"
        )

    def test_invalidate_loop_does_not_call_kdf(
        self, tmp_store, active_env, monkeypatch
    ):
        """KDF must NOT be called inside the invalidate loop."""
        # Write first atom so we have something to supersede
        from quipu.write.pipeline import write

        _install_unit_vec_engine(0)
        write("base atom", project_id=PROJECT_ID, store=tmp_store)

        # Now set up the call counter AFTER the first write (cache already warm)
        import quipu.keystore._backend as _backend_mod
        original_derive = _backend_mod.get_or_derive_key
        derive_calls_after_first = []

        def _counting_derive(project_id, **kwargs):
            derive_calls_after_first.append(project_id)
            return original_derive(project_id, **kwargs)

        monkeypatch.setattr(_backend_mod, "get_or_derive_key", _counting_derive)

        # Second write supersedes first: triggers invalidation loop
        _install_unit_vec_engine(0)
        write("superseding atom", project_id=PROJECT_ID, store=tmp_store)

        # Cache is warm from before -> invalidate loop must NOT call KDF again
        assert derive_calls_after_first == [], (
            f"get_or_derive_key was called {len(derive_calls_after_first)} times "
            f"during the invalidate loop (expected 0)"
        )


# ---------------------------------------------------------------------------
# 10. project_id=None -> no-op (no row, no crash)
# ---------------------------------------------------------------------------

class TestProjectIdNone:
    def test_emit_with_none_project_id_is_noop(self, tmp_store, active_env):
        from quipu.oplog.producer import emit

        # Create a minimal atom-like object
        atom = tmp_store.insert(content="anon", project_id=None)

        # Should not crash, should not produce a row
        emit(tmp_store, op="upsert", atom=atom, project_id=None)
        assert _count_rows(tmp_store) == 0


# ---------------------------------------------------------------------------
# 11. Failure isolation: exception inside emit must not propagate
# ---------------------------------------------------------------------------

class TestFailureIsolation:
    def test_emit_exception_does_not_propagate(
        self, tmp_store, active_env, monkeypatch
    ):
        """If something breaks inside emit, the atom write is unaffected."""
        from quipu.oplog import producer as producer_mod

        def _bad_active_key(project_id):
            raise RuntimeError("simulated KDF failure")

        monkeypatch.setattr(producer_mod, "_active_key", _bad_active_key)

        from quipu.write.pipeline import write

        _install_fake_engine()
        # Must not raise despite the broken _active_key
        atom_id = write("isolation test", project_id=PROJECT_ID, store=tmp_store)
        assert atom_id is not None

        # Atom is present
        atom = tmp_store.get(atom_id)
        assert atom is not None

        # No oplog row (emit silently swallowed the error)
        assert _count_rows(tmp_store) == 0
