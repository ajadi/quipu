"""CLI receipts and garbage-collection regressions."""

from types import SimpleNamespace

from quipu.cli import cmd_gc, cmd_receipts


def _store_must_not_open(_db_path):
    raise AssertionError("store must not open")


def test_receipts_requires_explicit_project_id_before_opening_store(monkeypatch, capsys):
    monkeypatch.delenv("QUIPU_PROJECT_ID", raising=False)
    monkeypatch.setattr(
        "quipu.storage.store",
        _store_must_not_open,
    )

    rc = cmd_receipts(None, None, None, "json", None)

    assert rc == 1
    assert "project_id required" in capsys.readouterr().err


def test_gc_requires_explicit_project_id_before_opening_store(monkeypatch, capsys):
    monkeypatch.delenv("QUIPU_PROJECT_ID", raising=False)
    monkeypatch.setattr("quipu.storage.store", _store_must_not_open)

    rc = cmd_gc(None, None, False, 90, 3)

    assert rc == 1
    assert "project_id required" in capsys.readouterr().err


class _EmptyStore:
    def __init__(self):
        self.project_ids = []
        self.closed = False

    def list_stale(self, *, project_id, min_age_days, min_access_count):
        self.project_ids.append(project_id)
        return []

    def close(self):
        self.closed = True


class _StaleStore(_EmptyStore):
    def __init__(self):
        super().__init__()
        self.updated_ids = []

    def list_stale(self, **kwargs):
        self.project_ids.append(kwargs["project_id"])
        return [
            SimpleNamespace(
                id="a" * 16,
                access_count=0,
                last_accessed=None,
                created_at="2026-01-01T00:00:00Z",
                content="stale",
            )
        ]

    def update_invalidated(self, atom_id, invalidated):
        self.updated_ids.append((atom_id, invalidated))

    def get(self, atom_id):
        return SimpleNamespace(
            id=atom_id,
            project_id=self.project_ids[-1],
            invalidated=True,
            updated_at="2026-01-02T00:00:00Z",
        )


def test_gc_explicit_argument_takes_precedence_over_environment(monkeypatch):
    store = _EmptyStore()
    monkeypatch.setenv("QUIPU_PROJECT_ID", "from-environment")
    monkeypatch.setattr("quipu.storage.store", lambda _db_path: store)

    assert cmd_gc(None, "from-argument", False, 90, 3) == 0
    assert store.project_ids == ["from-argument"]
    assert store.closed


def test_gc_uses_explicit_project_id_environment(monkeypatch):
    store = _EmptyStore()
    monkeypatch.setenv("QUIPU_PROJECT_ID", "from-environment")
    monkeypatch.setattr("quipu.storage.store", lambda _db_path: store)

    assert cmd_gc(None, None, False, 90, 3) == 0
    assert store.project_ids == ["from-environment"]
    assert store.closed


def test_gc_mutates_only_with_apply(monkeypatch):
    store = _StaleStore()
    monkeypatch.setattr("quipu.storage.store", lambda _db_path: store)

    assert cmd_gc(None, "explicit", False, 90, 3) == 0
    assert store.updated_ids == []

    assert cmd_gc(None, "explicit", True, 90, 3) == 0
    assert store.updated_ids == [("a" * 16, True)]


def test_gc_apply_emits_invalidations(monkeypatch):
    store = _StaleStore()
    emitted = []
    monkeypatch.setattr("quipu.storage.store", lambda _db_path: store)
    monkeypatch.setattr(
        "quipu.oplog.producer.emit",
        lambda source, **kwargs: emitted.append((source, kwargs)),
    )

    assert cmd_gc(None, "explicit", True, 90, 3) == 0

    assert len(emitted) == 1
    source, entry = emitted[0]
    assert source is store
    assert entry["op"] == "invalidate"
    assert entry["project_id"] == "explicit"
    assert entry["atom"].id == "a" * 16


class _ReceiptStore:
    _conn = object()

    def close(self):
        pass


class _EmptyOplog:
    def __init__(self, conn):
        assert conn is _ReceiptStore._conn

    def list_entries_by_project(self, blinded, *, op, limit):
        assert blinded == "blinded"
        return []


def test_receipts_uses_explicit_argument_before_environment(monkeypatch, capsys):
    requested_pids = []
    monkeypatch.setenv("QUIPU_PROJECT_ID", "from-environment")
    monkeypatch.setattr("quipu.storage.store", lambda _db_path: _ReceiptStore())
    monkeypatch.setattr(
        "quipu.keystore._backend.get_or_derive_key",
        lambda pid: requested_pids.append(pid) or b"key",
    )
    monkeypatch.setattr("quipu.sync._aad.aad_for", lambda _pid, _key: b"blinded")
    monkeypatch.setattr("quipu.sync.oplog_store.OplogStore", _EmptyOplog)

    assert cmd_receipts(None, "from-argument", None, "json", None) == 0
    assert requested_pids == ["from-argument"]
    assert capsys.readouterr().out.strip() == "[]"


def test_receipts_uses_explicit_project_id_environment(monkeypatch, capsys):
    requested_pids = []
    monkeypatch.setenv("QUIPU_PROJECT_ID", "from-environment")
    monkeypatch.setattr("quipu.storage.store", lambda _db_path: _ReceiptStore())
    monkeypatch.setattr(
        "quipu.keystore._backend.get_or_derive_key",
        lambda pid: requested_pids.append(pid) or b"key",
    )
    monkeypatch.setattr("quipu.sync._aad.aad_for", lambda _pid, _key: b"blinded")
    monkeypatch.setattr("quipu.sync.oplog_store.OplogStore", _EmptyOplog)

    assert cmd_receipts(None, None, None, "json", None) == 0
    assert requested_pids == ["from-environment"]
    assert capsys.readouterr().out.strip() == "[]"
