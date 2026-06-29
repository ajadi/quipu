"""Tests for push/pull idempotency, sequence_no dedup, and zero-knowledge transport."""

from tests.sync.conftest import write_local

from quipu.sync import pull, push


def test_push_sends_only_unpushed(make_store, transport, key, project_id, blinded):
    store = make_store("c1")
    write_local(store, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:01Z", content="a")
    write_local(store, key, project_id, "c1", op="upsert", record_id="r2",
                ts="2026-06-21T00:00:02Z", content="b")

    n = push(project_id, store=store, transport=transport, key=key, client_id="c1")
    assert n == 2
    assert len(transport._log[blinded]) == 2


def test_push_idempotent(make_store, transport, key, project_id, blinded):
    store = make_store("c1")
    write_local(store, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:01Z", content="a")

    first = push(project_id, store=store, transport=transport, key=key, client_id="c1")
    second = push(project_id, store=store, transport=transport, key=key, client_id="c1")
    assert first == 1
    assert second == 0  # nothing new
    assert len(transport._log[blinded]) == 1  # no dup on hub


def test_pull_applies_and_is_idempotent(make_store, transport, key, project_id):
    writer = make_store("w")
    reader = make_store("r")
    write_local(writer, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:01Z", content="hello")
    push(project_id, store=writer, transport=transport, key=key, client_id="c1")

    n1 = pull(project_id, store=reader, transport=transport, key=key, client_id="r")
    assert n1 == 1
    atom = reader.get("r1")
    assert atom is not None
    assert atom.content == "hello"

    # repeated pull: sequence_no dedup -> no new applies, state unchanged
    n2 = pull(project_id, store=reader, transport=transport, key=key, client_id="r")
    assert n2 == 0
    assert reader.get("r1").content == "hello"


def test_transport_dict_is_zero_knowledge(make_store, transport, key, project_id, blinded):
    """Hub dict carries only hub-visible fields — no source/pushed/plaintext/project_id/key."""
    store = make_store("c1")
    write_local(store, key, project_id, "c1", op="upsert", record_id="r1",
                ts="2026-06-21T00:00:01Z", content="SECRET-CONTENT")
    push(project_id, store=store, transport=transport, key=key, client_id="c1")

    d = transport._log[blinded][0]
    assert set(d.keys()) == {
        "entry_id", "client_id", "sequence_no", "op", "record_id",
        "blinded_project_id", "ts", "payload",
    }
    assert "source" not in d and "pushed" not in d
    assert d["blinded_project_id"] == blinded
    # Real project_id and plaintext never appear anywhere in the wire dict.
    serialized = repr(d)
    assert project_id not in serialized
    assert "SECRET-CONTENT" not in serialized
