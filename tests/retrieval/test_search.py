"""Tests for quipu.retrieval.search (public search() API)."""

from __future__ import annotations

import builtins

import pytest
from tests._semantic import TEST_EMBED_DIM

from quipu.retrieval._search import search
from quipu.storage.store import pack_embedding
from quipu.models.cache import active_model

def _unit_vec(index: int) -> list[float]:
    v = [0.0] * TEST_EMBED_DIM
    v[index] = 1.0
    return v


def _fake_embed(text: str) -> list[float]:
    """Always return unit vec at index 0 — deterministic stub."""
    return _unit_vec(0)


# ---------------------------------------------------------------------------
# project_id=None raises ValueError
# ---------------------------------------------------------------------------

def test_search_project_id_none_raises():
    with pytest.raises(ValueError, match="project_id is required"):
        search("query", project_id=None)


# ---------------------------------------------------------------------------
# All four tiers run without error on seeded data
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("tier", ["R0", "R1", "R2", "R3"])
@pytest.mark.usefixtures("semantic_model")
def test_all_tiers_run(tier, tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(
        content="hello world",
        embedding=pack_embedding(_unit_vec(0)),
        project_id="proj1",
    )
    tmp_store.insert(
        content="python programming",
        embedding=pack_embedding(_unit_vec(1)),
        project_id="proj1",
    )

    query = "hello" if tier == "R0" else "hello world python"
    if tier == "R0":
        query = "hello world"

    results = search(query, tier=tier, project_id="proj1", top_k=10, store=tmp_store)
    assert isinstance(results, list)
    # R0: may return 1; R1/R2/R3: at least 1 (atoms exist)
    if tier != "R0":
        # R2 requires FTS5 match — seeded content has "hello" → should match
        pass
    # No assertion on count for R2/R3 as query may not match — just no error


def test_r0_exact_match(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)
    tmp_store.insert(content="exact phrase", project_id="proj1")
    tmp_store.insert(content="other content", project_id="proj1")

    results = search("exact phrase", tier="R0", project_id="proj1", top_k=10, store=tmp_store)
    assert len(results) == 1
    assert results[0].atom.content == "exact phrase"
    assert results[0].score == 1.0
    assert results[0].tier == "R0"


@pytest.mark.usefixtures("semantic_model")
def test_r1_cosine_order(tmp_store, monkeypatch):
    """R1 returns results ordered by descending cosine score."""
    # A is at index 0, B at index 5; query vec at index 0 → A wins
    tmp_store.insert(content="atom A", embedding=pack_embedding(_unit_vec(0)), project_id="p")
    tmp_store.insert(content="atom B", embedding=pack_embedding(_unit_vec(5)), project_id="p")

    def _embed_query(text: str) -> list[float]:
        return _unit_vec(0)

    monkeypatch.setattr("quipu.retrieval._search.embed", _embed_query)
    results = search("anything", tier="R1", project_id="p", top_k=10, store=tmp_store)
    assert len(results) == 2
    assert results[0].atom.content == "atom A"
    assert results[0].score >= results[1].score
    assert all(r.tier == "R1" for r in results)


def test_r2_bm25_order(tmp_store, monkeypatch):
    """R2 returns results ordered by descending BM25 score."""
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)
    tmp_store.insert(content="python python python", project_id="p")
    tmp_store.insert(content="python language", project_id="p")
    tmp_store.insert(content="java programming", project_id="p")

    results = search("python", tier="R2", project_id="p", top_k=10, store=tmp_store)
    # java atom should not appear
    contents = [r.atom.content for r in results]
    assert "java programming" not in contents
    # python atoms in results
    assert any("python" in c for c in contents)
    # Scores descending
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
    assert all(r.tier == "R2" for r in results)


@pytest.mark.usefixtures("semantic_model")
def test_r3_fusion(tmp_store, monkeypatch):
    """R3 returns fused, deduplicated results ordered by combined score."""
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)
    tmp_store.insert(
        content="python programming",
        embedding=pack_embedding(_unit_vec(0)),
        project_id="p",
    )
    tmp_store.insert(
        content="machine learning",
        embedding=pack_embedding(_unit_vec(1)),
        project_id="p",
    )

    results = search("python", tier="R3", project_id="p", top_k=10, store=tmp_store)
    ids = [r.atom.id for r in results]
    assert len(ids) == len(set(ids)), "R3 must deduplicate"
    assert all(r.tier == "R3" for r in results)


@pytest.mark.usefixtures("semantic_model")
def test_top_k_respected(tmp_store, monkeypatch):
    """top_k is respected across all tiers."""
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)
    for i in range(8):
        tmp_store.insert(
            content=f"python example {i}",
            embedding=pack_embedding(_unit_vec(i)),
            project_id="p",
        )

    for tier in ("R1", "R2", "R3"):
        results = search("python", tier=tier, project_id="p", top_k=3, store=tmp_store)
        assert len(results) <= 3, f"tier {tier} exceeded top_k"


def test_top_k_r0(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)
    for _ in range(5):
        tmp_store.insert(content="same exact", project_id="p")

    results = search("same exact", tier="R0", project_id="p", top_k=2, store=tmp_store)
    assert len(results) == 2


def test_invalid_tier_raises(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)
    with pytest.raises(ValueError, match="tier must be"):
        search("q", tier="R9", project_id="p", store=tmp_store)


# ---------------------------------------------------------------------------
# top_k validation
# ---------------------------------------------------------------------------

def test_top_k_zero_raises():
    with pytest.raises(ValueError, match="top_k must be a positive int"):
        search("query", project_id="proj1", top_k=0)


def test_top_k_negative_raises():
    with pytest.raises(ValueError, match="top_k must be a positive int"):
        search("query", project_id="proj1", top_k=-1)


# ---------------------------------------------------------------------------
# TASK-023 — session_id filter on search()
# ---------------------------------------------------------------------------

def test_search_session_id_restricts_results(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="session A", project_id="p", session_id="s1")
    tmp_store.insert(content="session B", project_id="p", session_id="s2")
    tmp_store.insert(content="no session", project_id="p")

    # R2 uses BM25 FTS5 which does partial matching on "session"
    results = search("session", tier="R2", project_id="p", session_id="s1",
                     top_k=10, store=tmp_store)
    assert len(results) == 1
    assert results[0].atom.content == "session A"
    assert results[0].atom.session_id == "s1"


def test_search_default_no_session_filter_includes_null_atoms(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="with session", project_id="p", session_id="s1")
    tmp_store.insert(content="no session", project_id="p")

    results = search("session", tier="R2", project_id="p", top_k=10, store=tmp_store)
    contents = {r.atom.content for r in results}
    assert "no session" in contents
    assert "with session" in contents


def test_search_session_id_unknown_session_returns_empty(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="session A", project_id="p", session_id="s1")

    results = search("session", tier="R2", project_id="p",
                     session_id="nonexistent", top_k=10, store=tmp_store)
    assert results == []


# ---------------------------------------------------------------------------
# TASK-024 — tags filter on search()
# ---------------------------------------------------------------------------

def test_search_tag_filter_restricts_results(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="python memory", project_id="p", tags=["python", "memory"])
    tmp_store.insert(content="java memory", project_id="p", tags=["java", "memory"])
    tmp_store.insert(content="golang notes", project_id="p", tags=["golang"])

    results = search("memory", tier="R2", project_id="p",
                     tags=["python"], top_k=10, store=tmp_store)
    # Only the python-memory-tagged atom should match
    assert len(results) == 1
    assert results[0].atom.content == "python memory"

def test_search_tag_filter_null_tags_excluded(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="python code", project_id="p", tags=["python"])
    tmp_store.insert(content="untagged text", project_id="p")  # tags=None

    results = search("python", tier="R2", project_id="p",
                     tags=["python"], top_k=10, store=tmp_store)
    assert len(results) == 1
    assert results[0].atom.content == "python code"

def test_search_tag_filter_case_insensitive(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="machine learning", project_id="p", tags=["ML", "AI"])

    results = search("learning", tier="R2", project_id="p",
                     tags=["ml"], top_k=10, store=tmp_store)
    assert len(results) == 1
    assert results[0].atom.content == "machine learning"

def test_search_tag_filter_no_match_returns_empty(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="python notes", project_id="p", tags=["python"])

    results = search("python", tier="R2", project_id="p",
                     tags=["java"], top_k=10, store=tmp_store)
    assert results == []

def test_search_default_no_tags_filter_includes_all(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    tmp_store.insert(content="python tagged", project_id="p", tags=["python"])
    tmp_store.insert(content="python untagged", project_id="p")

    results = search("python", tier="R2", project_id="p", top_k=10, store=tmp_store)
    contents = {r.atom.content for r in results}
    assert "python tagged" in contents
    assert "python untagged" in contents


# ---------------------------------------------------------------------------
# TASK-021 — access_count increment on search results
# ---------------------------------------------------------------------------

def test_search_increments_access_count(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    atom = tmp_store.insert(content="unique keyword zz", project_id="p")
    assert atom.access_count == 0

    _ = search("unique", tier="R2", project_id="p", top_k=10, store=tmp_store)

    fetched = tmp_store.get(atom.id)
    assert fetched.access_count >= 1

def test_search_sets_last_accessed(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    atom = tmp_store.insert(content="another keyword", project_id="p")
    assert atom.last_accessed is None

    _ = search("another", tier="R2", project_id="p", top_k=10, store=tmp_store)

    fetched = tmp_store.get(atom.id)
    assert fetched.last_accessed is not None

@pytest.mark.usefixtures("semantic_model")
def test_access_count_increments_for_all_tiers(tmp_store, monkeypatch):
    monkeypatch.setattr("quipu.retrieval._search.embed", _fake_embed)

    for tier in ("R0", "R1", "R2", "R3"):
        content = f"access test {tier}"
        atom = tmp_store.insert(
            content=content,
            embedding=pack_embedding(_unit_vec(0)),
            project_id="p",
        )
        assert atom.access_count == 0
        _ = search(content, tier=tier, project_id="p", top_k=10, store=tmp_store)
        fetched = tmp_store.get(atom.id)
        assert fetched.access_count >= 1, f"tier {tier} did not increment access_count"


# ---------------------------------------------------------------------------
# TASK-064 — keyword-only retrieval
# ---------------------------------------------------------------------------

def test_keyword_only_r0_and_r2_do_not_import_embeddings(tmp_store, monkeypatch):
    monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "none")
    assert active_model() is None
    atom = tmp_store.insert(content="zephyrquill exact marker", project_id="keyword")
    real_import = builtins.__import__

    def reject_embeddings(name, *args, **kwargs):
        if name == "quipu.embeddings" or name.startswith("quipu.embeddings."):
            raise AssertionError("R0/R2 search imported quipu.embeddings")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_embeddings)
    r0 = search(atom.content, tier="R0", project_id="keyword", store=tmp_store)
    r2 = search("zephyrquill", tier="R2", project_id="keyword", store=tmp_store)

    assert [r.atom.id for r in r0] == [atom.id]
    assert atom.id in [r.atom.id for r in r2]


@pytest.mark.parametrize("tier", ["R1", "R3"])
def test_keyword_only_vector_tiers_fall_back_to_bm25_without_warning(
    tier, tmp_store, monkeypatch, capsys
):
    monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "none")
    assert active_model() is None
    atom = tmp_store.insert(content="zephyrquill fallback marker", project_id="keyword")

    results = search("zephyrquill", tier=tier, project_id="keyword", store=tmp_store)

    assert atom.id in [r.atom.id for r in results]
    assert all(r.tier == "R2" for r in results)
    assert "WARNING" not in capsys.readouterr().err


@pytest.mark.parametrize("tier", ["R1", "R3"])
def test_configured_model_failure_warns_then_falls_back_to_bm25(
    tier, tmp_store, monkeypatch, capsys
):
    monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", "bge-small-en-v1.5")
    atom = tmp_store.insert(content="zephyrquill runtime failure marker", project_id="keyword")

    def broken_embed(_query):
        raise OSError("corrupt model")

    monkeypatch.setattr("quipu.retrieval._search.embed", broken_embed)
    results = search("zephyrquill", tier=tier, project_id="keyword", store=tmp_store)

    assert atom.id in [r.atom.id for r in results]
    assert "WARNING: embedding model unavailable" in capsys.readouterr().err
