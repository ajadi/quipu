"""Unit tests for KG triple & edge CRUD, BFS traversal, and traverse method."""

import sqlite3
import json

import pytest

from quipu.storage import store, Atom
from quipu.storage.store import Store


@pytest.fixture
def s(tmp_path):
    db = tmp_path / "test_kg.db"
    with store(str(db)) as _store:
        yield _store


# ---------------------------------------------------------------------------
# kg_triples table exists after migration 0006
# ---------------------------------------------------------------------------

class TestKgTriplesTable:
    def test_kg_triples_table_exists(self, s):
        rows = s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_triples'"
        ).fetchall()
        assert len(rows) == 1

    def test_kg_triples_indexes_exist(self, s):
        indexes = {
            row[0] for row in s._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_kg_triples_subj" in indexes
        assert "idx_kg_triples_obj" in indexes
        assert "idx_kg_triples_pred" in indexes
        assert "idx_kg_triples_proj" in indexes


class TestKgEdgesTable:
    def test_kg_edges_table_exists(self, s):
        rows = s._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='kg_edges'"
        ).fetchall()
        assert len(rows) == 1

    def test_kg_edges_indexes_exist(self, s):
        indexes = {
            row[0] for row in s._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        assert "idx_kg_edges_from" in indexes
        assert "idx_kg_edges_to" in indexes
        assert "idx_kg_edges_type" in indexes
        assert "idx_kg_edges_proj" in indexes


# ---------------------------------------------------------------------------
# insert_triple
# ---------------------------------------------------------------------------

class TestInsertTriple:
    def test_insert_triple_returns_dict(self, s):
        result = s.insert_triple(
            subject="atom-a", predicate="causes", object="atom-b"
        )
        assert isinstance(result, dict)
        assert result["subject"] == "atom-a"
        assert result["predicate"] == "causes"
        assert result["object"] == "atom-b"

    def test_insert_triple_defaults(self, s):
        result = s.insert_triple(
            subject="s", predicate="p", object="o"
        )
        assert result["confidence"] == 1.0
        assert result["valid_from"] is not None
        assert "T" in result["valid_from"]
        assert result["valid_to"] is None
        assert result["source_ref"] is None
        assert result["project_id"] is None
        assert result["created_at"] is not None

    def test_insert_triple_with_all_fields(self, s):
        result = s.insert_triple(
            subject="sub", predicate="pred", object="obj",
            valid_from="2024-01-01T00:00:00Z",
            valid_to="2025-01-01T00:00:00Z",
            confidence=0.9,
            source_ref="src-1",
            project_id="proj-x",
        )
        assert result["valid_from"] == "2024-01-01T00:00:00Z"
        assert result["valid_to"] == "2025-01-01T00:00:00Z"
        assert result["confidence"] == 0.9
        assert result["source_ref"] == "src-1"
        assert result["project_id"] == "proj-x"

    def test_insert_triple_auto_increment_id(self, s):
        r1 = s.insert_triple(subject="a", predicate="p", object="b")
        r2 = s.insert_triple(subject="c", predicate="q", object="d")
        assert r2["id"] == r1["id"] + 1

    def test_insert_triple_confidence_boundary_zero(self, s):
        result = s.insert_triple(
            subject="a", predicate="p", object="b", confidence=0.0
        )
        assert result["confidence"] == 0.0

    def test_insert_triple_confidence_boundary_one(self, s):
        result = s.insert_triple(
            subject="a", predicate="p", object="b", confidence=1.0
        )
        assert result["confidence"] == 1.0

    def test_insert_triple_rejects_confidence_lt_zero(self, s):
        with pytest.raises(sqlite3.IntegrityError):
            s.insert_triple(
                subject="a", predicate="p", object="b", confidence=-0.1
            )

    def test_insert_triple_rejects_confidence_gt_one(self, s):
        with pytest.raises(sqlite3.IntegrityError):
            s.insert_triple(
                subject="a", predicate="p", object="b", confidence=1.1
            )


# ---------------------------------------------------------------------------
# insert_edge
# ---------------------------------------------------------------------------

class TestInsertEdge:
    def test_insert_edge_returns_dict(self, s):
        result = s.insert_edge(
            from_atom_id="atom-1", to_atom_id="atom-2", edge_type="depends_on"
        )
        assert isinstance(result, dict)
        assert result["from_atom_id"] == "atom-1"
        assert result["to_atom_id"] == "atom-2"
        assert result["edge_type"] == "depends_on"

    def test_insert_edge_with_project_id(self, s):
        result = s.insert_edge(
            from_atom_id="a", to_atom_id="b", edge_type="supersedes",
            project_id="proj-g",
        )
        assert result["project_id"] == "proj-g"

    def test_insert_edge_with_metadata(self, s):
        result = s.insert_edge(
            from_atom_id="a", to_atom_id="b", edge_type="blocks",
            metadata={"reason": "obsolete", "since": "2024"},
        )
        assert result["metadata"] == {"reason": "obsolete", "since": "2024"}

    def test_insert_edge_null_metadata(self, s):
        result = s.insert_edge(
            from_atom_id="a", to_atom_id="b", edge_type="causal",
        )
        assert result["metadata"] is None

    def test_insert_edge_created_at_present(self, s):
        result = s.insert_edge(
            from_atom_id="a", to_atom_id="b", edge_type="touches_file",
        )
        assert "T" in result["created_at"]
        assert "Z" in result["created_at"]

    def test_insert_edge_auto_increment_id(self, s):
        r1 = s.insert_edge(from_atom_id="a", to_atom_id="b", edge_type="depends_on")
        r2 = s.insert_edge(from_atom_id="c", to_atom_id="d", edge_type="supersedes")
        assert r2["id"] == r1["id"] + 1

    # Edge type validation
    @pytest.mark.parametrize("etype", [
        "supersedes", "blocks", "touches_file", "decided_by", "depends_on", "causal",
    ])
    def test_all_valid_edge_types_accepted(self, s, etype):
        result = s.insert_edge(
            from_atom_id="a", to_atom_id="b", edge_type=etype,
        )
        assert result["edge_type"] == etype

    def test_insert_edge_rejects_invalid_type(self, s):
        with pytest.raises(sqlite3.IntegrityError):
            s.insert_edge(
                from_atom_id="a", to_atom_id="b", edge_type="invalid_type",
            )

    def test_insert_edge_rejects_empty_type(self, s):
        with pytest.raises(sqlite3.IntegrityError):
            s.insert_edge(
                from_atom_id="a", to_atom_id="b", edge_type="",
            )


# ---------------------------------------------------------------------------
# get_connected_atoms — BFS traversal
# ---------------------------------------------------------------------------

class TestGetConnectedAtoms:
    def test_returns_empty_for_no_edges(self, s):
        s.insert(content="lonely atom", id="lonely", project_id="p")
        result = s.get_connected_atoms("lonely", project_id="p")
        assert result == []

    def test_returns_empty_for_nonexistent_atom(self, s):
        result = s.get_connected_atoms("nonexistent", project_id="p")
        assert result == []

    def test_single_edge_one_hop(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="leaf", edge_type="depends_on")

        connected = s.get_connected_atoms("root", project_id="p", max_depth=1)
        assert len(connected) == 1
        assert connected[0].id == "leaf"

    def test_excludes_start_atom(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="leaf", edge_type="depends_on")

        connected = s.get_connected_atoms("root", project_id="p")
        assert all(a.id != "root" for a in connected)

    def test_bidirectional_traversal(self, s):
        """Edges can be traversed in both directions."""
        s.insert(content="a", id="a", project_id="p")
        s.insert(content="b", id="b", project_id="p")
        s.insert(content="c", id="c", project_id="p")
        s.insert_edge(from_atom_id="b", to_atom_id="a", edge_type="depends_on")
        s.insert_edge(from_atom_id="b", to_atom_id="c", edge_type="causal")

        connected = s.get_connected_atoms("b", project_id="p", max_depth=1)
        ids = {a.id for a in connected}
        assert ids == {"a", "c"}

    def test_respects_max_depth(self, s):
        # Chain: root -> mid -> leaf
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="mid", id="mid", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="mid", edge_type="depends_on")
        s.insert_edge(from_atom_id="mid", to_atom_id="leaf", edge_type="depends_on")

        connected_1 = s.get_connected_atoms("root", project_id="p", max_depth=1)
        assert len(connected_1) == 1
        assert connected_1[0].id == "mid"

        connected_2 = s.get_connected_atoms("root", project_id="p", max_depth=2)
        ids = {a.id for a in connected_2}
        assert ids == {"mid", "leaf"}

    def test_max_depth_zero_returns_empty(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="leaf", edge_type="depends_on")

        connected = s.get_connected_atoms("root", project_id="p", max_depth=0)
        assert connected == []

    def test_edge_types_filter(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="dep", id="dep", project_id="p")
        s.insert(content="blk", id="blk", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="dep", edge_type="depends_on")
        s.insert_edge(from_atom_id="root", to_atom_id="blk", edge_type="blocks")

        connected = s.get_connected_atoms(
            "root", project_id="p", edge_types=["depends_on"]
        )
        assert len(connected) == 1
        assert connected[0].id == "dep"

    def test_diamond_dedup(self, s):
        """Diamond: root -> a, root -> b, a -> c, b -> c. c appears once."""
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="a", id="a", project_id="p")
        s.insert(content="b", id="b", project_id="p")
        s.insert(content="c", id="c", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="a", edge_type="depends_on")
        s.insert_edge(from_atom_id="root", to_atom_id="b", edge_type="depends_on")
        s.insert_edge(from_atom_id="a", to_atom_id="c", edge_type="depends_on")
        s.insert_edge(from_atom_id="b", to_atom_id="c", edge_type="depends_on")

        connected = s.get_connected_atoms("root", project_id="p", max_depth=2)
        ids = {a.id for a in connected}
        assert ids == {"a", "b", "c"}
        # c must appear only once
        c_count = sum(1 for a in connected if a.id == "c")
        assert c_count == 1

    def test_cycle_terminates(self, s):
        """A -> B -> C -> A cycle should not loop forever."""
        s.insert(content="a", id="a", project_id="p")
        s.insert(content="b", id="b", project_id="p")
        s.insert(content="c", id="c", project_id="p")
        s.insert_edge(from_atom_id="a", to_atom_id="b", edge_type="depends_on")
        s.insert_edge(from_atom_id="b", to_atom_id="c", edge_type="depends_on")
        s.insert_edge(from_atom_id="c", to_atom_id="a", edge_type="depends_on")

        connected = s.get_connected_atoms("a", project_id="p", max_depth=5)
        ids = {a.id for a in connected}
        assert ids == {"b", "c"}

    def test_returns_atom_objects(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="leaf", edge_type="depends_on")

        connected = s.get_connected_atoms("root", project_id="p")
        assert all(isinstance(a, Atom) for a in connected)
        assert connected[0].content == "leaf"


# ---------------------------------------------------------------------------
# traverse — structured subgraph
# ---------------------------------------------------------------------------

class TestTraverse:
    def test_returns_dict_with_nodes_and_edges(self, s):
        s.insert(content="root", id="root", project_id="p")
        result = s.traverse("root", project_id="p", max_depth=1)
        assert isinstance(result, dict)
        assert "nodes" in result
        assert "edges" in result

    def test_includes_start_node(self, s):
        s.insert(content="root", id="root", project_id="p")
        result = s.traverse("root", project_id="p", max_depth=1)
        node_ids = {n["id"] for n in result["nodes"]}
        assert "root" in node_ids

    def test_nodes_are_dicts_without_embedding(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="leaf", edge_type="depends_on")

        result = s.traverse("root", project_id="p", max_depth=1)
        for node in result["nodes"]:
            assert isinstance(node, dict)
            assert "embedding" not in node
            assert "id" in node
            assert "content" in node
            assert "type" in node

    def test_edges_are_dicts(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="leaf", edge_type="depends_on")

        result = s.traverse("root", project_id="p", max_depth=1)
        assert len(result["edges"]) == 1
        edge = result["edges"][0]
        assert edge["from_atom_id"] == "root"
        assert edge["to_atom_id"] == "leaf"
        assert edge["edge_type"] == "depends_on"

    def test_edge_metadata_deserialized(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(
            from_atom_id="root", to_atom_id="leaf", edge_type="blocks",
            metadata={"reason": "test"},
        )

        result = s.traverse("root", project_id="p", max_depth=1)
        assert result["edges"][0]["metadata"] == {"reason": "test"}

    def test_respects_max_depth(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="mid", id="mid", project_id="p")
        s.insert(content="leaf", id="leaf", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="mid", edge_type="depends_on")
        s.insert_edge(from_atom_id="mid", to_atom_id="leaf", edge_type="depends_on")

        result = s.traverse("root", project_id="p", max_depth=1)
        node_ids = {n["id"] for n in result["nodes"]}
        assert "root" in node_ids
        assert "mid" in node_ids
        assert "leaf" not in node_ids

        result2 = s.traverse("root", project_id="p", max_depth=2)
        node_ids2 = {n["id"] for n in result2["nodes"]}
        assert "leaf" in node_ids2

    def test_edge_types_filter(self, s):
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="dep", id="dep", project_id="p")
        s.insert(content="blk", id="blk", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="dep", edge_type="depends_on")
        s.insert_edge(from_atom_id="root", to_atom_id="blk", edge_type="blocks")

        result = s.traverse(
            "root", project_id="p", edge_types=["depends_on"]
        )
        node_ids = {n["id"] for n in result["nodes"]}
        assert "dep" in node_ids
        assert "blk" not in node_ids

    def test_max_depth_zero_returns_empty(self, s):
        s.insert(content="root", id="root", project_id="p")
        result = s.traverse("root", project_id="p", max_depth=0)
        assert result == {"nodes": [], "edges": []}

    def test_nonexistent_atom(self, s):
        result = s.traverse("nonexistent", project_id="p")
        assert result["nodes"] == []
        assert result["edges"] == []

    def test_diamond_dedup_edges(self, s):
        """Diamond shape: edges should not be duplicated."""
        s.insert(content="root", id="root", project_id="p")
        s.insert(content="a", id="a", project_id="p")
        s.insert(content="b", id="b", project_id="p")
        s.insert(content="c", id="c", project_id="p")
        s.insert_edge(from_atom_id="root", to_atom_id="a", edge_type="depends_on")
        s.insert_edge(from_atom_id="a", to_atom_id="c", edge_type="depends_on")
        s.insert_edge(from_atom_id="b", to_atom_id="c", edge_type="depends_on")
        s.insert_edge(from_atom_id="root", to_atom_id="b", edge_type="depends_on")

        result = s.traverse("root", project_id="p", max_depth=2)
        edge_ids = [e["id"] for e in result["edges"]]
        assert len(edge_ids) == len(set(edge_ids))
        node_ids = {n["id"] for n in result["nodes"]}
        assert node_ids == {"root", "a", "b", "c"}


# ---------------------------------------------------------------------------
# Integration: store + graph workflow
# ---------------------------------------------------------------------------

class TestGraphWorkflow:
    def test_write_then_link_then_traverse(self, s):
        """End-to-end: write atoms, link them, traverse subgraph."""
        # Write atoms
        a1 = s.insert(content="first fact", project_id="p")
        a2 = s.insert(content="second fact", project_id="p")
        a3 = s.insert(content="third fact", project_id="p")

        # Link them
        e1 = s.insert_edge(
            from_atom_id=a1.id, to_atom_id=a2.id, edge_type="causal",
            project_id="p",
        )
        e2 = s.insert_edge(
            from_atom_id=a2.id, to_atom_id=a3.id, edge_type="depends_on",
            project_id="p",
        )

        # Traverse
        result = s.traverse(a1.id, project_id="p", max_depth=2)
        node_ids = {n["id"] for n in result["nodes"]}
        assert a1.id in node_ids
        assert a2.id in node_ids
        assert a3.id in node_ids

        # Insert a triple about the connection
        triple = s.insert_triple(
            subject=a1.id, predicate="causes", object=a2.id,
            confidence=0.95, project_id="p",
        )
        assert triple["confidence"] == 0.95
