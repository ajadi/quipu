"""Tests for quipu.vec._build threshold / atom-count helpers."""

from __future__ import annotations

import pytest

from quipu.vec._build import atom_count, crossed
from quipu.storage.store import pack_embedding


def _unit_vec(i: int) -> list[float]:
    v = [0.0] * 384
    v[i] = 1.0
    return v


class TestAtomCount:
    def test_empty_db(self, tmp_conn):
        assert atom_count(tmp_conn) == 0

    def test_count_after_inserts(self, tmp_conn):
        for i in range(3):
            tmp_conn.execute(
                "INSERT INTO atoms (id, type, scope, content, project_id, invalidated)"
                " VALUES (?, 'diary', 'project', 'c', 'p', 0)",
                (f"id{i}",),
            )
        tmp_conn.commit()
        assert atom_count(tmp_conn) == 3

    def test_count_with_seeded_conn(self, seeded_conn):
        assert atom_count(seeded_conn) == 5


class TestCrossed:
    def test_below_threshold(self, tmp_conn):
        # 0 atoms, threshold 10 → not crossed
        assert crossed(tmp_conn, 10) is False

    def test_exactly_at_threshold(self, seeded_conn):
        # 5 atoms, threshold 5 → crossed
        assert crossed(seeded_conn, 5) is True

    def test_above_threshold(self, seeded_conn):
        # 5 atoms, threshold 3 → crossed
        assert crossed(seeded_conn, 3) is True

    def test_one_below_threshold(self, seeded_conn):
        # 5 atoms, threshold 6 → not crossed
        assert crossed(seeded_conn, 6) is False
