"""Tests for quipu.ranking.cosine (dot product)."""

import math
import pytest
from quipu.ranking.cosine import dot


def test_dot_orthogonal():
    """Orthogonal unit vectors → dot = 0."""
    a = [1.0, 0.0, 0.0]
    b = [0.0, 1.0, 0.0]
    assert dot(a, b) == pytest.approx(0.0)


def test_dot_identical():
    """Identical unit vectors → dot = 1.0 (cosine = 1)."""
    a = [1.0, 0.0, 0.0]
    assert dot(a, a) == pytest.approx(1.0)


def test_dot_opposite():
    """Opposite unit vectors → dot = -1.0."""
    a = [1.0, 0.0]
    b = [-1.0, 0.0]
    assert dot(a, b) == pytest.approx(-1.0)


def test_dot_general():
    """General case: [0.6, 0.8] · [0.8, 0.6] = 0.96."""
    a = [0.6, 0.8]
    b = [0.8, 0.6]
    assert dot(a, b) == pytest.approx(0.6 * 0.8 + 0.8 * 0.6)


def test_dot_single_element():
    assert dot([1.0], [0.5]) == pytest.approx(0.5)


def test_dot_384_dims():
    """Smoke test on 384-dim vectors."""
    a = [1.0] + [0.0] * 383
    b = [0.0] * 383 + [1.0]
    assert dot(a, b) == pytest.approx(0.0)
