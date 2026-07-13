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


# ---------------------------------------------------------------------------
# TASK-053 — dim-mismatch must raise, never silently truncate (was: zip)
# ---------------------------------------------------------------------------

def test_dot_raises_value_error_on_length_mismatch():
    """A shorter/longer vector must raise, not silently zip-truncate."""
    a = [1.0] * 768
    b = [1.0] * 384
    with pytest.raises(ValueError, match=r"dim mismatch: 768 != 384"):
        dot(a, b)


def test_dot_raises_on_single_element_mismatch():
    with pytest.raises(ValueError, match=r"dim mismatch: 3 != 2"):
        dot([1.0, 0.0, 0.0], [1.0, 0.0])


def test_dot_correct_product_on_equal_length_vectors():
    """Regression guard: the ValueError check must not affect the correct
    product when lengths DO match."""
    a = [1.0, 2.0, 3.0]
    b = [4.0, 5.0, 6.0]
    assert dot(a, b) == pytest.approx(1.0 * 4.0 + 2.0 * 5.0 + 3.0 * 6.0)
