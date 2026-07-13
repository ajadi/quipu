"""Tests for the canonical semantic-model test support."""

from quipu.models.cache import MODELS
from quipu.models.cache import active_model
from tests._semantic import SEMANTIC_MODEL, TEST_EMBED_DIM


def test_semantic_model_dimension_is_registry_derived():
    assert TEST_EMBED_DIM == MODELS[SEMANTIC_MODEL]["dim"]


def test_semantic_model_fixture_uses_canonical_model(semantic_model):
    assert active_model() == SEMANTIC_MODEL
