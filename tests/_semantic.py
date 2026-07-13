"""Canonical semantic-model values shared by test fakes and vectors."""

from quipu.models.cache import MODELS


SEMANTIC_MODEL = "bge-small-en-v1.5"
TEST_EMBED_DIM = int(MODELS[SEMANTIC_MODEL]["dim"])
