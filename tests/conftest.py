"""Root test configuration."""

from __future__ import annotations

import pytest

from tests._semantic import SEMANTIC_MODEL


@pytest.fixture()
def semantic_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register the model required by a test that packs or embeds vectors."""
    monkeypatch.setenv("QUIPU_EMBEDDING_MODEL", SEMANTIC_MODEL)
