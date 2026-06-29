"""Quipu write pipeline: store content with embeddings and local extraction."""

from .pipeline import write
from .flush import flush

__all__ = ["write", "flush"]
