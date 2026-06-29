"""Quipu capture: drain queue consumer + secret scanner."""

from .drain import drain
from .secrets import looks_like_secret

__all__ = ["drain", "looks_like_secret"]
