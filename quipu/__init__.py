"""Quipu — local-first memory system."""

try:
    from quipu._build_version import __version__
except ImportError:
    try:
        from importlib.metadata import version as _get_version

        __version__ = _get_version("quipu-mcp")
    except Exception:
        __version__ = "0.0.0"
