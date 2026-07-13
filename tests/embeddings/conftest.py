"""Network safety rails for the embeddings test suite.

The unit suite exercises the missing-model path in ``quipu.models.loader``.
That path normally attempts a Hugging Face download, so keep it hermetic unless
a deliberately marked real-network test is explicitly requested.
"""

from __future__ import annotations

import os

import pytest


_NETWORK_OPT_IN = "--run-network"


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the explicit switch for real Hugging Face tests."""
    parser.addoption(
        _NETWORK_OPT_IN,
        action="store_true",
        default=False,
        help="run tests marked 'network' (they may contact external services)",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Make the normal embeddings suite offline before test code runs."""
    config.addinivalue_line(
        "markers",
        "network: real external-service test; requires --run-network",
    )
    os.environ["HF_HUB_OFFLINE"] = "1"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep real-network tests opt-in even when they are collected locally."""
    if config.getoption(_NETWORK_OPT_IN):
        return

    skip_network = pytest.mark.skip(
        reason="network tests require the explicit --run-network option",
    )
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip_network)


@pytest.fixture(autouse=True)
def block_huggingface_downloads_by_default(monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest):
    """Defensively block ``snapshot_download`` outside opted-in network tests.

    ``HF_HUB_OFFLINE`` is honored by supported huggingface_hub releases, but
    the mock keeps this suite network-free even with an incompatible or older
    client.  Tests that intentionally replace ``huggingface_hub`` in
    ``sys.modules`` still retain their local fake implementation.
    """
    is_opted_in_network_test = (
        request.config.getoption(_NETWORK_OPT_IN)
        and request.node.get_closest_marker("network") is not None
    )
    if is_opted_in_network_test:
        monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
        return

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    try:
        import huggingface_hub
    except ImportError:
        return

    def offline_snapshot_download(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(
            "Hugging Face downloads are disabled for unit tests; "
            "use --run-network for an explicitly marked network test"
        )

    monkeypatch.setattr(huggingface_hub, "snapshot_download", offline_snapshot_download)
