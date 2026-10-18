"""Pytest configuration and test runner fixtures for AI-Examples."""

from __future__ import annotations

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-heavy",
        action="store_true",
        default=False,
        help="Run heavy/slow tests (e.g., ONNX/graph exports, JIT compiler captures, full crate integration)",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "heavy: mark test as heavy/slow (skipped by default unless --run-heavy is specified)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-heavy"):
        # --run-heavy given in cli: do not skip heavy tests
        return
    skip_heavy = pytest.mark.skip(
        reason="Heavy test locked by default. Pass --run-heavy to execute."
    )
    for item in items:
        if "heavy" in item.keywords:
            item.add_marker(skip_heavy)
