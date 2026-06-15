"""Shared fixtures for langchain_talordata tests."""

from __future__ import annotations

import json
import os
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest

from langchain_talordata.schema import EngineRegistry, EngineSchema, Field, FieldGroup


def pytest_addoption(parser):
    parser.addoption(
        "--run-integration",
        action="store_true",
        default=False,
        help="Run integration tests that call the upstream Talor SERP API.",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-integration"):
        return
    skip_integration = pytest.mark.skip(reason="need --run-integration option to run")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


@pytest.fixture(scope="session")
def registry() -> EngineRegistry:
    """Shared EngineRegistry loaded once per session."""
    return EngineRegistry()


@pytest.fixture
def google_schema(registry: EngineRegistry) -> EngineSchema:
    return registry.engine("google")


@pytest.fixture
def bing_news_schema(registry: EngineRegistry) -> EngineSchema:
    return registry.engine("bing_news")


@pytest.fixture
def patents_details_schema(registry: EngineRegistry) -> EngineSchema:
    return registry.engine("google_patents_details")


@pytest.fixture
def flights_schema(registry: EngineRegistry) -> EngineSchema:
    return registry.engine("google_flights")


@pytest.fixture
def mock_api_response():
    """Return a factory for mock API responses."""

    def _make(
        data: Dict[str, Any] | None = None,
        status_code: int = 200,
    ) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.raise_for_status = MagicMock()
        if data is not None:
            resp.json.return_value = data
        else:
            resp.json.return_value = {
                "organic": [
                    {
                        "title": "Test Result",
                        "link": "https://example.com",
                        "snippet": "Test snippet",
                        "position": 1,
                    }
                ]
            }
        resp.text = json.dumps(resp.json.return_value)
        return resp

    return _make
