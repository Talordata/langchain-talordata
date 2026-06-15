"""Integration tests — these call the real Talor SERP API.

Run with:
    python -m pytest tests/test_integration.py -v --run-integration

Requires TALOR_API_KEY environment variable or TalorSerpAPIWrapper default.
"""

from __future__ import annotations

import os

import pytest

from langchain_talordata.wrapper import TalorSerpAPIWrapper

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def wrapper() -> TalorSerpAPIWrapper:
    """Create a real wrapper for integration tests."""
    api_key = os.environ.get("TALOR_API_KEY")
    if not api_key:
        pytest.skip("TALOR_API_KEY not set, skipping integration tests")
    return TalorSerpAPIWrapper(talor_api_key=api_key)


# ---------- Wrapper.results() ----------


class TestResultsIntegration:
    def test_google_basic(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.results("LangChain tutorial")
        assert result["ok"] is True
        assert result["engine"] == "google"
        assert "organic" in result["data"]

    def test_bing_news(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.results("AI news", engine="bing_news")
        assert result["ok"] is True
        assert result["engine"] == "bing_news"

    def test_google_with_params(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.results(
            "machine learning",
            engine="google",
            gl="cn",
            hl="zh",
            num=5,
        )
        assert result["ok"] is True
        assert "organic" in result["data"]

    def test_duckduckgo(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.results("python programming", engine="duckduckgo")
        assert result["ok"] is True

    def test_yandex(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.results("test query", engine="yandex")
        assert result["ok"] is True

    def test_google_patents_details(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.results(
            "",
            engine="google_patents_details",
            parent_id="patent/US11734097B1/en",
            gl="us",
            hl="en",
        )
        assert result["ok"] is True


# ---------- Wrapper.run() ----------


class TestRunIntegration:
    def test_run_returns_text(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.run("what is LangChain")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_run_with_engine(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.run("hello world", engine="bing")
        assert isinstance(result, str)


# ---------- Wrapper engine info ----------


class TestEngineInfoIntegration:
    def test_list_engines(self, wrapper: TalorSerpAPIWrapper):
        engines = wrapper.list_engines()
        assert len(engines) > 30
        assert "google" in engines

    def test_engine_description(self, wrapper: TalorSerpAPIWrapper):
        desc = wrapper.engine_description("google")
        assert "google" in desc.lower()

    def test_engine_param_schema(self, wrapper: TalorSerpAPIWrapper):
        schema = wrapper.engine_param_schema("google")
        assert schema is not None
        assert schema["type"] == "object"


# ---------- History & Statistics ----------


class TestHistoryIntegration:
    def test_history(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.history(page=1, page_size=5)
        assert result["ok"] is True

    def test_statistics(self, wrapper: TalorSerpAPIWrapper):
        result = wrapper.statistics("2024-01-01", "2024-12-31")
        assert result["ok"] is True


# ---------- Tool integration ----------


class TestToolIntegration:
    def test_tool_invoke(self, wrapper: TalorSerpAPIWrapper):
        from langchain_talordata.tool import create_talor_serp_tool

        tool = create_talor_serp_tool(wrapper)
        result = tool.invoke({"query": "hello"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_tool_invoke_with_engine(self, wrapper: TalorSerpAPIWrapper):
        from langchain_talordata.tool import create_talor_serp_tool

        tool = create_talor_serp_tool(wrapper)
        result = tool.invoke({"query": "news", "engine": "bing_news"})
        assert isinstance(result, str)
