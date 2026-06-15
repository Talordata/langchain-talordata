"""Tests for langchain_talordata.tool module."""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import StructuredTool

from langchain_talordata.tool import (
    TalorSerpTool,
    create_talor_serp_history_tool,
    create_talor_serp_list_engines_tool,
    create_talor_serp_statistics_tool,
    create_talor_serp_tool,
)
from langchain_talordata.wrapper import TalorSerpAPIWrapper


@pytest.fixture
def wrapper():
    return TalorSerpAPIWrapper(talor_api_key="test-key")


class TestCreateTalorSerpTool:
    def test_returns_structured_tool(self, wrapper):
        tool = create_talor_serp_tool(wrapper)
        assert isinstance(tool, StructuredTool)

    def test_tool_name(self, wrapper):
        tool = create_talor_serp_tool(wrapper)
        assert tool.name == "talor_serp_search"

    def test_custom_name(self, wrapper):
        tool = create_talor_serp_tool(wrapper, name="my_search")
        assert tool.name == "my_search"

    def test_has_description(self, wrapper):
        tool = create_talor_serp_tool(wrapper)
        assert tool.description is not None
        assert len(tool.description) > 0

    def test_args_schema_allows_empty_query(self, wrapper):
        tool = create_talor_serp_tool(wrapper)
        schema = tool.args_schema.model_json_schema()
        assert "required" not in schema or "query" not in schema.get("required", [])

    def test_invoke_calls_wrapper(self, wrapper):
        with patch("langchain_talordata.wrapper.TalorSerpAPIWrapper.run", return_value="mocked result") as mock_run:
            tool = create_talor_serp_tool(wrapper)
            result = tool.invoke({"query": "test"})
            mock_run.assert_called_once_with("test", engine=None)
            assert result == "mocked result"

    def test_invoke_with_engine(self, wrapper):
        with patch("langchain_talordata.wrapper.TalorSerpAPIWrapper.run", return_value="mocked result") as mock_run:
            tool = create_talor_serp_tool(wrapper)
            tool.invoke({"query": "test", "engine": "bing"})
            mock_run.assert_called_once_with("test", engine="bing")

    def test_invoke_with_params(self, wrapper):
        with patch("langchain_talordata.wrapper.TalorSerpAPIWrapper.run", return_value="mocked result") as mock_run:
            tool = create_talor_serp_tool(wrapper)
            tool.invoke({
                "query": "test",
                "engine": "google",
                "params": {"gl": "cn", "hl": "zh"},
            })
            mock_run.assert_called_once_with("test", engine="google", gl="cn", hl="zh")

    def test_invoke_with_params_json_string(self, wrapper):
        with patch("langchain_talordata.wrapper.TalorSerpAPIWrapper.run", return_value="mocked result") as mock_run:
            tool = create_talor_serp_tool(wrapper)
            tool.invoke({
                "query": "test",
                "params": '{"gl": "cn"}',
            })
            mock_run.assert_called_once_with("test", engine=None, gl="cn")

    def test_invoke_without_query(self, wrapper):
        with patch("langchain_talordata.wrapper.TalorSerpAPIWrapper.run", return_value="mocked result") as mock_run:
            tool = create_talor_serp_tool(wrapper)
            tool.invoke({"engine": "google", "params": {"parent_id": "patent/US1"}})
            mock_run.assert_called_once_with("", engine="google", parent_id="patent/US1")

    def test_custom_description(self, wrapper):
        tool = create_talor_serp_tool(wrapper, description="Custom desc")
        assert tool.description == "Custom desc"


class TestCreateTalorSerpHistoryTool:
    def test_returns_structured_tool(self, wrapper):
        tool = create_talor_serp_history_tool(wrapper)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "talor_serp_history"

    def test_invoke(self, wrapper):
        with patch("langchain_talordata.wrapper.TalorSerpAPIWrapper.history", return_value={"ok": True}) as mock_h:
            tool = create_talor_serp_history_tool(wrapper)
            result = tool.invoke({"page": 1, "page_size": 10})
            mock_h.assert_called_once_with(
                page=1, page_size=10, search_query=None, search_engine=None,
                status="all", start_time=None, end_time=None, timezone=None,
            )


class TestCreateTalorSerpListEnginesTool:
    def test_returns_structured_tool(self, wrapper):
        tool = create_talor_serp_list_engines_tool(wrapper)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "talor_serp_list_engines"

    def test_list_all(self, wrapper):
        tool = create_talor_serp_list_engines_tool(wrapper)
        result = tool.invoke({})
        assert "Total engines:" in result
        assert "google" in result

    def test_inspect_engine(self, wrapper):
        tool = create_talor_serp_list_engines_tool(wrapper)
        result = tool.invoke({"engine": "google"})
        assert "google" in result.lower()
        assert "Parameters:" in result

    def test_unknown_engine(self, wrapper):
        tool = create_talor_serp_list_engines_tool(wrapper)
        result = tool.invoke({"engine": "nonexistent"})
        assert "Unknown engine" in result


class TestCreateTalorSerpStatisticsTool:
    def test_returns_structured_tool(self, wrapper):
        tool = create_talor_serp_statistics_tool(wrapper)
        assert isinstance(tool, StructuredTool)
        assert tool.name == "talor_serp_statistics"

    def test_invoke(self, wrapper):
        with patch("langchain_talordata.wrapper.TalorSerpAPIWrapper.statistics", return_value={"ok": True}) as mock_s:
            tool = create_talor_serp_statistics_tool(wrapper)
            tool.invoke({"start_date": "2024-01-01", "end_date": "2024-01-31"})
            mock_s.assert_called_once_with(
                start_date="2024-01-01", end_date="2024-01-31",
                engines=None, timezone=None,
            )


class TestTalorSerpTool:
    def test_from_api_key(self):
        tool = TalorSerpTool.from_api_key("test-key")
        assert isinstance(tool, StructuredTool)

    def test_from_env(self):
        with patch.dict("os.environ", {"TALOR_API_KEY": "test-key"}):
            tool = TalorSerpTool.from_env()
            assert isinstance(tool, StructuredTool)

    def test_from_wrapper(self, wrapper):
        tool = TalorSerpTool.from_wrapper(wrapper)
        assert isinstance(tool, StructuredTool)

    def test_history_from_env(self):
        with patch.dict("os.environ", {"TALOR_API_KEY": "test-key"}):
            tool = TalorSerpTool.history_from_env()
            assert isinstance(tool, StructuredTool)

    def test_statistics_from_env(self):
        with patch.dict("os.environ", {"TALOR_API_KEY": "test-key"}):
            tool = TalorSerpTool.statistics_from_env()
            assert isinstance(tool, StructuredTool)

    def test_tools_from_env(self):
        with patch.dict("os.environ", {"TALOR_API_KEY": "test-key"}):
            tools = TalorSerpTool.tools_from_env()
            assert isinstance(tools, list)
            assert len(tools) == 4
            names = [t.name for t in tools]
            assert "talor_serp_search" in names
            assert "talor_serp_list_engines" in names
            assert "talor_serp_history" in names
            assert "talor_serp_statistics" in names

    def test_tools_from_api_key(self):
        tools = TalorSerpTool.tools_from_api_key("test-key")
        assert len(tools) == 4
