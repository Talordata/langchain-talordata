"""Tests for langchain_talordata.wrapper module."""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import httpx

from langchain_talordata.wrapper import TalorSerpAPIWrapper


@pytest.fixture
def wrapper():
    return TalorSerpAPIWrapper(talor_api_key="test-key")


class TestBuildPayload:
    def test_basic_payload(self, wrapper):
        payload = wrapper._build_payload("test query")
        assert payload["engine"] == "google"
        assert payload["q"] == "test query"
        assert payload["json"] == "1"

    def test_engine_override(self, wrapper):
        payload = wrapper._build_payload("test", engine="bing")
        assert payload["engine"] == "bing"

    def test_default_params(self, wrapper):
        payload = wrapper._build_payload("test")
        assert payload["gl"] == "us"
        assert payload["hl"] == "en"
        assert payload["device"] == "desktop"

    def test_user_params_override(self, wrapper):
        payload = wrapper._build_payload("test", gl="cn", hl="zh")
        assert payload["gl"] == "cn"
        assert payload["hl"] == "zh"

    def test_json_override(self, wrapper):
        payload = wrapper._build_payload("test", json="2")
        assert payload["json"] == "2"

    def test_patent_id_field(self, wrapper):
        payload = wrapper._build_payload(
            "",
            engine="google_patents_details",
            parent_id="patent/US11734097B1/en",
        )
        assert "patent_id" in payload
        assert payload["engine"] == "google_patents_details"

    def test_query_not_in_kwargs(self, wrapper):
        payload = wrapper._build_payload("hello")
        assert "query" not in payload
        assert payload["q"] == "hello"


class TestResolveApiKey:
    def test_from_instance(self, wrapper):
        key = wrapper._resolve_api_key({})
        assert key == "test-key"

    def test_from_kwargs(self, wrapper):
        key = wrapper._resolve_api_key({"talor_api_key": "override-key"})
        assert key == "override-key"

    def test_from_env(self):
        wrapper = TalorSerpAPIWrapper()
        with patch.dict("os.environ", {"TALOR_API_KEY": "env-key"}):
            key = wrapper._resolve_api_key({})
            assert key == "env-key"


class TestResults:
    @patch("langchain_talordata.wrapper.httpx.post")
    def test_results_success(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "organic": [{"title": "Test", "link": "https://test.com", "snippet": "Snippet"}]
        }
        mock_post.return_value = mock_resp

        result = wrapper.results("test query")
        assert result["ok"] is True
        assert result["status"] == 200
        assert result["engine"] == "google"
        assert "data" in result

    @patch("langchain_talordata.wrapper.httpx.post")
    def test_results_calls_correct_endpoint(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        wrapper.results("test")
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert call_args[0][0] == wrapper.endpoint

    @patch("langchain_talordata.wrapper.httpx.post")
    def test_results_auth_header(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        wrapper.results("test")
        call_kwargs = mock_post.call_args
        headers = call_kwargs[1]["headers"]
        assert headers["Authorization"] == "Bearer test-key"
        assert headers["Origin"] == "langchain_py"

    @patch("langchain_talordata.wrapper.httpx.post")
    def test_results_json_parse_error(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.side_effect = ValueError("bad json")
        mock_resp.text = "not json"
        mock_post.return_value = mock_resp

        result = wrapper.results("test")
        assert "raw" in result
        assert result["raw"] == "not json"


class TestRun:
    @patch("langchain_talordata.wrapper.httpx.post")
    def test_run_returns_string(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "organic": [
                {"title": "Result 1", "link": "https://a.com", "snippet": "Snippet 1", "position": 1}
            ]
        }
        mock_post.return_value = mock_resp

        result = wrapper.run("test")
        assert isinstance(result, str)
        assert "Result 1" in result
        assert "https://a.com" in result

    @patch("langchain_talordata.wrapper.httpx.post")
    def test_run_knowledge_graph(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "knowledge_graph": {
                "title": "Batman",
                "description": "Fictional superhero",
                "attributes": {"Created by": "Bob Kane"},
            }
        }
        mock_post.return_value = mock_resp

        result = wrapper.run("batman")
        assert "Batman" in result
        assert "Fictional superhero" in result

    @patch("langchain_talordata.wrapper.httpx.post")
    def test_run_answer_box(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {
            "answer_box": {"answer": "42"}
        }
        mock_post.return_value = mock_resp

        result = wrapper.run("meaning of life")
        assert "42" in result

    @patch("langchain_talordata.wrapper.httpx.post")
    def test_run_no_results(self, mock_post, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_post.return_value = mock_resp

        result = wrapper.run("xyznonexistent")
        assert isinstance(result, str)


class TestEngineInfo:
    def test_list_engines(self, wrapper):
        engines = wrapper.list_engines()
        assert "google" in engines
        assert len(engines) > 30

    def test_engine_description(self, wrapper):
        desc = wrapper.engine_description("google")
        assert "google" in desc.lower()

    def test_engine_description_unknown(self, wrapper):
        desc = wrapper.engine_description("nonexistent")
        assert "Unknown" in desc

    def test_engine_param_schema(self, wrapper):
        schema = wrapper.engine_param_schema("google")
        assert schema is not None
        assert schema["type"] == "object"

    def test_engine_param_schema_unknown(self, wrapper):
        assert wrapper.engine_param_schema("nonexistent") is None

    def test_get_engine_schema(self, wrapper):
        schema = wrapper.get_engine_schema("google")
        assert schema is not None
        assert schema.key == "google"


class TestHistory:
    @patch("langchain_talordata.wrapper.httpx.get")
    def test_history_success(self, mock_get, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"data": []}
        mock_get.return_value = mock_resp

        result = wrapper.history(page=1, page_size=10)
        assert result["ok"] is True

    @patch("langchain_talordata.wrapper.httpx.get")
    def test_history_params(self, mock_get, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        wrapper.history(
            page=2, page_size=50, search_query="test",
            search_engine="google", status="success",
            start_time=1000, end_time=2000,
        )
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"]
        assert params["page"] == 2
        assert params["page_size"] == 50
        assert params["search_query"] == "test"
        assert params["search_engine"] == "google"
        assert params["status"] == "success"

    @patch("langchain_talordata.wrapper.httpx.get")
    def test_history_timezone(self, mock_get, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        wrapper.history(timezone="Asia/Shanghai")
        call_kwargs = mock_get.call_args
        headers = call_kwargs[1]["headers"]
        assert headers["X-Time-Zone"] == "Asia/Shanghai"


class TestStatistics:
    @patch("langchain_talordata.wrapper.httpx.get")
    def test_statistics_success(self, mock_get, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {"total_requests": 100}
        mock_get.return_value = mock_resp

        result = wrapper.statistics("2024-01-01", "2024-01-31")
        assert result["ok"] is True

    @patch("langchain_talordata.wrapper.httpx.get")
    def test_statistics_params(self, mock_get, wrapper):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.raise_for_status = MagicMock()
        mock_resp.json.return_value = {}
        mock_get.return_value = mock_resp

        wrapper.statistics("2024-01-01", "2024-01-31", engines="google,bing", timezone="+08:00")
        call_kwargs = mock_get.call_args
        params = call_kwargs[1]["params"]
        assert params["engines"] == "google,bing"
        assert params["timezone"] == "+08:00"


class TestResponseProcessing:
    def test_unwrap_nested(self, wrapper):
        data = {"code": 0, "data": {"organic": []}}
        result = wrapper._unwrap_data(data)
        assert "organic" in result

    def test_unwrap_double_nested(self, wrapper):
        data = {"code": 0, "data": {"code": 0, "data": {"organic": [{"title": "A"}]}}}
        result = wrapper._unwrap_data(data)
        assert "organic" in result
        assert len(result["organic"]) == 1

    def test_unwrap_json_html(self, wrapper):
        inner = {"organic": [{"title": "A"}]}
        data = {"json": json.dumps(inner), "html": "<html>"}
        result = wrapper._unwrap_data(data)
        assert "organic" in result

    def test_unwrap_json_html_dict(self, wrapper):
        inner = {"organic": [{"title": "A"}]}
        data = {"json": inner, "html": "<html>"}
        result = wrapper._unwrap_data(data)
        assert "organic" in result

    def test_unwrap_non_dict(self, wrapper):
        assert wrapper._unwrap_data("string") == {}
        assert wrapper._unwrap_data(None) == {}

    def test_compact_mode(self):
        wrapper = TalorSerpAPIWrapper(response_mode="compact")
        data = {
            "organic": [{"title": "A", "link": "http://a.com"}],
            "search_metadata": {"engine": "google"},
            "search_information": {"total": 100},
        }
        result = wrapper._process_response({"data": data})
        assert "A" in result
        assert "search_metadata" not in result
