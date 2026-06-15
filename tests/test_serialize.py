"""Tests for langchain_talordata.serialize module."""

from langchain_talordata.schema import Field, FieldGroup, EngineSchema
from langchain_talordata.serialize import (
    compact_response_data,
    serialize,
    _bool_string,
    _split_list,
    _join_list,
    _stringify,
    _is_empty,
    _serialize_country_restrict,
    _split_date_range,
)


def _make_schema(key: str, fields: list) -> EngineSchema:
    """Helper to build a minimal EngineSchema for testing."""
    return EngineSchema.from_dict({
        "key": key,
        "name": key,
        "query_field": "q",
        "groups": [{"key": "g", "title": "G", "fields": fields}],
    })


class TestBoolString:
    def test_true_bool(self):
        assert _bool_string(True) == "true"

    def test_false_bool(self):
        assert _bool_string(False) == "false"

    def test_string_true(self):
        assert _bool_string("true") == "true"
        assert _bool_string("True") == "true"
        assert _bool_string("1") == "true"
        assert _bool_string("yes") == "true"
        assert _bool_string("on") == "true"

    def test_string_false(self):
        assert _bool_string("false") == "false"
        assert _bool_string("0") == "false"
        assert _bool_string("no") == "false"
        assert _bool_string("") == "false"

    def test_int_zero(self):
        assert _bool_string(0) == "false"

    def test_int_nonzero(self):
        assert _bool_string(1) == "true"
        assert _bool_string(-1) == "true"

    def test_none(self):
        assert _bool_string(None) == "false"


class TestSplitList:
    def test_list_input(self):
        assert _split_list(["a", "b"]) == ["a", "b"]

    def test_string_input(self):
        assert _split_list("a,b,c") == ["a", "b", "c"]

    def test_empty_string(self):
        assert _split_list("") == []

    def test_none(self):
        assert _split_list(None) == []

    def test_single_string(self):
        assert _split_list("abc") == ["abc"]


class TestJoinList:
    def test_list(self):
        assert _join_list(["a", "b"], ",") == "a,b"

    def test_with_spaces(self):
        assert _join_list([" a ", " b "], ",") == "a,b"

    def test_empty_items(self):
        assert _join_list(["a", "", "b"], ",") == "a,b"


class TestStringify:
    def test_string(self):
        assert _stringify("hello") == "hello"

    def test_int(self):
        assert _stringify(42) == "42"

    def test_none(self):
        assert _stringify(None) == ""

    def test_float(self):
        assert _stringify(3.14) == "3.14"


class TestIsEmpty:
    def test_none(self):
        assert _is_empty(None) is True

    def test_empty_string(self):
        assert _is_empty("") is True
        assert _is_empty("  ") is True

    def test_non_empty(self):
        assert _is_empty("hello") is False

    def test_zero(self):
        assert _is_empty(0) is False


class TestSplitDateRange:
    def test_two_items(self):
        assert _split_date_range(["2024-01-01", "2024-12-31"]) == ("2024-01-01", "2024-12-31")

    def test_one_item(self):
        assert _split_date_range(["2024-01-01"]) == ("2024-01-01", "")

    def test_empty(self):
        assert _split_date_range([]) == ("", "")

    def test_string(self):
        assert _split_date_range("2024-01-01,2024-12-31") == ("2024-01-01", "2024-12-31")


class TestSerializeCountryRestrict:
    def test_list(self):
        result = _serialize_country_restrict(["us", "uk"])
        assert "countryUS" in result
        assert "countryUK" in result
        assert "|" in result

    def test_already_prefixed(self):
        result = _serialize_country_restrict(["countryUS"])
        assert result == "countryUS"

    def test_empty(self):
        assert _serialize_country_restrict([]) == ""


class TestSerialize:
    def test_switch_true(self):
        schema = _make_schema("test", [
            {"key": "no_cache", "type": "switch", "default_value": False},
        ])
        result = serialize(schema, {"q": "test", "no_cache": True})
        assert result["no_cache"] == "true"

    def test_switch_false(self):
        schema = _make_schema("test", [
            {"key": "no_cache", "type": "switch", "default_value": False},
        ])
        result = serialize(schema, {"q": "test", "no_cache": False})
        assert result["no_cache"] == "false"

    def test_number(self):
        schema = _make_schema("test", [
            {"key": "num", "type": "number", "default_value": 10},
        ])
        result = serialize(schema, {"q": "test", "num": 20})
        assert result["num"] == "20"

    def test_number_empty(self):
        schema = _make_schema("test", [
            {"key": "num", "type": "number", "default_value": 10},
        ])
        result = serialize(schema, {"q": "test", "num": None})
        assert result["num"] == ""

    def test_tags(self):
        schema = _make_schema("test", [
            {"key": "lr", "type": "tags", "options": [{"value": "en", "label": "English"}]},
        ])
        result = serialize(schema, {"q": "test", "lr": ["en", "fr"]})
        assert result["lr"] == "en,fr"

    def test_tags_cr(self):
        schema = _make_schema("test", [
            {"key": "cr", "type": "tags", "options": [{"value": "us", "label": "US"}]},
        ])
        result = serialize(schema, {"q": "test", "cr": ["us"]})
        assert result["cr"] == "countryUS"

    def test_date_range(self):
        schema = _make_schema("test", [
            {
                "key": "date_range",
                "type": "date_range",
                "range_keys": {"start": "start_date", "end": "end_date"},
            },
        ])
        result = serialize(schema, {"q": "test", "date_range": ["2024-01-01", "2024-12-31"]})
        assert result["start_date"] == "2024-01-01"
        assert result["end_date"] == "2024-12-31"
        assert "date_range" not in result

    def test_none_values_filtered(self):
        schema = _make_schema("test", [
            {"key": "opt", "type": "select", "default_value": "a"},
        ])
        result = serialize(schema, {"q": "test"})
        assert "opt" not in result

    def test_null_schema(self):
        assert serialize(None, {"q": "test"}) == {}

    def test_empty_values(self):
        schema = _make_schema("test", [{"key": "q", "type": "text"}])
        assert serialize(schema, {}) == {}

    def test_flight_iata_normalize(self):
        schema = _make_schema("google_flights", [
            {"key": "departure_id", "type": "text"},
            {"key": "arrival_id", "type": "text"},
        ])
        result = serialize(schema, {
            "q": "flights",
            "departure_id": "sfo",
            "arrival_id": "nrt",
        })
        assert result["departure_id"] == "SFO"
        assert result["arrival_id"] == "NRT"

    def test_flight_iata_already_upper(self):
        schema = _make_schema("google_flights", [
            {"key": "departure_id", "type": "text"},
        ])
        result = serialize(schema, {"q": "flights", "departure_id": "SFO"})
        assert result["departure_id"] == "SFO"


class TestCompactResponseData:
    def test_removes_metadata(self):
        data = {
            "organic": [{"title": "A"}],
            "search_metadata": {"engine": "google"},
            "search_parameters": {"q": "test"},
            "search_information": {"total_results": 100},
            "pagination": {"next": "url"},
        }
        result = compact_response_data(data)
        assert "organic" in result
        assert "search_metadata" not in result
        assert "search_parameters" not in result
        assert "search_information" not in result
        assert "pagination" not in result

    def test_non_dict_passthrough(self):
        assert compact_response_data("string") == "string"
        assert compact_response_data(42) == 42

    def test_empty_dict(self):
        assert compact_response_data({}) == {}
