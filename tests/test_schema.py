"""Tests for langchain_talordata.schema module."""

from langchain_talordata.schema import (
    EngineRegistry,
    EngineSchema,
    Field,
    FieldGroup,
    FieldOption,
    RangeKeys,
)


class TestFieldOption:
    def test_from_dict(self):
        opt = FieldOption.from_dict({"value": "us", "label": "United States"})
        assert opt.value == "us"
        assert opt.label == "United States"
        assert opt.children is None

    def test_from_dict_with_children(self):
        opt = FieldOption.from_dict({
            "value": "a",
            "label": "A",
            "children": [{"value": "a1", "label": "A1"}],
        })
        assert len(opt.children) == 1
        assert opt.children[0].value == "a1"

    def test_from_dict_with_options_key(self):
        opt = FieldOption.from_dict({
            "value": "x",
            "label": "X",
            "options": [{"value": "x1", "label": "X1"}],
        })
        assert opt.children is not None
        assert opt.children[0].value == "x1"


class TestField:
    def test_from_dict(self):
        f = Field.from_dict({"key": "gl", "type": "select", "required": False})
        assert f.key == "gl"
        assert f.type == "select"
        assert f.required is False

    def test_from_dict_required(self):
        f = Field.from_dict({"key": "q", "type": "text", "required": True})
        assert f.required is True

    def test_from_dict_with_options(self):
        f = Field.from_dict({
            "key": "device",
            "type": "select",
            "options": [{"value": "desktop", "label": "Desktop"}],
        })
        assert len(f.options) == 1
        assert f.options[0].value == "desktop"

    def test_from_dict_with_range_keys(self):
        f = Field.from_dict({
            "key": "date_range",
            "type": "date_range",
            "range_keys": {"start": "start_date", "end": "end_date"},
        })
        assert f.range_keys is not None
        assert f.range_keys.start == "start_date"
        assert f.range_keys.end == "end_date"

    def test_default_value(self):
        f = Field.from_dict({"key": "num", "type": "number", "default_value": 10})
        assert f.default_value == 10


class TestFieldGroup:
    def test_from_dict(self):
        g = FieldGroup.from_dict({
            "key": "basic",
            "title": "Basic",
            "fields": [{"key": "q", "type": "text"}],
        })
        assert g.key == "basic"
        assert g.title == "Basic"
        assert len(g.fields) == 1

    def test_collapsible(self):
        g = FieldGroup.from_dict({"key": "adv", "title": "Adv", "collapsible": True})
        assert g.collapsible is True


class TestEngineSchema:
    def test_from_dict(self):
        schema = EngineSchema.from_dict({
            "key": "google",
            "name": "Search",
            "query_field": "q",
            "groups": [
                {
                    "key": "basic",
                    "title": "Basic",
                    "fields": [
                        {"key": "q", "type": "text", "required": True},
                        {"key": "gl", "type": "select", "default_value": "us"},
                    ],
                }
            ],
            "category": {"key": "google", "name": "Google"},
            "is_default_engine": True,
        })
        assert schema.key == "google"
        assert schema.name == "Search"
        assert schema.query_field == "q"
        assert schema.is_default is True
        assert schema.category == "google"
        assert len(schema.groups) == 1

    def test_all_fields(self, google_schema):
        fields = google_schema.all_fields()
        assert len(fields) > 0
        field_keys = [f.key for f in fields]
        assert "q" in field_keys

    def test_field_map(self, google_schema):
        fm = google_schema.field_map()
        assert "q" in fm
        assert fm["q"].type == "text"

    def test_required_fields(self, google_schema):
        req = google_schema.required_fields()
        assert len(req) > 0
        assert all(f.required for f in req)

    def test_to_param_schema(self, google_schema):
        ps = google_schema.to_param_schema()
        assert ps["type"] == "object"
        assert "properties" in ps
        assert "q" in ps["properties"]
        assert "required" in ps
        assert "q" in ps["required"]

    def test_to_description(self, google_schema):
        desc = google_schema.to_description()
        assert "google" in desc.lower()
        assert "Search" in desc

    def test_patents_details_query_field(self, patents_details_schema):
        assert patents_details_schema.query_field == "patent_id"
        req = patents_details_schema.required_fields()
        assert any(f.key == "patent_id" for f in req)


class TestEngineRegistry:
    def test_load_engines(self, registry):
        assert len(registry.engine_keys) > 30

    def test_default_engine(self, registry):
        assert registry.default_engine == "google"

    def test_engine(self, registry):
        schema = registry.engine("google")
        assert schema is not None
        assert schema.key == "google"

    def test_engine_not_found(self, registry):
        assert registry.engine("nonexistent") is None

    def test_engines(self, registry):
        engines = registry.engines()
        assert isinstance(engines, dict)
        assert "google" in engines

    def test_categories(self, registry):
        cats = registry.categories()
        assert "google" in cats
        assert "bing" in cats
        assert "google" in cats["google"]

    def test_all_engines_have_schema(self, registry):
        for key in registry.engine_keys:
            schema = registry.engine(key)
            assert schema is not None, f"Missing schema for {key}"
            assert schema.key == key
            assert schema.name != ""
            # Each schema should produce a valid param schema
            ps = schema.to_param_schema()
            assert ps["type"] == "object"
