"""Engine schema loader — reads bundled JSON schemas from the data/ directory.

Mirrors internal/engines/registry.go to provide Python-side access to
all 33 engine parameter definitions. The JSON files are bundled with
the package so it works as a standalone PyPI package.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from importlib import resources
from typing import Any, Dict, List, Optional


@dataclass
class RangeKeys:
    start: str
    end: str


@dataclass
class FieldOption:
    value: Any
    label: str
    children: Optional[List["FieldOption"]] = None

    @classmethod
    def from_dict(cls, d: dict) -> "FieldOption":
        children = None
        if "children" in d and d["children"]:
            children = [cls.from_dict(c) for c in d["children"]]
        elif "options" in d and d["options"]:
            children = [cls.from_dict(c) for c in d["options"]]
        return cls(value=d.get("value", ""), label=d.get("label", ""), children=children)


@dataclass
class Field:
    key: str
    type: str  # text, select, switch, tags, number, date_range, date
    required: bool = False
    label: str = ""
    help: str = ""
    default_value: Any = None
    options: List[FieldOption] = field(default_factory=list)
    range_keys: Optional[RangeKeys] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Field":
        range_keys = None
        if "range_keys" in d and d["range_keys"]:
            rk = d["range_keys"]
            range_keys = RangeKeys(start=rk.get("start", ""), end=rk.get("end", ""))

        options = []
        if "options" in d:
            options = [FieldOption.from_dict(o) for o in d["options"]]

        return cls(
            key=d.get("key", ""),
            type=d.get("type", "text"),
            required=d.get("required", False),
            label=d.get("label", d.get("key", "")),
            help=d.get("help", ""),
            default_value=d.get("default_value"),
            options=options,
            range_keys=range_keys,
        )


@dataclass
class FieldGroup:
    key: str
    title: str
    collapsible: bool = False
    fields: List[Field] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "FieldGroup":
        fields = [Field.from_dict(f) for f in d.get("fields", [])]
        return cls(
            key=d.get("key", ""),
            title=d.get("title", d.get("key", "")),
            collapsible=d.get("collapsible", False),
            fields=fields,
        )


@dataclass
class EngineSchema:
    key: str
    name: str
    query_field: str = "q"
    groups: List[FieldGroup] = field(default_factory=list)
    category: str = ""
    is_default: bool = False

    @classmethod
    def from_dict(cls, d: dict) -> "EngineSchema":
        groups = [FieldGroup.from_dict(g) for g in d.get("groups", [])]
        cat = d.get("category", {})
        if isinstance(cat, dict):
            cat = cat.get("key", "")
        return cls(
            key=d.get("key", ""),
            name=d.get("name", ""),
            query_field=d.get("query_field", "q"),
            groups=groups,
            category=cat,
            is_default=d.get("is_default_engine", False),
        )

    def all_fields(self) -> List[Field]:
        """Flatten all fields across groups."""
        fields = []
        for group in self.groups:
            fields.extend(group.fields)
        return fields

    def field_map(self) -> Dict[str, Field]:
        """Get field lookup by key."""
        return {f.key: f for f in self.all_fields()}

    def required_fields(self) -> List[Field]:
        """Get required fields only."""
        return [f for f in self.all_fields() if f.required]

    def to_param_schema(self) -> Dict[str, Any]:
        """Convert to a JSON-Schema-like dict for LangChain tool input."""
        props: Dict[str, Any] = {}
        required_keys: List[str] = []

        for f in self.all_fields():
            prop: Dict[str, Any] = {"description": f.help or f.label}

            if f.type == "select" and f.options:
                values = [str(o.value) for o in f.options if o.value != ""]
                if values:
                    prop["enum"] = values
                prop["type"] = "string"
                if f.default_value:
                    prop["default"] = f.default_value

            elif f.type == "switch":
                prop["type"] = "boolean"
                if f.default_value is not None:
                    prop["default"] = f.default_value

            elif f.type == "tags" and f.options:
                values = [str(o.value) for o in f.options]
                prop["type"] = "array"
                prop["items"] = {"type": "string", "enum": values}
                prop["description"] = f"{f.help} (select one or more)"

            elif f.type == "number":
                prop["type"] = "number"
                if f.default_value is not None:
                    prop["default"] = f.default_value

            elif f.type == "date_range":
                prop["type"] = "array"
                prop["items"] = {"type": "string", "format": "date"}
                prop["description"] = f"{f.help} (array of [start_date, end_date])"

            elif f.type == "date":
                prop["type"] = "string"
                prop["format"] = "date"
                prop["description"] = f"{f.help} (YYYY-MM-DD)"

            else:
                # text or unknown
                prop["type"] = "string"
                if f.default_value:
                    prop["default"] = str(f.default_value)

            props[f.key] = prop
            if f.required:
                required_keys.append(f.key)

        schema: Dict[str, Any] = {
            "type": "object",
            "properties": props,
        }
        if required_keys:
            schema["required"] = required_keys

        return schema

    def to_description(self) -> str:
        """Generate a human-readable description of all available parameters."""
        lines = [f"Engine: {self.name} ({self.key})"]
        lines.append(f"Query field: {self.query_field}")
        lines.append("")

        for group in self.groups:
            lines.append(f"[{group.title}]")
            for f in group.fields:
                req = " (required)" if f.required else ""
                default = f" (default: {f.default_value})" if f.default_value else ""
                type_info = f.type
                if f.type == "select" and f.options:
                    vals = [str(o.value) for o in f.options[:10]]
                    extra = f"..." if len(f.options) > 10 else ""
                    type_info = f"select({', '.join(vals)}{extra})"
                elif f.type == "switch":
                    type_info = "boolean"
                elif f.type == "tags" and f.options:
                    vals = [str(o.value) for o in f.options[:5]]
                    extra = f"..." if len(f.options) > 5 else ""
                    type_info = f"tags({', '.join(vals)}{extra})"
                elif f.type == "number":
                    type_info = "number"
                elif f.type == "date_range":
                    type_info = "date_range [start, end]"

                lines.append(
                    f"  {f.key}: {type_info}{req}{default} — {f.help or f.label}"
                )
            lines.append("")

        return "\n".join(lines)


class EngineRegistry:
    """Loads and caches all engine schemas from bundled data/.

    The engine JSON schemas are bundled inside the package at
    langchain_talordata/data/. This makes the package fully
    self-contained for PyPI distribution.
    """

    def __init__(self) -> None:
        self._schemas: Dict[str, EngineSchema] = {}
        self._index: Optional[dict] = None
        self._load()

    def _load(self):
        data_dir = resources.files("langchain_talordata") / "data"

        index_text = (data_dir / "index.json").read_text(encoding="utf-8")
        self._index = json.loads(index_text)

        if not self._index:
            return

        for engine_ref in self._index.get("engines", []):
            key = engine_ref.get("key", "")
            filename = engine_ref.get("file", "")
            if not key or not filename:
                continue

            schema_text = (data_dir / filename).read_text(encoding="utf-8")
            data = json.loads(schema_text)

            schema = EngineSchema.from_dict(data)
            if not schema.key:
                schema.key = key
            if not schema.name:
                schema.name = engine_ref.get("name", key)
            if not schema.category:
                schema.category = engine_ref.get("category", "")

            self._schemas[key] = schema

    @property
    def default_engine(self) -> str:
        if self._index:
            return self._index.get("default_engine", "google")
        return "google"

    @property
    def engine_keys(self) -> List[str]:
        return sorted(self._schemas.keys())

    def engine(self, key: str) -> Optional[EngineSchema]:
        return self._schemas.get(key)

    def engines(self) -> Dict[str, EngineSchema]:
        return dict(self._schemas)

    def categories(self) -> Dict[str, List[str]]:
        if not self._index:
            return {}
        result: Dict[str, List[str]] = {}
        for cat in self._index.get("categories", []):
            cat_key = cat.get("key", "")
            result[cat_key] = [
                e["key"]
                for e in self._index.get("engines", [])
                if e.get("category") == cat_key
            ]
        return result
