"""Parameter serialization — mirrors internal/serp/serialize.go.

Transforms user-facing parameter values into the format expected by
the upstream Talor SERP API.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .schema import EngineSchema

_FLIGHT_CODE_RE = re.compile(r"^[A-Za-z]{3}$")


def serialize(schema: EngineSchema, values: Dict[str, Any]) -> Dict[str, Any]:
    """Serialize user-facing params into upstream API format.

    Mirrors the Go Serialize() function — handles type conversions for
    date_range, tags, switch, number, date, and the google_flights
    IATA code normalization.
    """
    if schema is None or not values:
        return {}

    out = dict(values)

    for group in schema.groups:
        for field in group.fields:
            if field.key not in values:
                continue

            value = values[field.key]

            if field.type == "date_range" and field.range_keys:
                start, end = _split_date_range(value)
                out[field.range_keys.start] = start
                out[field.range_keys.end] = end
                del out[field.key]

            elif field.type == "tags":
                if field.key == "cr":
                    out[field.key] = _serialize_country_restrict(value)
                else:
                    out[field.key] = _join_list(value, ",")

            elif field.type == "switch":
                out[field.key] = _bool_string(value)

            elif field.type == "date":
                out[field.key] = _stringify(value)

            elif field.type == "number":
                if _is_empty(value):
                    out[field.key] = ""
                else:
                    out[field.key] = _stringify(value)

    _normalize_engine_params(schema, out)
    return out


def compact_response_data(data: Any) -> Any:
    """Remove metadata fields from response (compact mode)."""
    if not isinstance(data, dict):
        return data

    out = dict(data)
    for key in [
        "search_metadata",
        "search_parameters",
        "search_information",
        "pagination",
        "serpapi_pagination",
    ]:
        out.pop(key, None)
    return out


# ── Internal helpers ─────────────────────────────────────────────────────────


def _split_date_range(value: Any) -> tuple:
    items = _split_list(value)
    if not items:
        return ("", "")
    if len(items) == 1:
        return (items[0].strip(), "")
    return (items[0].strip(), items[1].strip())


def _join_list(value: Any, sep: str) -> str:
    items = _split_list(value)
    return sep.join(item.strip() for item in items if item.strip())


def _serialize_country_restrict(value: Any) -> str:
    items = _split_list(value)
    out = []
    for item in items:
        trimmed = item.strip()
        if not trimmed:
            continue
        if not trimmed.lower().startswith("country"):
            trimmed = "country" + trimmed.upper()
        out.append(trimmed)
    return "|".join(out)


def _bool_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return "true"
        return "false"
    if isinstance(value, (int, float)):
        return "true" if value != 0 else "false"
    return "false"


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _is_empty(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _split_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        if not value:
            return []
        return value.split(",")
    if value is None:
        return []
    return [str(value)]


def _normalize_engine_params(schema: EngineSchema, out: Dict[str, Any]):
    """Google Flights IATA code normalization."""
    if schema.key != "google_flights":
        return

    for key in ("departure_id", "arrival_id"):
        if key in out:
            out[key] = _normalize_flight_ids(out[key])


def _normalize_flight_ids(value: Any) -> str:
    items = _split_list(value)
    for i, item in enumerate(items):
        trimmed = item.strip()
        if _FLIGHT_CODE_RE.match(trimmed) and not trimmed.startswith("/"):
            items[i] = trimmed.upper()
        else:
            items[i] = trimmed
    return ",".join(items)
