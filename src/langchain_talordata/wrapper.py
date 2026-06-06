"""LangChain wrapper for Talor SERP API.

Follows the same API surface as GoogleSerperAPIWrapper from langchain_community,
so users can swap integrations with minimal code changes.

Engine schemas are bundled inside the package for full parameter awareness.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel

from .engines import DEFAULT_ENGINE
from .schema import EngineRegistry, EngineSchema
from .serialize import compact_response_data, serialize

_DEFAULT_ENDPOINT = "https://serpapi.talordata.net/serp/v1/request"

# Singleton registry — lazily loaded
_registry: Optional[EngineRegistry] = None


def _get_registry() -> EngineRegistry:
    global _registry
    if _registry is None:
        _registry = EngineRegistry()
    return _registry


class TalorSerpAPIWrapper(BaseModel):
    """Wrapper for the Talor SERP API.

    Supports 33 search engines across Google, Bing, Yandex and DuckDuckGo.
    Engine schemas are bundled inside the package for full parameter
    validation, serialization, and documentation.

    Example:
        .. code-block:: python

            from langchain_talordata import TalorSerpAPIWrapper

            wrapper = TalorSerpAPIWrapper(talor_api_key="your-token")

            # Basic search
            text = wrapper.run("LangChain tutorial")

            # Search with engine-specific params
            text = wrapper.run(
                "laptop",
                engine="google_shopping",
                min_price="500",
                max_price="1000",
            )

            # Flight search
            text = wrapper.run(
                "flights",
                engine="google_flights",
                departure_id="SFO",
                arrival_id="NRT",
                outbound_date="2025-03-01",
                return_date="2025-03-15",
                adults=2,
            )
    """

    talor_api_key: str = ""
    endpoint: str = _DEFAULT_ENDPOINT
    engine: str = DEFAULT_ENGINE
    gl: str = "us"
    hl: str = "en"
    device: str = "desktop"
    response_mode: str = "compact"
    timeout: int = 150
    k: int = 5

    class Config:
        extra = "forbid"

    def _registry(self) -> EngineRegistry:
        return _get_registry()

    def get_engine_schema(self, engine_key: Optional[str] = None) -> Optional[EngineSchema]:
        """Get the full schema for an engine."""
        key = engine_key or self.engine
        return self._registry().engine(key)

    def list_engines(self) -> List[str]:
        """List all available engine keys."""
        return self._registry().engine_keys

    def engine_description(self, engine_key: str) -> str:
        """Get a human-readable description of an engine's parameters."""
        schema = self._registry().engine(engine_key)
        if schema is None:
            return f"Unknown engine: {engine_key}"
        return schema.to_description()

    def engine_param_schema(self, engine_key: str) -> Optional[Dict[str, Any]]:
        """Get the JSON-Schema-like param definition for an engine."""
        schema = self._registry().engine(engine_key)
        if schema is None:
            return None
        return schema.to_param_schema()

    def _resolve_api_key(self, kwargs: Dict[str, Any]) -> str:
        key = kwargs.pop("talor_api_key", None) or self.talor_api_key
        if not key:
            key = get_from_dict_or_env(
                {"talor_api_key": key}, "talor_api_key", "TALOR_API_KEY"
            )
        return key

    def _build_payload(
        self, query: str = "", engine: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Build the form-encoded payload using engine schema + serialization.

        This mirrors the Go MCP server's logic:
        1. Resolve engine and load its schema
        2. Apply defaults from schema
        3. Serialize params based on field types
        """
        engine_key = engine or kwargs.pop("engine", None) or self.engine

        # Determine the query field name from schema
        schema = self._registry().engine(engine_key)
        query_field = schema.query_field if schema else "q"

        # Build raw params dict
        raw: Dict[str, Any] = {"engine": engine_key}

        # Set query on the correct field
        raw[query_field] = query

        # Set json format (default to "2" for JSON + HTML)
        if "json" not in kwargs:
            raw["json"] = "2"

        # Apply defaults from schema for common params
        defaults = {
            "gl": self.gl,
            "hl": self.hl,
            "device": self.device,
        }
        if schema:
            for f in schema.all_fields():
                if f.key in ("q", "engine", "text") or f.key == query_field:
                    continue
                if f.default_value is not None and f.key not in kwargs and f.key not in defaults:
                    defaults[f.key] = f.default_value

        # Merge: schema defaults < wrapper defaults < user kwargs
        for key, value in defaults.items():
            if key not in kwargs:
                raw[key] = value

        for key, value in kwargs.items():
            if value is not None:
                raw[key] = value

        # Serialize based on schema field types
        if schema:
            raw = serialize(schema, raw)

        return raw

    def results(
        self, query: str = "", engine: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Return raw search results from the Talor SERP API.

        Args:
            query: The search query.
            engine: Override the default engine (e.g. "google_images", "bing_news").
            **kwargs: Engine-specific parameters. Use engine_description(engine)
                or engine_param_schema(engine) to see available params.

        Returns:
            Dict with keys: ok, status, engine, data (parsed JSON), raw (if non-JSON).
        """
        api_key = self._resolve_api_key(kwargs)
        payload = self._build_payload(query, engine, **kwargs)

        response = httpx.post(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "mcp",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        result: Dict[str, Any] = {
            "ok": 200 <= response.status_code < 300,
            "status": response.status_code,
            "engine": payload.get("engine", self.engine),
        }

        try:
            data = response.json()
            result["data"] = data
        except Exception:
            result["raw"] = response.text

        return result

    async def aresults(
        self, query: str = "", engine: Optional[str] = None, **kwargs: Any
    ) -> Dict[str, Any]:
        """Async version of results()."""
        api_key = self._resolve_api_key(kwargs)
        payload = self._build_payload(query, engine, **kwargs)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                self.endpoint,
                data=payload,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "mcp",
                },
            )
            response.raise_for_status()

        result: Dict[str, Any] = {
            "ok": 200 <= response.status_code < 300,
            "status": response.status_code,
            "engine": payload.get("engine", self.engine),
        }

        try:
            data = response.json()
            result["data"] = data
        except Exception:
            result["raw"] = response.text

        return result

    def _unwrap_data(self, data: Any) -> Dict[str, Any]:
        """Unwrap nested API response and json_html format.

        The upstream API can return:
        - Direct: {"organic": [...], "knowledge_graph": {...}}
        - Nested:  {"code": 0, "data": {"organic": [...]}}
        - json_html: {"html": "...", "json": "{...json string...}"}
        """
        if not isinstance(data, dict):
            return {}

        # Handle nested {"code": 0, "data": {...}} wrapper
        if "code" in data and "data" in data and isinstance(data["data"], dict):
            data = data["data"]

        # Handle json_html format: {"html": "...", "json": "{...}"}
        if "json" in data and "html" in data:
            json_field = data["json"]
            if isinstance(json_field, str):
                try:
                    import json as _json
                    data = _json.loads(json_field)
                except (ValueError, TypeError):
                    pass
            elif isinstance(json_field, dict):
                data = json_field

        return data if isinstance(data, dict) else {}

    def _process_response(self, res: Dict[str, Any], k: Optional[int] = None) -> str:
        """Format search results into a readable string for LLM consumption.

        Extracts organic results, knowledge graph, answer boxes, etc.
        """
        k = k or self.k
        snippets = []

        data = res.get("data", {})
        if not data:
            return res.get("raw", "No results found.")

        # Unwrap nested/json_html format
        data = self._unwrap_data(data)

        # In compact mode, strip metadata
        if self.response_mode == "compact":
            data = compact_response_data(data)

        # Organic results
        organic = data.get("organic", [])
        if organic:
            for i, result in enumerate(organic[:k], 1):
                title = result.get("title", "")
                link = result.get("link", "")
                snippet = result.get("snippet", "")
                position = result.get("position", i)
                parts = [f"{position}. {title}"]
                if snippet:
                    parts.append(f"   {snippet}")
                if link:
                    parts.append(f"   URL: {link}")
                snippets.append("\n".join(parts))

        # Knowledge graph
        kg = data.get("knowledge_graph", {})
        if kg:
            title = kg.get("title", "")
            description = kg.get("description", "")
            if title:
                kg_parts = [f"Knowledge Graph: {title}"]
                if description:
                    kg_parts.append(f"  {description}")
                for attr in kg.get("attributes", {}).items():
                    kg_parts.append(f"  {attr[0]}: {attr[1]}")
                snippets.insert(0, "\n".join(kg_parts))

        # Answer box
        answer_box = data.get("answer_box", {})
        if answer_box:
            answer = answer_box.get("answer", answer_box.get("snippet", ""))
            if answer:
                snippets.insert(0, f"Answer: {answer}")

        # AI Overview
        ai_overview = data.get("ai_overview", {})
        if ai_overview:
            overview_text = ai_overview.get("text", "")
            if overview_text:
                snippets.insert(0, f"AI Overview: {overview_text}")

        # Google Flights results
        best_flights = data.get("best_flights", [])
        if best_flights:
            for i, option in enumerate(best_flights[:3], 1):
                trip_type = option.get("type", "trip")
                legs = option.get("flight", [])
                parts = [f"Option {i} ({trip_type}):"]
                for leg in legs:
                    airline = leg.get("airline", "")
                    flight_no = leg.get("flight_number", "")
                    dep = leg.get("departure_airport", {})
                    arr = leg.get("arrival_airport", {})
                    dep_info = f"{dep.get('id', '')} {dep.get('time', '')}"
                    arr_info = f"{arr.get('id', '')} {arr.get('time', '')}"
                    parts.append(f"  {airline} {flight_no}: {dep_info} -> {arr_info}")
                snippets.append("\n".join(parts))

        if not snippets:
            return str(data) if data else "No results found."

        return "\n\n".join(snippets)

    def run(self, query: str = "", engine: Optional[str] = None, **kwargs: Any) -> str:
        """Search and return formatted text results.

        This is the primary method for LLM/Agent consumption.
        """
        k = kwargs.pop("k", None) or self.k
        res = self.results(query, engine, **kwargs)
        return self._process_response(res, k)

    async def arun(
        self, query: str = "", engine: Optional[str] = None, **kwargs: Any
    ) -> str:
        """Async version of run()."""
        k = kwargs.pop("k", None) or self.k
        res = await self.aresults(query, engine, **kwargs)
        return self._process_response(res, k)

    # ── History ──────────────────────────────────────────────────────────

    _history_endpoint = "https://api.talordata.com/accounts/v1/serp/mcp/history"

    def history(
        self,
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None,
        search_engine: Optional[str] = None,
        status: str = "all",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query SERP search history.

        Args:
            page: Page number (default 1).
            page_size: Page size (default 20).
            search_query: Filter by search keyword.
            search_engine: Filter by engine name.
            status: "all", "success", or "error".
            start_time: Start unix timestamp (seconds).
            end_time: End unix timestamp (seconds).
            timezone: Timezone header, e.g. "Asia/Shanghai" or "+08:00".

        Returns:
            Dict with ok, status, data.
        """
        api_key = self._resolve_api_key({})
        params: Dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "status": status,
            "api_token_id": api_key,
        }
        if search_query:
            params["search_query"] = search_query
        if search_engine:
            params["search_engine"] = search_engine
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "mcp",
        }
        if timezone:
            headers["X-Time-Zone"] = timezone

        response = httpx.get(
            self._history_endpoint,
            params=params,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result: Dict[str, Any] = {
            "ok": 200 <= response.status_code < 300,
            "status": response.status_code,
        }
        try:
            result["data"] = response.json()
        except Exception:
            result["raw"] = response.text
        return result

    async def ahistory(
        self,
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None,
        search_engine: Optional[str] = None,
        status: str = "all",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of history()."""
        api_key = self._resolve_api_key({})
        params: Dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "status": status,
            "api_token_id": api_key,
        }
        if search_query:
            params["search_query"] = search_query
        if search_engine:
            params["search_engine"] = search_engine
        if start_time:
            params["start_time"] = start_time
        if end_time:
            params["end_time"] = end_time

        headers: Dict[str, str] = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/x-www-form-urlencoded",
            "Origin": "mcp",
        }
        if timezone:
            headers["X-Time-Zone"] = timezone

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self._history_endpoint,
                params=params,
                headers=headers,
            )
            response.raise_for_status()

        result: Dict[str, Any] = {
            "ok": 200 <= response.status_code < 300,
            "status": response.status_code,
        }
        try:
            result["data"] = response.json()
        except Exception:
            result["raw"] = response.text
        return result

    # ── Statistics ───────────────────────────────────────────────────────

    _statistics_endpoint = "https://api.talordata.com/pay_package_view/v1/serp/mcp/statistics"

    def statistics(
        self,
        start_date: str,
        end_date: str,
        engines: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Query SERP usage statistics.

        Args:
            start_date: Start date in YYYY-MM-DD (required).
            end_date: End date in YYYY-MM-DD (required).
            engines: Comma-separated engine keys, e.g. "google,bing".
            timezone: Timezone offset, e.g. "+08:00".

        Returns:
            Dict with ok, status, data.
        """
        api_key = self._resolve_api_key({})
        params: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "api_token_id": api_key,
        }
        if engines:
            params["engines"] = engines
        if timezone:
            params["timezone"] = timezone

        response = httpx.get(
            self._statistics_endpoint,
            params=params,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "mcp",
            },
            timeout=self.timeout,
        )
        response.raise_for_status()

        result: Dict[str, Any] = {
            "ok": 200 <= response.status_code < 300,
            "status": response.status_code,
        }
        try:
            result["data"] = response.json()
        except Exception:
            result["raw"] = response.text
        return result

    async def astatistics(
        self,
        start_date: str,
        end_date: str,
        engines: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Async version of statistics()."""
        api_key = self._resolve_api_key({})
        params: Dict[str, Any] = {
            "start_date": start_date,
            "end_date": end_date,
            "api_token_id": api_key,
        }
        if engines:
            params["engines"] = engines
        if timezone:
            params["timezone"] = timezone

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(
                self._statistics_endpoint,
                params=params,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Origin": "mcp",
                },
            )
            response.raise_for_status()

        result: Dict[str, Any] = {
            "ok": 200 <= response.status_code < 300,
            "status": response.status_code,
        }
        try:
            result["data"] = response.json()
        except Exception:
            result["raw"] = response.text
        return result
