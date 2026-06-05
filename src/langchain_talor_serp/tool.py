"""LangChain Tool adapter for Talor SERP API.

Provides pre-built tools that can be directly bound to LangChain agents.
Engine schemas are used to generate rich descriptions and parameter metadata.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, field_validator

from .schema import EngineRegistry
from .wrapper import TalorSerpAPIWrapper


def _build_engines_summary(registry: EngineRegistry) -> str:
    """Build a compact summary of all engines and their key params."""
    lines = []
    for key in registry.engine_keys:
        schema = registry.engine(key)
        if schema is None:
            continue
        req = [f.key for f in schema.required_fields() if f.key not in ("q", "text")]
        req_str = f" [required: {', '.join(req)}]" if req else ""
        lines.append(f"  {key}: {schema.name}{req_str}")
    return "\n".join(lines)


def _build_engine_params_description(schema, max_options: int = 8) -> str:
    """Build a compact param description for a single engine."""
    lines = []
    for group in schema.groups:
        for f in group.fields:
            if f.key in ("q", "text", "engine"):
                continue
            req = " *required*" if f.required else ""
            default = f" (default: {f.default_value})" if f.default_value else ""
            type_info = f.type

            if f.type == "select" and f.options:
                vals = [str(o.value) for o in f.options[:max_options]]
                extra = f" (+{len(f.options) - max_options} more)" if len(f.options) > max_options else ""
                type_info = f"select: {', '.join(vals)}{extra}"
            elif f.type == "switch":
                type_info = "boolean"
            elif f.type == "tags" and f.options:
                vals = [str(o.value) for o in f.options[:max_options]]
                extra = f" (+{len(f.options) - max_options} more)" if len(f.options) > max_options else ""
                type_info = f"tags: {', '.join(vals)}{extra}"
            elif f.type == "number":
                type_info = "number"
            elif f.type == "date_range":
                type_info = "array [start_date, end_date]"
            elif f.type == "date":
                type_info = "string YYYY-MM-DD"

            lines.append(f"    {f.key} ({type_info}){req}{default}: {f.help or f.label}")

    return "\n".join(lines)


class TalorSerpSearchInput(BaseModel):
    """Input for a Talor SERP search."""

    query: str = Field(description="The search query to execute.")
    engine: Optional[str] = Field(
        default=None,
        description=(
            "Search engine key. Common engines: "
            "google, google_web, google_images, google_news, google_shopping, "
            "google_maps, google_flights, google_hotels, google_scholar, "
            "google_jobs, google_videos, google_trends, google_finance, "
            "bing, bing_images, bing_news, bing_shopping, bing_videos, "
            "yandex, duckduckgo. "
            "Use the talor_serp_list_engines tool to see all options."
        ),
    )
    params: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Engine-specific parameters as a JSON object. "
            "Common params: gl (country code, e.g. 'us'), hl (language, e.g. 'en'), "
            "device ('desktop'/'mobile'), location, no_cache (boolean). "
            "Each engine has unique params — check the engine schema for details. "
            'Example: {"gl": "cn", "hl": "zh", "device": "mobile"}'
        ),
    )

    @field_validator("params", mode="before")
    @classmethod
    def _parse_params_json(cls, value: Any) -> Any:
        if value is None or isinstance(value, dict):
            return value
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError("params JSON must decode to an object")
            return parsed
        return value


class TalorSerpHistoryInput(BaseModel):
    """Input for querying SERP history."""

    page: int = Field(default=1, description="Page number.")
    page_size: int = Field(default=20, description="Page size.")
    search_query: Optional[str] = Field(
        default=None,
        description="Filter by search keyword.",
    )
    search_engine: Optional[str] = Field(
        default=None,
        description="Filter by engine name, such as 'google' or 'bing'.",
    )
    status: str = Field(
        default="all",
        description="Status filter: all, success, or error.",
    )
    start_time: Optional[int] = Field(
        default=None,
        description="Start unix timestamp in seconds.",
    )
    end_time: Optional[int] = Field(
        default=None,
        description="End unix timestamp in seconds.",
    )
    timezone: Optional[str] = Field(
        default=None,
        description="Timezone header, e.g. 'Asia/Shanghai' or '+08:00'.",
    )


class TalorSerpStatisticsInput(BaseModel):
    """Input for querying SERP usage statistics."""

    start_date: str = Field(description="Start date in YYYY-MM-DD.")
    end_date: str = Field(description="End date in YYYY-MM-DD.")
    engines: Optional[str] = Field(
        default=None,
        description="Comma-separated engine keys, e.g. 'google,bing'.",
    )
    timezone: Optional[str] = Field(
        default=None,
        description="Timezone offset, e.g. '+08:00'.",
    )


class TalorSerpListEnginesInput(BaseModel):
    """Input for listing engines or inspecting one engine."""

    engine: Optional[str] = Field(
        default=None,
        description="Optional engine key to inspect in detail.",
    )
    args: Optional[str] = Field(
        default=None,
        description="Compatibility field for models that emit a generic 'args' string.",
    )


def create_talor_serp_tool(
    wrapper: Optional[TalorSerpAPIWrapper] = None,
    name: str = "talor_serp_search",
    description: Optional[str] = None,
) -> StructuredTool:
    """Create a LangChain StructuredTool from a TalorSerpAPIWrapper instance.

    The tool description is auto-generated from engine schemas, providing
    agents with full awareness of all 33 engines and their parameters.

    Args:
        wrapper: Pre-configured wrapper. If None, creates one from TALOR_API_KEY env var.
        name: Tool name for agent binding.
        description: Override the auto-generated description.

    Returns:
        A StructuredTool ready for agent.bind_tools() or ToolExecutor.
    """
    if wrapper is None:
        import os

        api_key = os.environ.get("TALOR_API_KEY", "")
        wrapper = TalorSerpAPIWrapper(talor_api_key=api_key)

    registry = wrapper._registry()

    if description is None:
        engines_summary = _build_engines_summary(registry)
        description = (
            "Search the web using Talor SERP API.\n\n"
            "Available engines:\n"
            f"{engines_summary}\n\n"
            "Parameters:\n"
            "  query: the search query (required)\n"
            "  engine: engine key (default: google)\n"
            "  params: engine-specific parameters as JSON object\n"
            "\n"
            "Common params across all engines:\n"
            "  gl: country/region code (e.g. 'us', 'cn', 'uk')\n"
            "  hl: interface language (e.g. 'en', 'zh', 'ja')\n"
            "  device: 'desktop' or 'mobile'\n"
            "  location: geographic targeting\n"
            "  no_cache: boolean, force fresh results\n"
        )

    def _run(query: str, engine: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        extra = params or {}
        return wrapper.run(query, engine=engine, **extra)

    async def _arun(query: str, engine: Optional[str] = None, params: Optional[Dict[str, Any]] = None) -> str:
        extra = params or {}
        return await wrapper.arun(query, engine=engine, **extra)

    return StructuredTool(
        name=name,
        description=description,
        func=_run,
        coroutine=_arun,
        args_schema=TalorSerpSearchInput,
    )


def create_talor_serp_history_tool(
    wrapper: Optional[TalorSerpAPIWrapper] = None,
    name: str = "talor_serp_history",
    description: Optional[str] = None,
) -> StructuredTool:
    """Create a tool that queries Talor SERP history records."""
    if wrapper is None:
        import os

        api_key = os.environ.get("TALOR_API_KEY", "")
        wrapper = TalorSerpAPIWrapper(talor_api_key=api_key)

    if description is None:
        description = (
            "Query Talor SERP search history.\n\n"
            "Parameters:\n"
            "  page: page number (default: 1)\n"
            "  page_size: page size (default: 20)\n"
            "  search_query: optional keyword filter\n"
            "  search_engine: optional engine filter such as google or bing\n"
            "  status: all, success, or error\n"
            "  start_time: optional start unix timestamp in seconds\n"
            "  end_time: optional end unix timestamp in seconds\n"
            "  timezone: optional timezone header such as Asia/Shanghai or +08:00\n"
        )

    def _run(
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None,
        search_engine: Optional[str] = None,
        status: str = "all",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        return wrapper.history(
            page=page,
            page_size=page_size,
            search_query=search_query,
            search_engine=search_engine,
            status=status,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
        )

    async def _arun(
        page: int = 1,
        page_size: int = 20,
        search_query: Optional[str] = None,
        search_engine: Optional[str] = None,
        status: str = "all",
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await wrapper.ahistory(
            page=page,
            page_size=page_size,
            search_query=search_query,
            search_engine=search_engine,
            status=status,
            start_time=start_time,
            end_time=end_time,
            timezone=timezone,
        )

    return StructuredTool(
        name=name,
        description=description,
        func=_run,
        coroutine=_arun,
        args_schema=TalorSerpHistoryInput,
    )


def create_talor_serp_list_engines_tool(
    wrapper: Optional[TalorSerpAPIWrapper] = None,
    name: str = "talor_serp_list_engines",
) -> StructuredTool:
    """Create a tool that lists all available engines and their parameters.

    Useful for agents to discover what engines and params are available
    before calling the search tool.
    """
    if wrapper is None:
        import os

        api_key = os.environ.get("TALOR_API_KEY", "")
        wrapper = TalorSerpAPIWrapper(talor_api_key=api_key)

    registry = wrapper._registry()

    def _run(engine: Optional[str] = None, args: Optional[str] = None) -> str:
        engine = engine or args
        if engine:
            schema = registry.engine(engine)
            if schema is None:
                return f"Unknown engine: {engine}. Available: {', '.join(registry.engine_keys)}"
            return f"{schema.to_description()}\n\nParameters:\n{_build_engine_params_description(schema)}"
        lines = [f"Total engines: {len(registry.engine_keys)}"]
        for cat_key, engine_keys in registry.categories().items():
            lines.append(f"\n[{cat_key}]")
            for key in engine_keys:
                schema = registry.engine(key)
                if schema:
                    lines.append(f"  {key}: {schema.name}")
        lines.append(f"\nDefault engine: {registry.default_engine}")
        lines.append("\nUse engine='engine_key' to get detailed parameter info for a specific engine.")
        return "\n".join(lines)

    async def _arun(engine: Optional[str] = None, args: Optional[str] = None) -> str:
        return _run(engine, args)

    return StructuredTool(
        name=name,
        description=(
            "List all available Talor SERP search engines and their parameters. "
            "Call with engine='engine_key' to get detailed param info for a specific engine."
        ),
        func=_run,
        coroutine=_arun,
        args_schema=TalorSerpListEnginesInput,
    )


def create_talor_serp_statistics_tool(
    wrapper: Optional[TalorSerpAPIWrapper] = None,
    name: str = "talor_serp_statistics",
    description: Optional[str] = None,
) -> StructuredTool:
    """Create a tool that queries Talor SERP usage statistics."""
    if wrapper is None:
        import os

        api_key = os.environ.get("TALOR_API_KEY", "")
        wrapper = TalorSerpAPIWrapper(talor_api_key=api_key)

    if description is None:
        description = (
            "Query Talor SERP usage statistics.\n\n"
            "Parameters:\n"
            "  start_date: start date in YYYY-MM-DD (required)\n"
            "  end_date: end date in YYYY-MM-DD (required)\n"
            "  engines: optional comma-separated engine keys such as google,bing\n"
            "  timezone: optional timezone offset such as +08:00\n"
        )

    def _run(
        start_date: str,
        end_date: str,
        engines: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        return wrapper.statistics(
            start_date=start_date,
            end_date=end_date,
            engines=engines,
            timezone=timezone,
        )

    async def _arun(
        start_date: str,
        end_date: str,
        engines: Optional[str] = None,
        timezone: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await wrapper.astatistics(
            start_date=start_date,
            end_date=end_date,
            engines=engines,
            timezone=timezone,
        )

    return StructuredTool(
        name=name,
        description=description,
        func=_run,
        coroutine=_arun,
        args_schema=TalorSerpStatisticsInput,
    )


class TalorSerpTool:
    """Factory for creating Talor SERP LangChain tools.

    Example:
        .. code-block:: python

            from langchain_talor_serp import TalorSerpTool

            # Get both search + list_engines tools
            tools = TalorSerpTool.tools_from_env()

            # Or just the search tool
            tool = TalorSerpTool.from_env()

            # Bind to an agent
            agent = llm.bind_tools(tools)
    """

    @staticmethod
    def from_api_key(
        api_key: str,
        engine: str = "google",
        endpoint: str = "https://serpapi.talordata.net/serp/v1/request",
        **kwargs: Any,
    ) -> StructuredTool:
        """Create a search tool from an explicit API key."""
        wrapper = TalorSerpAPIWrapper(
            talor_api_key=api_key,
            engine=engine,
            endpoint=endpoint,
            **kwargs,
        )
        return create_talor_serp_tool(wrapper)

    @staticmethod
    def from_env(**kwargs: Any) -> StructuredTool:
        """Create a search tool using the TALOR_API_KEY environment variable."""
        return create_talor_serp_tool(**kwargs)

    @staticmethod
    def from_wrapper(wrapper: TalorSerpAPIWrapper, **kwargs: Any) -> StructuredTool:
        """Create a search tool from a pre-configured wrapper."""
        return create_talor_serp_tool(wrapper, **kwargs)

    @staticmethod
    def history_from_env(**kwargs: Any) -> StructuredTool:
        """Create a history tool using the TALOR_API_KEY environment variable."""
        return create_talor_serp_history_tool(**kwargs)

    @staticmethod
    def history_from_wrapper(wrapper: TalorSerpAPIWrapper, **kwargs: Any) -> StructuredTool:
        """Create a history tool from a pre-configured wrapper."""
        return create_talor_serp_history_tool(wrapper, **kwargs)

    @staticmethod
    def statistics_from_env(**kwargs: Any) -> StructuredTool:
        """Create a statistics tool using the TALOR_API_KEY environment variable."""
        return create_talor_serp_statistics_tool(**kwargs)

    @staticmethod
    def statistics_from_wrapper(wrapper: TalorSerpAPIWrapper, **kwargs: Any) -> StructuredTool:
        """Create a statistics tool from a pre-configured wrapper."""
        return create_talor_serp_statistics_tool(wrapper, **kwargs)

    @staticmethod
    def tools_from_env(**kwargs: Any) -> List[StructuredTool]:
        """Create search, list_engines, history, and statistics tools from env."""
        import os

        api_key = os.environ.get("TALOR_API_KEY", "")
        wrapper = TalorSerpAPIWrapper(talor_api_key=api_key, **kwargs)
        return [
            create_talor_serp_tool(wrapper),
            create_talor_serp_list_engines_tool(wrapper),
            create_talor_serp_history_tool(wrapper),
            create_talor_serp_statistics_tool(wrapper),
        ]

    @staticmethod
    def tools_from_api_key(api_key: str, **kwargs: Any) -> List[StructuredTool]:
        """Create search, list_engines, history, and statistics tools from an explicit key."""
        wrapper = TalorSerpAPIWrapper(talor_api_key=api_key, **kwargs)
        return [
            create_talor_serp_tool(wrapper),
            create_talor_serp_list_engines_tool(wrapper),
            create_talor_serp_history_tool(wrapper),
            create_talor_serp_statistics_tool(wrapper),
        ]
