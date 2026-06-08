"""LangChain integration for Talor SERP API.

Provides a drop-in replacement for GoogleSerperAPIWrapper with support
for 33 search engines across Google, Bing, Yandex, and DuckDuckGo.

Quick start:
    >>> import os
    >>> os.environ["TALOR_API_KEY"] = "your-token"
    >>> from langchain_talordata import TalorSerpAPIWrapper, TalorSerpTool
    >>> wrapper = TalorSerpAPIWrapper()
    >>> wrapper.run("LangChain tutorial")
    >>> tool = TalorSerpTool.from_env()
"""

from .engines import (
    DEFAULT_ENGINE,
    ENGINE_CATEGORIES,
    SUPPORTED_ENGINES,
)
from .schema import EngineRegistry, EngineSchema
from .tool import (
    TalorSerpTool,
    create_talor_serp_history_tool,
    create_talor_serp_list_engines_tool,
    create_talor_serp_statistics_tool,
    create_talor_serp_tool,
)
from .wrapper import TalorSerpAPIWrapper

__all__ = [
    "TalorSerpAPIWrapper",
    "TalorSerpTool",
    "EngineRegistry",
    "EngineSchema",
    "create_talor_serp_tool",
    "create_talor_serp_list_engines_tool",
    "create_talor_serp_history_tool",
    "create_talor_serp_statistics_tool",
    "DEFAULT_ENGINE",
    "SUPPORTED_ENGINES",
    "ENGINE_CATEGORIES",
]

__version__ = "0.1.2"
