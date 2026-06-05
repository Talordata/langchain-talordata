# langchain-talor-serp

**LangChain integration for TalorData's SERP APIs**

[![PyPI version](https://img.shields.io/pypi/v/langchain-talor-serp?color=blue)](https://pypi.org/project/langchain-talor-serp/)
[![Python versions](https://img.shields.io/pypi/pyversions/langchain-talor-serp)](https://pypi.org/project/langchain-talor-serp/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

[Installation](#installation) •
[Quick Start](#quick-start) •
[Tools](#tools) •
[Resources](#resources)

LangChain integration for the Talor SERP API.

This package provides:

- `TalorSerpAPIWrapper` for direct sync and async API access
- `TalorSerpTool` for creating LangChain tools
- bundled engine schemas for 30+ search engines
- support for search, history, and statistics endpoints

## Overview

`langchain-talor-serp` provides LangChain tools for [TalorData](https://talordata.com)'s SERP APIs, enabling your AI agents to:

- **Search** - Query search engines with geo-targeting and language customization
- **Inspect engines** - Discover supported engines and engine-specific parameters
- **Query history** - Fetch SERP request history with filters
- **View statistics** - Retrieve usage statistics by date range and engine

## Installation

```bash
pip install langchain-talor-serp
```

## Quick start
### 1. Get your API key

Sign up at [TalorData](https://talordata.com) and get your API key from the dashboard.

### 2. Set up authentication
```python
import os
os.environ["TALOR_API_KEY"] = "your-token"
```

### 3. Modern agent usage
```python
from langchain_talor_serp import TalorSerpTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tool = TalorSerpTool.from_env()

# Tool calling without langchain_classic agents
model_with_tools = llm.bind_tools([tool])
response = model_with_tools.invoke("Search for the latest LangChain news")
print(response)
```

### 4. Search tool
```python
from langchain_talor_serp import TalorSerpTool

search_tool = TalorSerpTool.from_env()

result = search_tool.invoke({
    "query": "LangChain tutorial",
    "engine": "google",
    "params": {
        "gl": "us",
        "hl": "en",
        "device": "desktop",
    },
})

print(result)
```

Search parameters:

- `query`: required search query text
- `engine`: optional engine key such as `google`, `google_news`, `google_images`, `bing`, `duckduckgo`
- `params`: optional engine-specific parameter object
- common `params` fields include `gl`, `hl`, `device`, `location`, and `no_cache`
- use `talor_serp_list_engines` to inspect detailed parameters for a specific engine

`params` also accepts a JSON string when returned by a model tool call, for example:

```python
result = search_tool.invoke({
    "query": "LangChain tutorial",
    "engine": "google",
    "params": "{\"hl\": \"zh-CN\", \"gl\": \"cn\"}",
})
```

### 5. History tool
```python
from langchain_talor_serp import TalorSerpTool

history_tool = TalorSerpTool.history_from_env()

result = history_tool.invoke({
    "page": 1,
    "page_size": 20,
    "search_query": "langchain",
    "search_engine": "google",
    "status": "success",
    "timezone": "Asia/Shanghai",
})

print(result)
```

History parameters:

- `page`: page number, default `1`
- `page_size`: page size, default `20`
- `search_query`: optional keyword filter
- `search_engine`: optional engine filter such as `google` or `bing`
- `status`: `all`, `success`, or `error`
- `start_time`: optional unix timestamp in seconds
- `end_time`: optional unix timestamp in seconds
- `timezone`: optional timezone header such as `Asia/Shanghai` or `+08:00`

### 6. Statistics tool
```python
from langchain_talor_serp import TalorSerpTool

statistics_tool = TalorSerpTool.statistics_from_env()

result = statistics_tool.invoke({
    "start_date": "2026-06-01",
    "end_date": "2026-06-05",
    "engines": "google,bing",
    "timezone": "+08:00",
})

print(result)
```

Statistics parameters:

- `start_date`: required, format `YYYY-MM-DD`
- `end_date`: required, format `YYYY-MM-DD`
- `engines`: optional comma-separated engine keys such as `google,bing`
- `timezone`: optional timezone offset such as `+08:00`

### 7. Bind multiple tools
```python
from langchain_talor_serp import TalorSerpTool
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = TalorSerpTool.tools_from_env()

model_with_tools = llm.bind_tools(tools)
response = model_with_tools.invoke("Show my SERP usage statistics for 2026-06-01 to 2026-06-05")
print(response)
```

`bind_tools()` only lets the model generate `tool_calls`. To actually execute
the selected tool, either:

- call `tool.invoke(...)` directly, or
- use an agent workflow that executes tools automatically

## Tools

- `talor_serp_search` - search the web with engine-specific parameters
- `talor_serp_list_engines` - inspect supported engines and detailed parameter schemas
- `talor_serp_history` - query historical SERP requests
- `talor_serp_statistics` - query usage statistics for a date range

### Compatibility note

If you are using modern package versions such as:

- `langchain-core>=1.0`
- `langchain-classic>=1.0`
- `langchain-openai>=1.0`

avoid `langchain_classic.agents.create_openai_functions_agent()` and other
legacy `initialize_agent` / `AgentType.OPENAI_FUNCTIONS` flows with chat models.
Those classic agent paths call `llm.invoke(..., callbacks=...)`, while modern
`BaseChatModel.invoke()` forwards callbacks through `config`, which can lead to:

```text
TypeError: BaseLLM.generate_prompt() got multiple values for keyword argument 'callbacks'
```

Use one of these approaches instead:

- `llm.bind_tools([tool])` for direct tool calling
- `langchain.agents.create_agent(...)` for new agent workflows
- `agent.invoke(...)` instead of deprecated `agent.run(...)`

## Features

- compatible with LangChain tool workflows
- supports multiple Talor SERP engines such as Google, Bing, Yandex, and DuckDuckGo
- exposes engine schema metadata for parameter-aware integrations
- includes helper APIs for usage history and statistics

## Resources

- PyPI: [langchain-talor-serp](https://pypi.org/project/langchain-talor-serp/)
- TalorData: [talordata.com](https://talordata.com)
