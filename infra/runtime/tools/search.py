"""web_search / web_fetch 工具：基于 Tavily 的网络搜索与网页抓取。"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from tavily import TavilyClient

from infra.core.tools import Tool
from infra.core.types import ToolResult

load_dotenv()

DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_CONTENT_LENGTH = 8000

# 复用的 Tavily 客户端；测试时可直接替换成假客户端。
_CLIENT: TavilyClient | None = None


def _get_client() -> TavilyClient:
    """返回复用的 Tavily 客户端；未配置 API key 时抛 ValueError。"""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("未设置 TAVILY_API_KEY 环境变量，无法使用网络搜索")
        _CLIENT = TavilyClient(api_key=api_key)
    return _CLIENT


def _error(message: str, error_type: str) -> ToolResult:
    return ToolResult(tool_call_id="", content=message, is_error=True, error_type=error_type)


class WebSearchTool(Tool):
    """搜索互联网，返回网页标题、URL 和摘要。"""

    name = "web_search"
    permission_group = "network"
    description = "搜索互联网，返回网页标题、URL 和摘要；需要网页完整正文时再用 web_fetch 抓取。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词或自然语言问题"},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 10,
                "default": DEFAULT_MAX_RESULTS,
            },
            "search_depth": {
                "type": "string",
                "enum": ["basic", "advanced"],
                "default": "basic",
                "description": "basic 更快，advanced 结果更深入",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            return _error("参数 query 必须是非空字符串", "schema_error")
        max_results = arguments.get("max_results", DEFAULT_MAX_RESULTS)
        if not isinstance(max_results, int) or isinstance(max_results, bool) or not 1 <= max_results <= 10:
            return _error("max_results 必须是 1-10 的整数", "schema_error")
        search_depth = arguments.get("search_depth", "basic")
        if search_depth not in ("basic", "advanced"):
            return _error("search_depth 只能是 basic 或 advanced", "schema_error")

        try:
            response = _get_client().search(
                query=query,
                max_results=max_results,
                search_depth=search_depth,
                include_answer=False,  # 让模型自己总结，不依赖 Tavily 的 answer
            )
        except ValueError as error:  # 未配置 API key
            return _error(str(error), "exec_error")
        except Exception as error:
            return _error(f"搜索失败: {error}", "exec_error")

        results = response.get("results", [])
        if not results:
            return ToolResult(tool_call_id="", content="没有找到相关结果。")
        lines = [f"共 {len(results)} 条结果，需要完整正文请用 web_fetch 抓取："]
        for item in results:
            lines.append(
                f"- {item.get('title', '')}\n  {item.get('url', '')}\n  {item.get('content', '')}"
            )
        return ToolResult(tool_call_id="", content="\n\n".join(lines))


class WebFetchTool(Tool):
    """抓取指定网页的详细正文（清洗后的文本）。"""

    name = "web_fetch"
    permission_group = "network"
    description = "抓取 1-5 个网页的详细正文，用于阅读完整文章或文档；URL 来自 web_search 结果。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "urls": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
                "description": "要抓取的网页 URL 列表",
            },
            "max_content_length": {
                "type": "integer",
                "minimum": 500,
                "maximum": 20000,
                "default": DEFAULT_MAX_CONTENT_LENGTH,
                "description": "每个页面最多返回的字符数，防止内容过长",
            },
        },
        "required": ["urls"],
        "additionalProperties": False,
    }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        urls = arguments.get("urls")
        if not isinstance(urls, list) or not urls or not all(isinstance(u, str) and u.strip() for u in urls):
            return _error("参数 urls 必须是非空 URL 字符串列表", "schema_error")
        max_content_length = arguments.get("max_content_length", DEFAULT_MAX_CONTENT_LENGTH)
        if not isinstance(max_content_length, int) or isinstance(max_content_length, bool) or not 500 <= max_content_length <= 20000:
            return _error("max_content_length 必须是 500-20000 的整数", "schema_error")

        try:
            response = _get_client().extract(urls=urls)
        except ValueError as error:  # 未配置 API key
            return _error(str(error), "exec_error")
        except Exception as error:
            return _error(f"抓取失败: {error}", "exec_error")

        parts = []
        for item in response.get("results", []):
            content = item.get("raw_content") or item.get("content") or ""
            if len(content) > max_content_length:
                content = content[:max_content_length] + "\n\n...[内容过长已截断]"
            parts.append(f"# {item.get('title', '')}\n{item.get('url', '')}\n{content}")
        for failed in response.get("failed_results", []):
            parts.append(f"# 抓取失败\n{failed.get('url', '')}\n{failed.get('error', '未知错误')}")
        if not parts:
            return ToolResult(tool_call_id="", content="没有抓到任何内容。")
        return ToolResult(tool_call_id="", content="\n\n".join(parts))
