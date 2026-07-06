"""ArxivTool — 把 arxiv-mcp-server 桥接进 accrete 的 function calling 层。

LLM 视角：一个名叫 `arxiv` 的 BaseTool，通过 `action` 字段分派到具体操作。
内部实现：每次 `_execute` 启动一次 MCP server subprocess（`uv tool run
arxiv-mcp-server`），完成 `call_tool` 后自动关闭。subprocess 冷启动开销
约 1-2 秒；若未来调用频繁，可升级为常驻连接 + 后台 asyncio thread。

为什么单一 BaseTool + action 分派而非 11 个独立 tool：
  - arxiv-mcp-server 暴露 11 个 tool，独立注册让 LLM schema 膨胀 + 注意力分散
  - 核心工作流只有 4 步（search → get_abstract → download → read），集中入口语义清晰
  - 对齐 skill_exec / load_skill：单入口 + 分派是 accrete 处理"大工具组"的统一范式
  - MCP 作为协议证明，完成桥接演示即达成目标；后续需要再拆
"""

import asyncio
from pathlib import Path
from typing import Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from accrete.core.logger import get_logger
from accrete.tool.base import BaseTool


_logger = get_logger("tool.arxiv")


_DEFAULT_STORAGE = Path("data/arxiv_papers")
_DEFAULT_TIMEOUT = 60
_MAX_OUTPUT_CHARS = 8000

# 必须和 arxiv-mcp-server 暴露的 tool name 完全对齐。
_ACTIONS = (
    "search_papers",
    "get_abstract",
    "download_paper",
    "read_paper",
    "list_papers",
    "citation_graph",
    "watch_topic",
    "check_alerts",
)


class ArxivTool(BaseTool):
    """通过 MCP 协议调用 arxiv-mcp-server。单一入口，按 action 分派。"""

    def __init__(
        self,
        storage_path: Path = _DEFAULT_STORAGE,
        timeout_seconds: int = _DEFAULT_TIMEOUT,
        max_output_chars: int = _MAX_OUTPUT_CHARS,
    ):
        self._storage = Path(storage_path).resolve()
        self._timeout = timeout_seconds
        self._max_output = max_output_chars

    @property
    def name(self) -> str:
        return "arxiv"

    def call_key(self, kwargs: dict) -> str:
        """arxiv 的 op 投影 = action(+query)（工具自声明）。

        修掉"8 个 action 全挤进一个 arxiv 桶"的塌缩：每个 action 独立计数；
        search_papers 这类带 query 的再按 query 细分（同一 query 反复失败才累加）。
        """
        d = kwargs or {}
        action = str(d.get("action", "") or "").strip()
        query = str(d.get("query", "") or "").strip()
        if action and query:
            return f"arxiv:{action}:{query}"
        return f"arxiv:{action}" if action else "arxiv"

    @property
    def description(self) -> str:
        return (
            "通过 arxiv-mcp-server（MCP 协议）访问 arXiv 论文库。"
            "按 action 分派到具体操作，典型工作流："
            "search_papers → get_abstract → download_paper → read_paper。\n"
            "可用 actions:\n"
            "- search_papers: 搜 arXiv 候选论文。传 query；可选 categories / date_from / date_to / max_results / sort_by。\n"
            "- get_abstract: 拿单篇 metadata + abstract 不下载全文（省 tokens）。传 paper_id。\n"
            "- download_paper: 下载全文到本地（HTML 优先，PDF fallback）。传 paper_id。\n"
            "- read_paper: 读已下载论文的 markdown 全文。传 paper_id。\n"
            "- list_papers: 列本地已下载论文 ID。\n"
            "- citation_graph: 拿论文的引用 / 被引用图。传 paper_id。\n"
            "- watch_topic: 注册持续监控某 topic 的新论文。传 topic 字符串。\n"
            "- check_alerts: 检查 watch_topic 注册的 topic 是否有新论文（topic 参数可选）。\n"
            "query 语法：支持 `ti:` / `au:` / `abs:` 字段限定、AND/OR/ANDNOT 布尔、引号短语。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(_ACTIONS),
                    "description": "要执行的 arxiv 操作名",
                },
                "query": {
                    "type": "string",
                    "description": (
                        "search_papers 的搜索表达式。"
                        "支持 ti:/au:/abs: 字段、AND/OR/ANDNOT 布尔、引号短语。"
                    ),
                },
                "paper_id": {
                    "type": "string",
                    "description": "arxiv ID（get_abstract / download_paper / read_paper / citation_graph 用）",
                },
                "max_results": {
                    "type": "integer",
                    "description": "search_papers 返回数量上限",
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "search_papers 的 arxiv 分类列表，如 ['cs.LG', 'cs.AI']",
                },
                "date_from": {
                    "type": "string",
                    "description": "search_papers 起始日期（YYYY-MM-DD）",
                },
                "date_to": {
                    "type": "string",
                    "description": "search_papers 截止日期（YYYY-MM-DD）",
                },
                "sort_by": {
                    "type": "string",
                    "enum": ["relevance", "date"],
                    "description": "search_papers 排序方式，默认 relevance",
                },
                "topic": {
                    "type": "string",
                    "description": "watch_topic / check_alerts 的 topic 字符串（语法同 query）",
                },
            },
            "required": ["action"],
        }

    def validate(self, **kwargs) -> Optional[str]:
        action = (kwargs.get("action") or "").strip()
        if not action:
            return "action 参数不能为空"
        if action not in _ACTIONS:
            return f"未知 action '{action}'，可选：{', '.join(_ACTIONS)}"
        return None

    def _execute(self, **kwargs) -> str:
        action = kwargs.pop("action").strip()
        mcp_args = {k: v for k, v in kwargs.items() if v not in (None, "", [])}

        _logger.info(f"[arxiv] action={action} keys={list(mcp_args.keys())}")
        try:
            return asyncio.run(self._call_mcp(action, mcp_args))
        except Exception as e:
            return f"[arxiv 错误] {type(e).__name__}: {e}"

    async def _call_mcp(self, action: str, args: dict) -> str:
        self._storage.mkdir(parents=True, exist_ok=True)
        params = StdioServerParameters(
            command="uv",
            args=[
                "tool", "run", "arxiv-mcp-server",
                "--storage-path", str(self._storage),
            ],
            env=None,
        )
        async with stdio_client(params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await asyncio.wait_for(session.initialize(), timeout=self._timeout)
                result = await asyncio.wait_for(
                    session.call_tool(action, args),
                    timeout=self._timeout,
                )
                return self._format_result(result)

    def _format_result(self, result) -> str:
        parts: list[str] = []
        for content in getattr(result, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                parts.append(text)
        output = "\n".join(parts).strip() or "(空结果)"
        if len(output) > self._max_output:
            output = (
                output[: self._max_output]
                + f"\n\n...(stdout {len(output)} 字，已截断)"
            )
        return output
