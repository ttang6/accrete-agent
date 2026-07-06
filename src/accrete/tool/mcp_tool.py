"""MCPTool —— 把一个 MCP server 的工具 auto-expand 成 accrete 的独立 BaseTool。

复用 HelloAgents 第 10 章的理念：一个 MCP server 加进来，自动把它 list_tools 发现的
每个工具，展开成 agent 能直接调用的独立工具（一个 server → N 个工具）。底层用
fastmcp.Client（吃标准 MCPConfig）。

异步桥同步（关键摩擦点）：fastmcp.Client 是纯异步，而 accrete 的 BaseTool._execute
是同步、返回字符串。这里取最简实现——**每次调用临时连一次 server**（reconnect-per-call：
每次 spawn 一次 server 子进程，简单但偏慢；要更快可改成持久连接 + 后台事件循环）。
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

from fastmcp import Client

from accrete.tool.base import BaseTool


def _run_sync(coro) -> Any:
    """把协程跑到结果。若当前线程已在事件循环里（如并行 tool 路径），丢到新线程跑，
    避开 'asyncio.run() cannot be called from a running event loop'。"""
    try:
        return asyncio.run(coro)
    except RuntimeError:
        box: dict[str, Any] = {}
        thread = threading.Thread(target=lambda: box.__setitem__("result", asyncio.run(coro)))
        thread.start()
        thread.join()
        return box["result"]


def _result_text(result: Any) -> str:
    """从 fastmcp call_tool 结果里抠出文本（content[].text，退而求其次取 .data）。"""
    parts = [
        content.text
        for content in (getattr(result, "content", None) or [])
        if getattr(content, "text", None) is not None
    ]
    if parts:
        return "\n".join(parts)
    data = getattr(result, "data", None)
    return str(data) if data is not None else str(result)


class MCPTool(BaseTool):
    """一个 MCP server 工具的代理：agent 调它 → 转成 fastmcp call_tool。"""

    def __init__(
        self,
        server_name: str,
        server_config: dict,
        tool_name: str,
        description: str,
        schema: dict | None,
    ) -> None:
        self._server_name = server_name
        self._client_config = {"mcpServers": {server_name: server_config}}
        self._tool_name = tool_name
        self._description = description
        self._schema = schema or {"type": "object", "properties": {}}

    @property
    def name(self) -> str:
        return f"mcp_{self._server_name}_{self._tool_name}"

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._schema

    def _execute(self, **kwargs) -> str:
        async def _call() -> Any:
            async with Client(self._client_config) as client:
                return await client.call_tool(self._tool_name, kwargs)

        return _result_text(_run_sync(_call()))


def register_mcp_server(registry, server_name: str, server_config: dict) -> int:
    """主动发现 + auto-expand：连 server、list_tools，每个工具注册成一个 MCPTool。

    registry：accrete 的 ToolRegistry。
    server_config：标准 MCPConfig 的单 server 配置（{command, args, env?}）。
    返回注册的工具数。
    """
    config = {"mcpServers": {server_name: server_config}}

    async def _discover() -> Any:
        async with Client(config) as client:
            return await client.list_tools()

    count = 0
    for tool in _run_sync(_discover()):
        registry.register(
            MCPTool(
                server_name=server_name,
                server_config=server_config,
                tool_name=tool.name,
                description=tool.description or "",
                schema=getattr(tool, "inputSchema", None),
            )
        )
        count += 1
    return count
