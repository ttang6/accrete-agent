import asyncio
import json
from typing import Callable, Optional

from nanoagent.tool.base import BaseTool, FunctionTool
from nanoagent.core.logger import get_logger

_logger = get_logger("tool.registry")


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册一个工具"""
        if tool.name in self._tools:
            _logger.warning(f"工具 '{tool.name}' 已存在，将被覆盖")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        """根据名称获取工具"""
        return self._tools.get(name)

    def execute(self, name: str, **kwargs) -> str:
        """执行指定工具"""
        tool = self.get(name)
        if tool is None:
            return f"错误: 未知工具 '{name}'"
        return tool.run(**kwargs)

    def get_tools_description(self) -> str:
        """生成供 LLM 阅读的工具描述文本"""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())

    def get_all_tools(self) -> list[BaseTool]:
        return list(self._tools.values())

    # ---------- 装饰器注册（从 decorator_registry 整合） ----------

    def tool(self, name: str, description: str) -> Callable:
        """装饰器：将函数注册为工具。

        用法：
            registry = ToolRegistry()

            @registry.tool("greet", "打招呼")
            def greet(args: str) -> str:
                return f"Hello, {args}!"

            # 等价于手动 register(FunctionTool("greet", "打招呼", greet))
        """
        def decorator(func: Callable) -> Callable:
            self.register(FunctionTool(name, description, func))
            return func
        return decorator

    # ---------- 并发执行（从 async_tool_executor 整合） ----------

    async def _execute_one(self, name: str, **kwargs) -> dict:
        """异步执行单个工具，捕获异常不影响其他工具。"""
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(
                None, lambda: self.execute(name, **kwargs)
            )
            return {"tool": name, "status": "success", "result": result}
        except Exception as e:
            return {"tool": name, "status": "error", "result": f"[执行错误] {name}: {e}"}

    async def _execute_parallel_async(self, calls: list[dict]) -> list[dict]:
        """并发执行多个工具调用。

        Args:
            calls: [{"name": "search", "kwargs": {"args": "北京天气"}}, ...]

        Returns:
            [{"tool": "search", "status": "success", "result": "..."}, ...]
        """
        tasks = [
            self._execute_one(call["name"], **call.get("kwargs", {}))
            for call in calls
        ]
        return await asyncio.gather(*tasks)

    def execute_parallel(self, calls: list[dict]) -> list[dict]:
        """并发执行多个工具调用（同步入口，内部用 asyncio）。

        用法：
            results = registry.execute_parallel([
                {"name": "search", "kwargs": {"args": "北京天气"}},
                {"name": "get_current_time", "kwargs": {}},
            ])
        """
        try:
            loop = asyncio.get_running_loop()
            # 已经在 async 环境中，不能用 asyncio.run
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self._execute_parallel_async(calls))
                return future.result()
        except RuntimeError:
            # 没有运行中的 event loop，直接 asyncio.run
            return asyncio.run(self._execute_parallel_async(calls))
