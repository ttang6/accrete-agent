from abc import ABC, abstractmethod
from typing import Optional
from accrete.tool.base import BaseTool
from accrete.tool.registry import ToolRegistry
from accrete.core.message import Message
from accrete.core.logger import get_logger, RunTracer

class BaseAgent(ABC):
    def __init__(self, name, llm, tool_registry=None, system_prompt: str = ""):
        self.name = name
        self.llm = llm
        self.tool_registry = tool_registry or ToolRegistry()
        self.system_prompt = system_prompt
        self.messages: list[Message] = []
        self._logger = get_logger(name)
        self._tracer: Optional[RunTracer] = None

    def add_tool(self, tool: BaseTool):
        """注册工具"""
        self.tool_registry.register(tool)

    def _get_tools_description(self) -> str:
        """获取所有工具的 prompt 描述"""
        return self.tool_registry.get_tools_description()

    def _call_llm(self, messages: list[Message], **kwargs) -> str:
        """
        调用 LLM
        子类不需要直接操作 self.llm，通过这个方法调用。
        per-call token 日志已下沉到 LLMClient._track_usage()，
        不管谁调 think() / think_with_tools() 都自动记录。
        """
        api_messages = [m.to_dict() for m in messages]
        temperature = kwargs.get("temperature", 0.0)
        return self.llm.think(messages=api_messages, temperature=temperature)

    def _log(self, message: str, level: str = "info", **extra):
        """统一日志接口。

        Args:
            message: 日志内容。
            level: 日志级别（debug / info / warning / error）。
            **extra: 附加字段，会写入 JSON 日志和轨迹。
        """
        log_func = getattr(self._logger, level, self._logger.info)
        log_func(message, extra=extra)

        # 同时记录到 tracer（如果正在 run）
        if self._tracer and extra:
            self._tracer.step(message=message, **extra)

    def _trace(self, **fields):
        """记录一个轨迹步骤（仅当 tracer 存在时）。"""
        if self._tracer:
            self._tracer.step(**fields)

    def reset(self):
        """重置消息历史"""
        self.messages = []

    @abstractmethod
    def run(self, user_input: str, **kwargs) -> str:
        """执行 Agent 的主逻辑"""
        pass

    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}, tools={len(self.tool_registry)}>"
