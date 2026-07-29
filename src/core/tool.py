"""工具模板方法与结构化错误转换。"""

from abc import ABC, abstractmethod
from enum import Flag, auto
import time

from .ports import OutputStore
from .state import elapsed_ms
from .types import ToolResult


def clip_output(text: str, max_chars: int, store: OutputStore | None = None) -> str:
    """把过长的工具输出截断到 max_chars，并尽量把完整内容交给 store 落盘。

    保留首尾两段：错误上下文通常在开头，结论通常在结尾。没有 store 时只说明省略了
    多少，不影响工具可用——落盘就位后，截断处会附上完整内容的位置。
    """
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    location = store.put(text) if store else None
    note = f"[已截断：原文 {len(text)} 字符，省略中间 {len(text) - max_chars} 字符"
    note += f"，完整内容见 {location}]" if location else "]"
    return f"{text[:head]}\n\n{note}\n\n{text[-(max_chars - head):]}"


class AccessType(Flag):
    """工具可能产生的外部访问类型，可按位组合。"""

    NONE = 0
    READ = auto()
    WRITE = auto()
    NETWORK = auto()


class ToolExecutionError(Exception):
    """允许工具声明确定错误类别的异常。"""

    def __init__(self, message: str, error_class: str = "exec_error") -> None:
        super().__init__(message)
        self.error_class = error_class


class BaseTool(ABC):
    """所有同步工具的统一执行入口。"""

    name: str
    description: str
    parameters: dict
    strict_compatible: bool = True
    access_type: AccessType = AccessType.NONE

    def execute(self, args: dict) -> ToolResult:
        """执行工具，将任何异常转换为带耗时的 ToolResult。"""
        started = time.perf_counter()
        try:
            raw = self._run(args)
            result = raw if isinstance(raw, ToolResult) else ToolResult(True, str(raw))
        except ToolExecutionError as exc:
            result = ToolResult(False, "", error=str(exc), error_class=exc.error_class)
        except TimeoutError as exc:
            result = ToolResult(False, "", error=str(exc), error_class="timeout")
        except Exception as exc:
            result = ToolResult(False, "", error=str(exc), error_class="exec_error")
        result.meta.setdefault("duration_ms", elapsed_ms(started))
        return result

    @abstractmethod
    def _run(self, args: dict) -> str | ToolResult:
        """执行具体工具逻辑。"""
