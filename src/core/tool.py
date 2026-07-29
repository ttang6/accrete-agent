"""工具模板方法与结构化错误转换。"""

from abc import ABC, abstractmethod
from enum import Flag, auto
import time

from .state import elapsed_ms
from .types import ToolResult


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
