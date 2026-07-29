"""Core 组件共享的数据类型与枚举。"""

from dataclasses import dataclass, field


ERROR_CLASSES = {
    "schema_error", "exec_error", "timeout", "permission", "not_found", "unsupported",
}

FINISH_REASONS = {
    "completed", "max_turns", "token_budget_exceeded", "time_exceeded",
    "repeated_format_error", "repeated_tool_failure", "run_error", "stop_condition", "killed",
}


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class Message:
    role: str
    content: str = ""
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


@dataclass
class Usage:
    input: int = 0
    output: int = 0
    complete: bool = True


@dataclass
class LLMResponse:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    stop_reason: str
    model: str

    def has_no_tool_calls(self) -> bool:
        """判断响应是否只包含文本回答。"""
        return not self.tool_calls

    def has_tool_calls(self) -> bool:
        """判断响应是否请求执行工具。"""
        return bool(self.tool_calls)


@dataclass
class ToolResult:
    success: bool
    output: str
    error: str | None = None
    error_class: str | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.error_class is not None and self.error_class not in ERROR_CLASSES:
            raise ValueError(f"未知 error_class: {self.error_class}")


@dataclass
class RunLimits:
    max_turns: int
    max_total_tokens: int
    wall_time_limit_s: float
    max_consecutive_format_errors: int
    max_repeated_tool_failures: int


@dataclass
class RunResult:
    final_text: str
    finish_reason: str
    turns: int
    usage: Usage
    error: str | None = None

    def __post_init__(self) -> None:
        if self.finish_reason not in FINISH_REASONS:
            raise ValueError(f"未知 finish_reason: {self.finish_reason}")
