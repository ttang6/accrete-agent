# hooks.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .events import AgentEvent
from .state import RunState
from .types import LLMResponse, Message, ToolCall, ToolResult


@dataclass
class HookContext:
    event: AgentEvent
    state: RunState
    turn: int = 0
    # 可变载荷：hook 可改
    messages: list[Message] | None = None
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    llm_response: LLMResponse | None = None
    error: BaseException | None = None
    trajectory: Any | None = None
    data: dict[str, Any] = field(default_factory=dict)
    abort: bool = False
    abort_reason: str | None = None


class Hook(Protocol):
    def __call__(self, ctx: HookContext) -> None: ...


class HookManager:
    def __init__(self) -> None:
        self._hooks: dict[AgentEvent, list[Hook]] = {e: [] for e in AgentEvent}

    def on(self, event: AgentEvent, hook: Hook, *, prepend: bool = False) -> None:
        bucket = self._hooks[event]
        if prepend:
            bucket.insert(0, hook)
        else:
            bucket.append(hook)

    def extend(self, other: HookManager) -> None:
        """把另一个 HookManager 的全部 hook 并入本管理器，后注册者后执行。"""
        for event, hooks in other._hooks.items():
            for hook in hooks:
                self._hooks[event].append(hook)

    def emit(self, ctx: HookContext) -> HookContext:
        for hook in self._hooks[ctx.event]:
            hook(ctx)
            if ctx.abort:
                break
        return ctx
