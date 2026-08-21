from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field

from .types import FinishReason, Message, RunLimits, ToolCall, ToolResult, Usage


def estimate_tokens(messages: list[Message]) -> int:
    """近似估算一次请求放入上下文的总 token 数。

    对每条消息的 content（及工具调用的参数 JSON）按 ``len(text) // 4``（约 4 字符≈1
    token）累加。各模型的字符/token 换算比不完全一致，这里只用于护栏预警，不用于计费。
    """
    total = 0
    for message in messages:
        if (message.content or "").strip():
            total += len(message.content or "") // 4
        for call in message.tool_calls or []:
            total += len(call.function.arguments or "") // 4
    return total


@dataclass
class RunState:
    """跨 turn 状态；可被 hook 读取/少量更新。不把止损计数暴露给模型。"""
    run_id: str
    task: str
    workdir: str
    started_at: float
    messages: list = field(default_factory=list)  # list[Message]
    turn: int = 0
    usage: Usage = field(default_factory=Usage)
    # 累计成本兑底；只记 CNY，USD 不计入（与 max_cost_cny 的约定一致）。
    total_cost_cny: float = 0.0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    # 任意扩展：lesson ids、experiment tags ...
    extras: dict = field(default_factory=dict)


class RunBudget:
    def __init__(self, limits: RunLimits, started_at: float) -> None:
        self.limits = limits
        self.started_at = started_at
        self.consecutive_schema_errors = 0
        self.repeated_tool_failures = 0
        self._last_sig: tuple[str, str] | None = None

    def check_turn(self, turn: int) -> FinishReason | None:
        if turn >= self.limits.max_turns:
            return "max_turns"
        if time.time() - self.started_at > self.limits.wall_time_s:
            return "time_exceeded"
        return None

    def check_cost(self, total_cost_cny: float) -> FinishReason | None:
        """累计 CNY 成本撞到上限即判超额；上限为 None 时不判定。"""
        limit = self.limits.max_cost_cny
        if limit is not None and total_cost_cny >= limit:
            return "cost_exceeded"
        return None

    def record_format_error(self) -> FinishReason | None:
        """记一次"模型没发工具调用"的协议违规，并当即判定是否放弃。

        与 record_tool 里的 schema_error 共用同一个连续计数：两者都是模型不按协议
        输出，一次正常的工具回合会把计数清零。必须在这里立刻判定，因为这一轮没有
        工具执行，走不到 record_tool。
        """
        self.consecutive_schema_errors += 1
        if self.consecutive_schema_errors >= self.limits.max_consecutive_schema_errors:
            return "repeated_format_error"
        return None

    def record_tool(self, call: ToolCall, result: ToolResult) -> FinishReason | None:
        sig = (call.function.name, _args_hash(call.function.arguments))
        if result.is_error and result.error_type == "schema_error":
            self.consecutive_schema_errors += 1
        else:
            self.consecutive_schema_errors = 0

        if result.is_error:
            if sig == self._last_sig:
                self.repeated_tool_failures += 1
            else:
                self.repeated_tool_failures = 1
        else:
            self.repeated_tool_failures = 0
        self._last_sig = sig

        if self.consecutive_schema_errors >= self.limits.max_consecutive_schema_errors:
            return "repeated_format_error"
        if self.repeated_tool_failures >= self.limits.max_repeated_tool_failures:
            return "repeated_tool_failure"
        return None


def _args_hash(arguments_json: str) -> str:
    try:
        obj = json.loads(arguments_json)
        payload = json.dumps(obj, sort_keys=True, ensure_ascii=False)
    except Exception:
        payload = arguments_json
    return hashlib.sha256(payload.encode()).hexdigest()