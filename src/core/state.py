"""单次运行的状态、护栏与计时。

这里分开放两类运行期状态：AgentState 最终会呈现给模型，RunGuards 只服务于终止
判断，模型永远读不到。两者都是 per-run，进程结束即弃。
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import time

from .types import RunLimits, ToolCall, ToolResult, Usage


def elapsed_ms(started: float) -> float:
    """从 time.perf_counter() 的起点算出毫秒耗时。"""
    return round((time.perf_counter() - started) * 1000, 3)


@dataclass
class AgentState:
    """单次运行中跨组件共享、且最终会呈现给模型的状态。

    这里只放模型可见或工具需要的东西。止损计数只有终止逻辑会读，模型永远看不到，
    所以它们在 RunGuards 上，不在这里。
    """

    workdir: str
    started_at: float
    tool_call_counts: dict[str, int] = field(default_factory=dict)

    def render_system_hint(self) -> str:
        """渲染本轮临时 system hint。"""
        now = datetime.now(timezone.utc).isoformat()
        lines = [f"Current UTC time: {now}"]
        for name, count in sorted(self.tool_call_counts.items()):
            if count >= 3:
                lines.append(f"Reminder: tool '{name}' has been called {count} times.")
        return "\n".join(lines)


class RunStopped(Exception):
    """运行应当终止。

    在触发点直接抛出，避免把终止原因逐层作为返回值往上传。
    detail 里的字段会原样写入轨迹，用于事后归因。
    """

    def __init__(self, reason: str, **detail) -> None:
        super().__init__(reason)
        self.reason = reason
        self.detail = detail


class RunGuards:
    """一次运行的记账与止损。所有终止判断都在这里，触发即抛 RunStopped。

    两类阈值的地位不同，改动时不要混为一谈：max_turns、max_total_tokens 和
    wall_time_limit_s 是测量预算的硬上限，放宽它们等于用更多资源换分数；
    max_consecutive_format_errors 与 max_repeated_tool_failures 只决定何时放弃，
    放宽也突破不了前者。
    """

    def __init__(self, limits: RunLimits, started_at: float, max_output_tokens: int) -> None:
        self.limits = limits
        self.started_at = started_at
        self.max_output_tokens = max_output_tokens
        self.turns = 0
        self.usage = Usage()
        self.last_tool_signature: tuple[str, str] | None = None
        self.repeated_tool_failures = 0
        self.consecutive_format_errors = 0

    def add_usage(self, usage: Usage) -> None:
        """把一次模型调用的用量计入总账；任一侧不完整则总账视为不完整。"""
        self.usage = Usage(self.usage.input + usage.input, self.usage.output + usage.output,
                           self.usage.complete and usage.complete)

    def record_call(self, usage: Usage) -> None:
        """记一次主循环回合：计入用量并推进轮次。"""
        self.add_usage(usage)
        self.turns += 1

    def before_turn(self) -> None:
        """回合开始前检查轮数与墙钟。

        Raises:
            RunStopped: max_turns 或 time_exceeded。
        """
        if self.turns >= self.limits.max_turns:
            raise RunStopped("max_turns", turns=self.turns)
        elapsed = time.time() - self.started_at
        if elapsed > self.limits.wall_time_limit_s:
            raise RunStopped("time_exceeded", elapsed_s=round(elapsed, 3))

    def before_call(self, prompt_tokens: int, max_input_tokens_per_call: int) -> None:
        """模型调用前预检 token：已耗 + 本轮输入 + 预留输出是否超总预算，或单轮输入超窗。

        Raises:
            RunStopped: token_budget_exceeded。
        """
        projected = self.usage.input + self.usage.output + prompt_tokens + self.max_output_tokens
        if projected > self.limits.max_total_tokens or prompt_tokens > max_input_tokens_per_call:
            raise RunStopped("token_budget_exceeded", prompt_tokens=prompt_tokens,
                             projected_tokens=projected)

    def record_tool(self, call: ToolCall, result: ToolResult) -> None:
        """更新两个止损计数：同参重复失败与连续格式错误。这两个计数模型看不到。"""
        signature = (call.name, _args_hash(call.arguments))
        if result.success:
            self.repeated_tool_failures = 0
        elif signature == self.last_tool_signature:
            self.repeated_tool_failures += 1
        else:
            self.repeated_tool_failures = 1
        self.last_tool_signature = signature
        if result.error_class == "schema_error":
            self.consecutive_format_errors += 1
        else:
            self.consecutive_format_errors = 0

    def after_tools(self) -> None:
        """本轮工具跑完后检查止损闸。

        detail 带上工具名、参数指纹和计数：轨迹归因要靠它区分「卡在同一个调用上」
        和「换着参数一直错」，这两种失败导出的 harness 改动完全不同。

        Raises:
            RunStopped: repeated_format_error 或 repeated_tool_failure。
        """
        tool, args_hash = self.last_tool_signature or ("", "")
        if self.consecutive_format_errors >= self.limits.max_consecutive_format_errors:
            raise RunStopped("repeated_format_error", tool=tool,
                             count=self.consecutive_format_errors)
        if self.repeated_tool_failures >= self.limits.max_repeated_tool_failures:
            raise RunStopped("repeated_tool_failure", tool=tool, args_hash=args_hash,
                             count=self.repeated_tool_failures)


def _args_hash(arguments: dict) -> str:
    """计算参数指纹。排序键名，使字面顺序不同、内容相同的调用得到同一指纹。"""
    return hashlib.sha256(
        json.dumps(arguments, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
