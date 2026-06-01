"""StopCondition — 结构化循环终止判断。

问题：`MainLoop.max_iterations=20` 只是 safety net（避免无限烧 token），
不是 agentic 决策。当同一 (tool, args) 连续失败 N 次时，继续 iter 就是
浪费——应该主动终止让用户看到失败原因，而不是等 max_iterations 耗尽。

入门友好视角：
- 这个模块只做"要不要停" 的判断，不直接停——返回 StopDecision 给 MainLoop
- 当前只实现一种判断：`check_repeated_failure`——同参数连续失败超阈值
- 未来可能会加更多 check（coverage 严重不足 / token budget 超 / 等）
- StopReason enum 提前把所有可能原因都命名好，方便后续扩展

停机判断的整体 formula：
```
coverage_ok && no_pending_recovery && evaluator_action == finalize → natural stop
same_failure_repeated >= threshold → forced stop
```
- natural stop 已由 MainLoop "LLM 无 tool_calls" 自然实现
- evaluator finalize 由 Harness 层实现
- 本模块专注补强 **forced stop 路径**
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nanoagent.runtime.failure_memory import FailureMemory


class StopReason(str, Enum):
    """循环终止原因枚举。值同时作为 trace/log 里的字符串标签。"""

    NATURAL_FINALIZE = "natural_finalize"        # LLM 无 tool_calls，主动结束
    MAX_ITERATIONS = "max_iterations"            # iter 数到 safety cap
    REPEATED_FAILURE = "repeated_failure"        # 同参数连续失败超阈值
    EVALUATOR_FINALIZE = "evaluator_finalize"    # 副 LLM 判 coverage_ok + finalize


@dataclass
class StopDecision:
    """`check_*` 函数返回的结构化判断。

    `should_stop=False` 时其他字段无意义（just carry default）。
    `should_stop=True` 时 reason 必填，details 放调试/trace 细节。
    """
    should_stop: bool
    reason: StopReason = StopReason.NATURAL_FINALIZE
    details: dict = field(default_factory=dict)


DEFAULT_REPEAT_FAILURE_THRESHOLD: int = 3


def check_repeated_failure(
    failure_memory: "FailureMemory",
    threshold: int = DEFAULT_REPEAT_FAILURE_THRESHOLD,
) -> StopDecision:
    """检查 FailureMemory 里是否有任一 entry 的 failure_count 超阈值。

    超阈值 → `should_stop=True, reason=REPEATED_FAILURE`，details 含命中的
    entry 字段。未超阈值 → `should_stop=False`。

    未来可以扩展：同 error_class 跨 tool 的重复也应触发。当前只看
    单 (tool, args_hash) 的本轮 count。
    """
    hits = failure_memory.entries_above_threshold(threshold)
    if not hits:
        return StopDecision(should_stop=False)

    first_hit = hits[0]
    return StopDecision(
        should_stop=True,
        reason=StopReason.REPEATED_FAILURE,
        details={
            "tool_key": first_hit["tool_key"],
            "args_hash": first_hit["args_hash"],
            "failure_count": first_hit["failure_count"],
            "last_error_type": first_hit["last_error_type"],
            "suggested_next_action": first_hit["suggested_next_action"],
            "threshold": threshold,
            "total_hits": len(hits),
        },
    )


def format_forced_stop_message(decision: StopDecision) -> str:
    """把 forced-stop 的 decision 格式化成给用户的最终回答字符串。

    LLM 被强制停时不会自己写 summary，harness/loop 要替它把"为什么停"
    告诉用户，避免用户看到空回答。
    """
    if decision.reason == StopReason.REPEATED_FAILURE:
        d = decision.details
        return (
            f"检测到同一工具同参数连续失败 {d.get('failure_count', '?')} 次（超过阈值 "
            f"{d.get('threshold', '?')}），已主动终止本轮以避免继续消耗。\n"
            f"- 失败工具：`{d.get('tool_key', '?')}`\n"
            f"- 错误类型：{d.get('last_error_type', '?')}\n"
            f"- 建议下一步：{d.get('suggested_next_action', '?')}"
        )
    # 未来扩展别的 reason 时补分支；默认通用文案
    return f"本轮因 {decision.reason.value} 被终止。details: {decision.details}"
