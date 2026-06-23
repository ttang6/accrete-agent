"""Per-op 熔断退避策略（确定性、ephemeral per-turn、不调 LLM）。

问题：某个 op（工具自声明的 op_key，见 BaseTool.op_key）在同一 turn 内连续失败时，
继续让 LLM 反复调它是浪费——但又不该停整个 run（别的 op 可能还能成）。

解法：per-op 熔断。连续失败达 klass policy 阈值 N → 把该 op 本轮禁掉
（disabled_ops 挂 TurnContext），后续同 op 调用直接回一条 `[gate-circuit-open]` 事实消息、
不执行，turn 继续、交回 LLM 换路子。state ephemeral：run 结束即丢，不进任何
持久库——源今天挂明天好，"回避"只属本轮（不是可持久化的 lesson）。

klass = **policy 不是 key**：klass（transient/permanent/None，由工具自声明）只决定
容忍几次、是否退避，不参与 op_key 的构成。
- transient（超时/连接/5xx/429）→ N=2：值得多给一次（in-tool 已短退避过一轮）
- permanent（4xx client error）→ N=1：不会自愈，第一次就禁
- None（说不清）→ N=2：保守容忍
- is_mutating（不可逆写）→ N=1 fail-fast：避免重复副作用，压过 klass

计数源单一：熔断不自己计数，直接读 FailureMemory.entries[op_key].failure_count
（main_loop 已让 FailureMemory 按同一把 op_key 计数）——两套计数不漂移。
"""

from __future__ import annotations

from typing import Optional

from nanoagent.runtime.context_sources import MARKER_GATE_CIRCUIT_OPEN

_BREAKER_N_TRANSIENT = 3   # 可恢复失败给 3 次（in-tool 退避 + 熔断容忍；RLEF 式更宽 budget）
_BREAKER_N_DEFAULT = 3     # klass=None（skill_exec 子进程串等说不清的失败）同样宽容到 3
_BREAKER_N_PERMANENT = 1
_BREAKER_N_MUTATING = 1    # 不可逆写 fail-fast，压过 klass


def breaker_threshold(klass: Optional[str], is_mutating: bool) -> int:
    """返回禁用该 op 前容忍的连续失败次数 N。is_mutating 压过 klass。"""
    if is_mutating:
        return _BREAKER_N_MUTATING
    if klass == "permanent":
        return _BREAKER_N_PERMANENT
    if klass == "transient":
        return _BREAKER_N_TRANSIENT
    return _BREAKER_N_DEFAULT


def format_breaker_message(
    tool_name: str, op_key: str, failure_count: int, klass: Optional[str]
) -> str:
    """熔断消息：只报事实、不给建议（怎么换路子留给 LLM 自己想）。

    格式：`[gate-circuit-open] 熔断：{tool} {目标} 连续失败 {N} 次（{原因}），本轮已暂停。`
    目标 = op_key 去掉 `tool:` 前缀；原因从 klass 来，没有则省略括号。
    （marker 收成 `[gate-circuit-open]` 命名空间标签，"熔断"中文移进消息体。）
    """
    target = op_key.split(":", 1)[1] if ":" in op_key else op_key
    reason = f"（{klass}）" if klass else ""
    return (
        f"{MARKER_GATE_CIRCUIT_OPEN} 熔断：{tool_name} {target} "
        f"连续失败 {failure_count} 次{reason}，本轮已暂停。"
    )
