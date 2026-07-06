"""Tool 失败三层防护（确定性、ephemeral per-turn、不调 LLM）。

按失败的**真实范围**分三层，反应也分层——不把一层的活套到另一层上（避免"某个死调用
就误伤整个工具"这类粒度错配）：

  Layer 1 · Tool-level circuit breaker
      底层**服务**整体挂了（同一 op、各种参数/目标**全**失败）→ 整个工具下线。
      **未实现**（占位 pass，见下）。触发信号需新增，现建属投机。

  Layer 2 · Target-level isolation
      单个**目标**坏了（某 URL / 某 endpoint）→ 只隔离那个目标，不误伤工具。
      **未实现**（占位 pass）。需工具暴露 target 维度，现建属投机。

  Layer 3 · Loop detection / repeat guard
      **相同参数（call_key）反复失败** → 挡住那个具体调用、本轮暂停，不误伤工具。
      **唯一已实现的层**，即本文件下方逻辑（历史上误名 "circuit breaker"，正名待
      改名 pass 收口为 loop detection）。

—— Layer 3 细节（现有行为，未改）——

某个 op（工具自声明的 call_key，见 BaseTool.call_key）在同一 turn 内连续失败时，继续让
LLM 反复调它是浪费——但又不该停整个 run（别的 op 可能还能成）。连续失败达 klass policy
阈值 N → 把该 call_key 本轮禁掉（blocked_calls 挂 AgentRuntimeState），后续同调用直接回一条
`[gate-circuit-open]` 事实消息、不执行，turn 继续、交回 LLM 换路子。state ephemeral：run
结束即丢，不进任何持久库。

klass = **policy 不是 key**：klass（transient/permanent/None，由工具自声明）只决定容忍
几次、是否退避，不参与 call_key 的构成。
- transient（超时/连接/5xx/429）→ N=3：值得多给几次（in-tool 已短退避过一轮）
- permanent（4xx client error）→ N=1：不会自愈，第一次就禁
- None（说不清）→ N=3：保守容忍
- is_mutating（不可逆写）→ N=1 fail-fast：避免重复副作用，压过 klass

计数源单一：不自己计数，直接读 FailureMemory.entries[call_key].failure_count
（main_loop 已让 FailureMemory 按同一把 call_key 计数）——两套计数不漂移。
"""

from __future__ import annotations

from typing import Optional

from accrete.runtime.context_sources import MARKER_GATE_CIRCUIT_OPEN


# ============================================================
# Layer 1 · Tool-level circuit breaker —— 占位，未实现
# ============================================================
def tool_service_down(op: str) -> bool:  # noqa: ARG001
    """判断某个工具依赖的底层服务是不是整体挂了、该不该把整个工具下线。

    未实现（占位）。将来的触发条件：同一个工具用各种不同参数去调都失败，说明不是某次
    调用的问题、而是它依赖的服务整体不可用，这时才把整个工具下线。现在没有这种失败的
    实际证据，先不建。

    Args:
        op: 工具的操作标识（同一工具可有多种操作）。

    Returns:
        整个工具是否应下线；未实现，恒 None。
    """
    pass


# ============================================================
# Layer 2 · Target-level isolation —— 占位，未实现
# ============================================================
def target_isolated(op: str, target: str) -> bool:  # noqa: ARG001
    """判断某个具体目标（比如某个网址、某个服务地址）是不是坏了、该不该单独隔离它。

    未实现（占位）。和整体下线不同，这里只针对单个坏目标，其它目标照常能用。将来要做，
    需要工具先能报出自己这次访问的是哪个目标。现在没有"某个目标反复坏、其它正常"的实际
    证据，先不建。

    Args:
        op: 工具的操作标识。
        target: 这次访问的具体目标，比如网址或服务地址。

    Returns:
        该目标是否应被隔离；未实现，恒 None。
    """
    pass


# ============================================================
# Layer 3 · Loop detection / repeat guard —— 已实现
# ============================================================

_BREAKER_N_TRANSIENT = 3   # 可恢复失败给 3 次（in-tool 退避 + 熔断容忍；RLEF 式更宽 budget）
_BREAKER_N_DEFAULT = 3     # klass=None（skill_exec 子进程串等说不清的失败）同样宽容到 3
_BREAKER_N_PERMANENT = 1
_BREAKER_N_MUTATING = 1    # 不可逆写 fail-fast，压过 klass


def loop_block_threshold(klass: Optional[str], is_mutating: bool) -> int:
    """返回禁用该 call_key 前容忍的连续失败次数 N。is_mutating 压过 klass。"""
    if is_mutating:
        return _BREAKER_N_MUTATING
    if klass == "permanent":
        return _BREAKER_N_PERMANENT
    if klass == "transient":
        return _BREAKER_N_TRANSIENT
    return _BREAKER_N_DEFAULT


def format_loop_block_message(
    tool_name: str, call_key: str, failure_count: int, klass: Optional[str]
) -> str:
    """loop 挡回消息：只报事实、不给建议（怎么换路子留给 LLM 自己想）。

    格式：`[gate-circuit-open] 熔断：{tool} {目标} 连续失败 {N} 次（{原因}），本轮已暂停。`
    目标 = call_key 去掉 `tool:` 前缀；原因从 klass 来，没有则省略括号。
    """
    target = call_key.split(":", 1)[1] if ":" in call_key else call_key
    reason = f"（{klass}）" if klass else ""
    return (
        f"{MARKER_GATE_CIRCUIT_OPEN} 熔断：{tool_name} {target} "
        f"连续失败 {failure_count} 次{reason}，本轮已暂停。"
    )
