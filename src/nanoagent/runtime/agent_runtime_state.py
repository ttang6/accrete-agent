"""Turn-scoped context for MainLoop：失败记忆 recovery。

协议级自主性基础设施。每次 `MainLoop.run()` 构造一份新实例，run 结束即丢弃。

本文件职责：
- `AgentRuntimeState` 聚合门面——持 failure_memory + 本轮 tool 状态

FailureMemory 实际代码在 `failure_memory.py`，这里只 re-export 保持
向后兼容的 import 路径（`from nanoagent.runtime.agent_runtime_state import FailureMemory`
仍然可用）。拆出去是为了给 ReflexionStore 提供清晰的消费出口。
"""

from __future__ import annotations

from dataclasses import dataclass, field

# re-export：FailureMemory 搬到独立模块后仍保持向后兼容 import
from nanoagent.runtime.failure_memory import (  # noqa: F401
    FailureEntry,
    FailureMemory,
    _canonical_args_hash,
    _classify_tool_failure,
    _is_tool_failure,
    _operation_key,
)


# ============================================================
# AgentRuntimeState（聚合门面）
# ============================================================


@dataclass
class AgentRuntimeState:
    """MainLoop 每次 run() 构造一份。聚合 failure_memory + 本轮 tool 状态。

    blocked_calls：per-op 熔断状态（ephemeral per-turn）。call_key → 已禁用时回给 LLM
    的 `[gate-circuit-open]` 消息。run 结束即丢，不持久化（见 circuit_breaker.py）。
    tool_outcomes：每次 tool 调用一个 bool（True=失败），供滑窗失败率总闸判定。
    """

    failure_memory: FailureMemory = field(default_factory=FailureMemory)
    blocked_calls: dict[str, str] = field(default_factory=dict)
    tool_outcomes: list[bool] = field(default_factory=list)
    # candidate store：本轮所有 tool 输出的未截断原文，按出现序累加。发布流程的
    # 出处核对读它——核"日报机器块里的条目 fingerprint 是否真在本轮抓回的候选里"，
    # 防模型凭空造一条候选集没有的 paper。框架不认"日报"业务，只攒原文；判定在
    # 发布流程（skill 侧）做。run 结束即丢。
    candidates: list[str] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        lesson_retriever=None,
        online_reflector=None,
    ) -> "AgentRuntimeState":
        """工厂方法：可选 backend lesson retriever + online reflector。"""
        return cls(
            failure_memory=FailureMemory(
                lesson_retriever=lesson_retriever,
                online_reflector=online_reflector,
            ),
        )
