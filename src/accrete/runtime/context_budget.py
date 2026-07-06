"""ContextBudget——per-run 的 token 预算 + 决策状态机。

替代 MainLoop 内联的 `ctx_tokens_est > self.max_context_tokens` 单点 warning
（main_loop.py:217-223），把"context 已涨到哪一步"+"该不该 force_store
新 tool output"两个决策集中到一处。

入门视角：
- 每次 MainLoop.run() 新建一份 ContextBudget（per-run 状态，run 结束丢）
- 维护两个 running tally：
  1. `accumulated_tool_tokens`：本 run 内所有 tool output 的 token 总和
  2. `warned_levels`：已经 emit 过哪个 warning level（防止每 iter 刷屏）
- 三个查询：
  1. `should_force_store()`: 累计 tool tokens 超 max_total_tool_tokens_per_run
     → caller 让 ToolOutputStore 强制落盘所有后续 tool output
  2. `check_total_context(total_tokens)`: 总 context 超 warn / hard ratio →
     返回 'warning' / 'exceeded' / None，caller 据此 emit trace event
  3. `record_tool_tokens(n)`: 累加 tool tokens（caller 在 tool 执行后调）

关键决策——超 hard_ratio 时**只发 trace 事件不动 messages**：
- 主动截 history / 压 hint 都是有副作用的应用层决策，不放底层
- HistoryManager 已有 max_messages=100 硬截作兜底
- LLM provider 自身会在超 context window 时报错，是更明确的失败信号
- 第三方建议的"先压 tool / 再压 evaluator / 再压 lesson / 再压 history"被驳回——
  evaluator/lesson/runtime_hint 是**信号**不是噪声

不做的事：
- 不做 history 截断（HistoryManager 兜底）
- 不做 hint 数量限制（hint 是信号）
- 不做按 token / 字符的 LLM provider 真实计数（TokenCounter 估算够用）

下游：
- MainLoop._run_inner 每 iter 末调 check_total_context → emit trace
- MainLoop._run_inner tool 执行后调 record_tool_tokens + should_force_store
- main.py 装配区可定制 ContextBudgetConfig（None 时用默认）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Literal, Optional


# warning level 常量。返回值约束在这三个字面量。
LEVEL_WARNING: Final[str] = "warning"
LEVEL_EXCEEDED: Final[str] = "exceeded"


@dataclass(frozen=True)
class ContextBudgetConfig:
    """ContextBudget 的不可变配置。main.py 装配时传入，run 内不变。

    默认值参考 main_loop.py 当前 max_context_tokens=80_000：
    - warn_ratio=0.8 → 64K trace warning
    - hard_ratio=0.95 → 76K trace exceeded（不主动截）
    - max_single_tool_tokens=2_000 → 比当前 max_tool_output_chars=5000(~1250 tok)
      放宽 60%；日报场景 fetch_hf 经常贴近上限
    - max_total_tool_tokens_per_run=12_000 → ~6 tool 平均 2K，留足余量给 history
    """

    max_context_tokens: int = 80_000
    warn_ratio: float = 0.8
    hard_ratio: float = 0.95
    max_single_tool_tokens: int = 2_000
    max_total_tool_tokens_per_run: int = 12_000

    def __post_init__(self) -> None:
        # frozen dataclass 也可以做 sanity check
        if not 0.0 < self.warn_ratio < self.hard_ratio <= 1.0:
            raise ValueError(
                f"必须 0 < warn_ratio({self.warn_ratio}) < "
                f"hard_ratio({self.hard_ratio}) <= 1.0"
            )
        if self.max_context_tokens <= 0:
            raise ValueError(f"max_context_tokens must be > 0")
        if self.max_single_tool_tokens <= 0:
            raise ValueError(f"max_single_tool_tokens must be > 0")
        if self.max_total_tool_tokens_per_run <= 0:
            raise ValueError(f"max_total_tool_tokens_per_run must be > 0")

    @property
    def warn_threshold(self) -> int:
        """向上取整避免 float 比较抖动。"""
        return int(self.max_context_tokens * self.warn_ratio)

    @property
    def hard_threshold(self) -> int:
        return int(self.max_context_tokens * self.hard_ratio)


@dataclass
class ContextBudget:
    """per-run 状态机。每次 MainLoop.run() 新建一份。

    用法（caller 视角）：
        budget = ContextBudget(config=cfg)
        for iteration in ...:
            # 跑完一个 iter 后
            tool_tokens = counter.count_text(tool_output)
            budget.record_tool_tokens(tool_tokens)
            if budget.should_force_store():
                # 让后续 tool output 都强制落盘
                ...
            level = budget.check_total_context(total_tokens)
            if level:
                tracer.step(action=ACTION_CONTEXT_BUDGET_WARNING, ...)

    设计取舍：
    - check_total_context 只对每个 level 通知一次（防 iter 刷屏）。同一个
      level 只会 emit 一次 trace event。
    - 没暴露 reset() —— per-run 实例，重启即新对象，不需要 reset
    - record_tool_tokens 接受 0 也合法（调用方不需要先判断 store 是否触发）
    """

    config: ContextBudgetConfig
    accumulated_tool_tokens: int = 0
    warned_levels: set[str] = field(default_factory=set)

    def record_tool_tokens(self, tokens: int) -> None:
        """累加一次 tool output 的 token 数。"""
        if tokens < 0:
            return  # 防御 caller 误传负数
        self.accumulated_tool_tokens += tokens

    def should_force_store(self) -> bool:
        """累计 tool tokens 是否已超总额度——caller 据此让 ToolOutputStore
        force_store=True。

        返回 True 后，所有后续 tool output 都强制落盘，prompt 里只留 preview。
        这避免单个 run 内几个 fetch_* 累计撑爆 context。
        """
        return self.accumulated_tool_tokens > self.config.max_total_tool_tokens_per_run

    def check_total_context(self, total_tokens: int) -> Optional[str]:
        """根据当前 total context tokens 决定 warning / exceeded / None。

        优先返回更高级别（exceeded > warning），且每个级别一 run 内只返回
        一次（避免每 iter 都 emit）。

        Returns:
            LEVEL_EXCEEDED ("exceeded"): 首次跨过 hard_threshold
            LEVEL_WARNING ("warning"): 首次跨过 warn_threshold（且未超 hard）
            None: 未达阈值，或对应级别已 warn 过

        Note: caller 应只在返回非 None 时 emit trace event。返回的字符串
        值就是建议的 trace event level 字段。
        """
        if total_tokens >= self.config.hard_threshold:
            if LEVEL_EXCEEDED not in self.warned_levels:
                self.warned_levels.add(LEVEL_EXCEEDED)
                # 同时也标记 warning（未来不再重复 warn 较低 level）
                self.warned_levels.add(LEVEL_WARNING)
                return LEVEL_EXCEEDED
            return None

        if total_tokens >= self.config.warn_threshold:
            if LEVEL_WARNING not in self.warned_levels:
                self.warned_levels.add(LEVEL_WARNING)
                return LEVEL_WARNING
            return None

        return None

    def snapshot(self) -> dict:
        """状态快照——给 trace finalize 时附加。后续 ReflexionStore 可消费。"""
        return {
            "accumulated_tool_tokens": self.accumulated_tool_tokens,
            "max_total_tool_tokens": self.config.max_total_tool_tokens_per_run,
            "max_single_tool_tokens": self.config.max_single_tool_tokens,
            "warn_threshold": self.config.warn_threshold,
            "hard_threshold": self.config.hard_threshold,
            "warned_levels": sorted(self.warned_levels),
        }
