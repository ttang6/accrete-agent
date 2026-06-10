"""Turn-scoped context for MainLoop：硬覆盖检查 + 失败记忆 recovery。

协议级自主性基础设施。每次 `MainLoop.run()` 构造一份新实例，run 结束即丢弃。

本文件职责：
- `CoverageChecker` —— 解析 tool 输出 markdown header 抽 item 数，
  按 manifest 声明的 category specs 判达标，不够则 build_hint 供 MainLoop 注入 LLM
- `TurnContext` 聚合门面——持 coverage + failure_memory 两份状态

CoverageChecker 完全不知道任何具体 skill。category 定义
全部来自 `skills/<name>/skill.yaml` 的 coverage 段；装配层（main.py / harness）通过
`aggregate_coverage_specs(contracts)` union 合并多 skill manifest 后传入。未传 specs
时 CoverageChecker 退化为"无任何 category 检查"——`missing_categories()` 空 →
`build_hint()` 返 None，等价于跳过 coverage 检查（适合非 ai-digest skill / 无 manifest
启动场景）。

FailureMemory 实际代码在 `failure_memory.py`，这里只 re-export 保持
向后兼容的 import 路径（`from nanoagent.runtime.turn_context import FailureMemory`
仍然可用）。拆出去是为了给 ReflexionStore 提供清晰的消费出口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# re-export：FailureMemory 搬到独立模块后仍保持向后兼容 import
from nanoagent.runtime.failure_memory import (  # noqa: F401
    FailureEntry,
    FailureMemory,
    _canonical_args_hash,
    _classify_tool_failure,
    _is_tool_failure,
    _operation_key,
)
from nanoagent.runtime.obligation_tracker import ObligationTracker
from nanoagent.skills.contract import CoverageCategorySpec

# ============================================================
# CoverageChecker
# ============================================================


@dataclass
class CoverageChecker:
    """硬覆盖检查：按 manifest 声明的 category specs 判 item 数达标。

    per-category 取 max across calls（同源 retry 不累加），避免 LLM 反复调
    某个 fetch tool 放大计数。

    区分 "未观察" vs "观察过但事实空源"：
    - 未观察（首次失败 / 非 ai-digest tool）→ 仍判 missing，hint 告诉 LLM 补
    - 观察过但 count 始终 ≤ 0（如 RSS 24h 无新文章是真实事实）→ 不再 missing
    避免 evaluator 看到 missing=[news] 触发无意义 retry——RSS 空源被反复要求
    补 fetch 会浪费大量步数。

    `category_specs` 替代之前的 `_CATEGORY_MAP` /
    `_DEFAULT_THRESHOLDS` / `_COUNT_RE` 模块级常量。空 list 默认值意味着
    framework 不内建任何 skill 知识——装配层从 SkillContract.coverage_manifest
    union 后显式传入；未传则跳过 coverage 检查（missing 永远空）。
    """

    category_specs: List[CoverageCategorySpec] = field(default_factory=list)
    # thresholds 接受外部 override；空 dict 时由 specs 派生（构造 hook 在 __post_init__）。
    # 历史 API：旧 callsite 传 `thresholds={"paper": 5}` 时，覆盖对应 category 的阈值。
    thresholds: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    # observed 过至少一次但 count 始终 ≤ 0 的 category——视为"事实空源"已确认
    observed_empty: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        # 旧 API 语义：调用方显式 thresholds={...} → 完全替换（也决定哪些 category
        # 参与达标检查）。空 dict（默认 default_factory）→ 由 specs 派生 thresholds。
        # 不做 merge：避免旧测试 `CoverageChecker(thresholds={"news": 1})`
        # 语义破坏（其期望 paper/oss 不参与 missing 判定）。
        if not self.thresholds:
            self.thresholds = {s.name: s.threshold for s in self.category_specs}
        # 派生快查表（构造期一次性算好，observe 路径 O(1)）
        self._spec_by_key: dict[
            tuple[str, Optional[str], Optional[str]], CoverageCategorySpec
        ] = {(s.tool, s.skill, s.script): s for s in self.category_specs}
        self._spec_by_name: dict[str, CoverageCategorySpec] = {
            s.name: s for s in self.category_specs
        }

    def _resolve_category(self, tool_name: str, kwargs: dict) -> Optional[str]:
        """把 (tool_name, kwargs) 映射到 coverage category 名，无映射返回 None。"""
        skill = kwargs.get("skill") if isinstance(kwargs, dict) else None
        script = kwargs.get("script") if isinstance(kwargs, dict) else None
        spec = self._spec_by_key.get((tool_name, skill, script))
        return spec.name if spec else None

    def _extract_count(self, tool_output: str, category: str) -> int:
        """从 tool 输出第一行 markdown header 抽 item count，使用 spec 的
        count_pattern。无 header / 无数字 → 0（用于空返回场景）。"""
        spec = self._spec_by_name.get(category)
        if spec is None:
            return 0
        for line in tool_output.splitlines():
            stripped = line.strip()
            if stripped.startswith("# "):
                m = spec.count_pattern.search(stripped)
                return int(m.group(1)) if m else 0
        return 0

    def observe(self, tool_name: str, kwargs: dict, tool_output: str) -> Optional[tuple[str, int]]:
        """观察一次 tool 输出，更新对应 category 的 max count。

        Returns: (category, count) 当 observe 生效时；None 当 tool 不在 category map 里。
        observe 成功后 counts[category] 必然被 set（即使 count=0），方便调用方判断
        "本轮是否 observe 过此 category"。

        副作用：count > 0 时把 category 从 observed_empty 移除（即使之前空过，
        本次有数据就不再算"事实空源"）；count == 0 且当前 max 仍为 0 时加入
        observed_empty。
        """
        category = self._resolve_category(tool_name, kwargs)
        if category is None:
            return None
        count = self._extract_count(tool_output, category)
        prev = self.counts.get(category, 0)
        new_max = max(prev, count)
        self.counts[category] = new_max

        if new_max > 0:
            self.observed_empty.discard(category)
        elif new_max == 0:
            self.observed_empty.add(category)

        return (category, new_max)

    def missing_categories(self) -> list[str]:
        """返回未达阈值的 category 列表（按 threshold map 迭代顺序）。

        observed_empty 中的 category 视为已 confirmed empty，
        不算 missing——避免对事实空源反复要求补 fetch。
        """
        missing = []
        for cat, threshold in self.thresholds.items():
            if cat in self.observed_empty:
                continue  # 事实空源跳过
            if self.counts.get(cat, 0) < threshold:
                missing.append(cat)
        return missing

    def build_hint(self) -> Optional[str]:
        """若有 missing category，构造一条提示给 LLM；否则 None。

        例：missing=['paper', 'oss'] →
            "[harness-coverage] 当前覆盖未达阈值：paper 0/3, oss 1/2。
             建议补调：fetch_hf / fetch_github（可放宽 max_results 或调整 sort）。"

        修复建议来源：spec.suggestion（manifest 声明）→ spec.script（无 suggestion 时）
        → 跳过（两者都 None 的 spec）。
        """
        missing = self.missing_categories()
        if not missing:
            return None
        parts = []
        for cat in missing:
            parts.append(f"{cat} {self.counts.get(cat, 0)}/{self.thresholds[cat]}")
        suggestions: list[str] = []
        for cat in missing:
            spec = self._spec_by_name.get(cat)
            if spec is None:
                continue
            sugg = spec.suggestion or spec.script
            if sugg:
                suggestions.append(sugg)
        hint = (
            "[harness-coverage] 当前覆盖未达阈值："
            + ", ".join(parts)
            + "。建议补调："
            + " / ".join(suggestions)
            + "（可放宽 max_results 或调整 sort / category）。"
        )
        return hint


# ============================================================
# TurnContext（聚合门面）
# ============================================================


@dataclass
class TurnContext:
    """MainLoop 每次 run() 构造一份。聚合 coverage + failure_memory + obligation_tracker。

    disabled_ops：per-op 熔断状态（ephemeral per-turn）。op_key → 已禁用时回给 LLM
    的 `[熔断]` 消息。run 结束即丢，不持久化（见 circuit_breaker.py）。
    tool_outcomes：每次 tool 调用一个 bool（True=失败），供滑窗失败率总闸判定。
    """

    coverage: CoverageChecker = field(default_factory=CoverageChecker)
    failure_memory: FailureMemory = field(default_factory=FailureMemory)
    obligation_tracker: ObligationTracker = field(default_factory=ObligationTracker)
    disabled_ops: dict[str, str] = field(default_factory=dict)
    tool_outcomes: list[bool] = field(default_factory=list)

    @classmethod
    def create(
        cls,
        coverage_thresholds: Optional[dict[str, int]] = None,
        lesson_retriever=None,
        coverage_specs: Optional[List[CoverageCategorySpec]] = None,
    ) -> "TurnContext":
        """工厂方法：可选 manifest specs + 可选阈值 override + 可选 backend lesson retriever。

        `coverage_specs` 是装配层从 SkillContract.coverage_manifest
        union 后传的 specs 列表；为 None 时 CoverageChecker 退化为无 category 检查
        （missing 永远空，build_hint 返 None）。`coverage_thresholds` 优先级最高
        （覆盖任何 spec.threshold），保旧 API 兼容。
        """
        kwargs: dict = {}
        if coverage_specs is not None:
            kwargs["category_specs"] = list(coverage_specs)
        if coverage_thresholds is not None:
            kwargs["thresholds"] = dict(coverage_thresholds)
        coverage = CoverageChecker(**kwargs)
        return cls(
            coverage=coverage,
            failure_memory=FailureMemory(lesson_retriever=lesson_retriever),
            obligation_tracker=ObligationTracker(),
        )
