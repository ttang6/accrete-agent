"""Eval harness 数据契约（dataclasses + YAML loader）。

设计取舍：
- 全部 frozen dataclass，跨 phase 不可变（grader / aggregator / report 各自只读）
- 字段保守——能从 trace 直接抽的不进 spec；spec 仅描述 task 自身意图
- mode 仅 baseline / evolved 二档；mock 走单独旁路（smoke 测试用）
- summary dict 而非 dataclass：metric 集合可能演进，dict 比 dataclass 更宽容
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import yaml


@dataclass(frozen=True)
class TaskSpec:
    """单 eval task 定义。从 YAML 加载。

    字段语义：
    - id: 唯一 ID，文件名去 .yaml 后缀；run_eval 用此索引 trace
    - description: 人类可读的简短描述（10-40 字），diff_report 表格里展示
    - query: 模拟用户输入文本（送给 harness.handle）。
        str = 单轮；List[str] = 多轮（按顺序喂给同一 harness session，
        grader 评最后一轮 trace——决定性步骤请放最后一轮）
    - skill: 可选——run_eval 跑前先发 `/skill <name>` 切换。None = 不切
    - expected_obligations: action_contract id 列表（grader 比对实际 fulfill 数）
    - expected_tool_calls: 期望出现的 tool call 模式（dict: tool/skill/script）
    - expected_coverage: ai-digest 的 paper/oss/news 等类目（grader 看 coverage missing）
    - max_iterations: 单 task 上限（超过判 success=False, reason="max_iter"）
    - max_duration_s: 单 task 时长上限（subprocess timeout 兜底）
    - success_keywords: final answer 软匹配关键词（任一命中算"内容相关"）
    - tier: 难度档 single | multi | long_multi —— 决定 pass^k（eval-only，agent 看不到）。
        未显式给时由 query 类型回退：str → single，List[str] → multi。
    - inject_failures: 确定性失败注入规格列表（eval-only，agent 看不到）。每条 dict：
        {target: "skill:ai-digest/fetch_rss" | "tool:arxiv",
         on_call: 第 N 次匹配调用时失败（int, 从 1 起）,
         mode: timeout|rate_limit|malformed|unavailable|schema_mismatch,
         recovers_to: 可选，标注期望恢复方式（分析/部分信用用，注入器不消费）}
        由 evals/failure_injection.py 解释执行；生产路径 inject_failures 恒为空 → no-op。
    """
    id: str
    description: str
    query: Union[str, List[str]]
    skill: Optional[str] = None
    expected_obligations: List[str] = field(default_factory=list)
    expected_tool_calls: List[dict] = field(default_factory=list)
    expected_coverage: List[str] = field(default_factory=list)
    max_iterations: int = 25
    max_duration_s: int = 300
    success_keywords: List[str] = field(default_factory=list)
    tier: str = "single"
    inject_failures: List[dict] = field(default_factory=list)
    recovery_type: str = ""  # R1_transient/R2_switch_source/R3_arg_fix/R4_unrecoverable/observed_empty（eval-only 元数据，供 aggregator 分桶）


@dataclass(frozen=True)
class TaskScore:
    """单 task 跑完的评分（grader 从 trace JSONL 抽）。"""
    task_id: str
    success: bool                           # finish 而非 max_iter / error
    finish_reason: str                      # "finish" / "max_iter" / "error" / "timeout"
    total_steps: int
    llm_calls: int
    tool_calls: int
    tool_failures: int                      # output 命中 _is_tool_failure
    obligations_required: int               # spec.expected_obligations 数
    obligations_repair_injected: int        # ACTION_OBLIGATION_REPAIR_INJECTED 数
    obligations_violations: int             # ACTION_OBLIGATION_VIOLATION 数
    expected_tool_calls_hit: int            # 命中 spec.expected_tool_calls 的数
    expected_tool_calls_required: int       # spec.expected_tool_calls 数
    lesson_uses: int                        # ACTION_LESSON_USED 数
    lesson_helped: int                      # ACTION_OUTCOME_UPDATE outcome=helped
    lesson_hurt: int                        # 同上 hurt
    lesson_ineffective: int                 # 同上 ineffective
    coverage_missing: List[str]             # 最终 coverage_check missing
    final_answer_chars: int
    duration_ms: int
    trace_path: str
    total_tokens: int = 0       # run_summary.tokens 累加（成本指标）
    recovery_type: str = ""     # 从 spec 透传，供 aggregator 按 R1-R4 分桶报告

    @property
    def obligation_completion_rate(self) -> float:
        if self.obligations_required == 0:
            return 1.0  # 无 obligation 默认满分
        # 完成 = 不在 violations 里（repair_injected 算"被拦下后仍调到了"）
        return max(0.0, 1.0 - self.obligations_violations / self.obligations_required)

    @property
    def tool_failure_rate(self) -> float:
        if self.tool_calls == 0:
            return 0.0
        return self.tool_failures / self.tool_calls

    @property
    def expected_tool_calls_coverage(self) -> float:
        if self.expected_tool_calls_required == 0:
            return 1.0
        return self.expected_tool_calls_hit / self.expected_tool_calls_required


@dataclass(frozen=True)
class EvalConfig:
    """单次跑（baseline 或 evolved）的环境配置。"""
    mode: Literal["baseline", "evolved", "mock"]
    enable_lesson_recall: bool
    enable_promotion_gate: bool
    sqlite_snapshot_path: Optional[Path] = None  # baseline 跑前备份；evolved 跑前 revert


@dataclass(frozen=True)
class EvalReport:
    """单次跑（baseline 或 evolved）的多 task 聚合。"""
    config: EvalConfig
    started_at: str
    finished_at: str
    task_scores: List[TaskScore]
    summary: dict[str, Any]  # success_rate / avg_steps / lesson_help_rate / 等

    @property
    def total_tasks(self) -> int:
        return len(self.task_scores)


# ============================================================
# YAML loader
# ============================================================


def load_task_spec(yaml_path: Path) -> TaskSpec:
    """从单个 YAML 文件加载 TaskSpec。fail-loud（损坏 YAML 立刻抛）。"""
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path}: YAML root 必须是 dict")
    raw_query = data["query"]  # 必填，缺则 KeyError
    if isinstance(raw_query, list):
        query: Union[str, List[str]] = [str(q) for q in raw_query]
        if not query:
            raise ValueError(f"{yaml_path}: query 是空 list")
    else:
        query = str(raw_query)
    # tier 未显式给时按 query 类型回退（向后兼容旧 fixture：单轮→single，多轮→multi）
    default_tier = "multi" if isinstance(query, list) else "single"
    tier = str(data.get("tier") or default_tier)
    return TaskSpec(
        id=str(data.get("id") or yaml_path.stem),
        description=str(data.get("description", "")),
        query=query,
        skill=data.get("skill"),
        expected_obligations=list(data.get("expected_obligations") or []),
        expected_tool_calls=list(data.get("expected_tool_calls") or []),
        expected_coverage=list(data.get("expected_coverage") or []),
        max_iterations=int(data.get("max_iterations", 25)),
        max_duration_s=int(data.get("max_duration_s", 300)),
        # str 化：YAML 里裸数字（年份 / arxiv id 如 2404.19756）会被解析成 int/float，
        # 下游 " ".join 与 `kw in answer` 都要求 str —— 入口统一强转，杜绝类型炸。
        success_keywords=[str(k) for k in (data.get("success_keywords") or [])],
        tier=tier,
        inject_failures=list(data.get("inject_failures") or []),
        recovery_type=str(data.get("recovery_type") or ""),
    )


def load_task_specs(tasks_dir: Path) -> List[TaskSpec]:
    """从目录加载所有 *.yaml 文件 → 排序后的 TaskSpec 列表。

    deterministic order：按 id 排序，让 baseline / evolved 跑顺序一致便于对照。
    """
    if not tasks_dir.exists():
        raise FileNotFoundError(f"tasks 目录不存在: {tasks_dir}")
    specs = [
        load_task_spec(p)
        for p in sorted(tasks_dir.glob("*.yaml"))
    ]
    return specs
