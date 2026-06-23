"""TraceErrorClassifier — FailureEvent + episode 上下文 → error_class enum。

规则式分类（无 LLM）：
- schema_mismatch
- repeated_same_args_failure
- coverage_gap
- soft_quality_issue
- semantic_failure  (skill script 输出 structured semantic_failures 的通道)
- tool_runtime_error

决策树按优先级，第一个 match 即返回。优先级理由：
1. coverage_gap / soft_quality_issue 用 error_type 区分（EpisodeExtractor 已打标）
2. semantic_failure：extras 含 skill 输出的结构化失败列表——比 generic runtime error
   信号丰富；先于 repeated/schema 命中，让 lesson 模板能用 failure_type / message
3. repeated_same_args_failure 早于 schema_mismatch——3 次同参数失败比"首次 schema 错"
   更值得学；金标 trace 里 dup_check 三次失败都应归到 repeated 类别（即便首次是
   schema_mismatch，后面 hint 没接住才是真正要学的"症状未消"信号）
4. schema_mismatch 兜底 schema 类
5. else → tool_runtime_error

`session_intent_drift` / `context_overload` / `user_correction` 三种启发式留待后续实现
（需要 LLM 协助 + 跨 turn 上下文，纯规则覆盖率低）。
"""

from __future__ import annotations

import os
from typing import Final, List, Optional, Tuple

from nanoagent.evolution.runtime_memory.schema import FailureEvent


def _schema_over_repeat_default() -> bool:
    """env 开关：选项③——schema_mismatch 坑即便被同参重试 ≥ 阈值，也判 schema 而非
    repeated（保住"加哪个参数"的具体修法，repeated 通用废话洗不掉它）。

    默认关（保持现状：repeated 优先）。改飞轮"学什么"的行为，先做成可开关、拿数据
    验证后再决定默认值（阶段六 §1）。"""
    return os.getenv("NANOAGENT_CLASSIFIER_SCHEMA_OVER_REPEAT", "").strip().lower() in (
        "1", "true", "yes", "on",
    )

ERROR_CLASSES: Final[Tuple[str, ...]] = (
    "schema_mismatch",
    "repeated_same_args_failure",
    "coverage_gap",
    "soft_quality_issue",
    "semantic_failure",
    "tool_runtime_error",
)


class TraceErrorClassifier:
    """规则式分类器。无状态，可作单例使用。"""

    def __init__(
        self,
        repeated_threshold: int = 2,
        schema_wins_over_repeat: Optional[bool] = None,
    ):
        """repeated_threshold：同 (tool_key, args_hash) 出现 ≥ 阈值次数即判 repeated。
        默认 2——配合 FailureMemory 的 augment 触发点（2nd 次失败）。

        schema_wins_over_repeat（选项③）：None 时读 env 默认（默认 False）。True 时
        schema_mismatch 坑即便同参重试也归 schema、不被 repeated 截胡——让蒸出的 lesson
        带具体修法而非通用"别重试"。repeated 仍兜非 schema 的纯重试。"""
        self._repeated_threshold = repeated_threshold
        self._schema_wins_over_repeat = (
            _schema_over_repeat_default()
            if schema_wins_over_repeat is None
            else schema_wins_over_repeat
        )

    def classify(
        self, fe: FailureEvent, episode_failures: List[FailureEvent]
    ) -> str:
        # 1. coverage_gap / soft_quality_issue 已由 EpisodeExtractor 打标
        if fe.error_type == "coverage_gap":
            return "coverage_gap"
        if fe.error_type == "soft_quality":
            return "soft_quality_issue"

        # 2. 通用 semantic_failure 通道：skill script 输出 JSON 含 semantic_failures
        # 列表时已被 EpisodeExtractor 抽进 extras。优先于 repeated/schema 命中，让
        # lesson 模板能用结构化 failure_type 与 message。
        sf = (fe.extras or {}).get("semantic_failures") if fe.extras else None
        if isinstance(sf, list) and sf:
            return "semantic_failure"

        # 3. 同参数重复失败 ≥ 阈值（含本条）
        # 选项③开启时：schema_mismatch 坑不被 repeated 截胡——具体修法（加哪个参数）
        # 比通用"别重试"更值得学；repeated 仍兜非 schema 的纯重试。
        if fe.tool_key and fe.args_hash:
            same_count = sum(
                1
                for other in episode_failures
                if other.tool_key == fe.tool_key
                and other.args_hash == fe.args_hash
            )
            if same_count >= self._repeated_threshold and not (
                self._schema_wins_over_repeat and fe.error_type == "schema_mismatch"
            ):
                return "repeated_same_args_failure"

        # 4. schema 类
        if fe.error_type == "schema_mismatch":
            return "schema_mismatch"

        # 5. 兜底
        return "tool_runtime_error"
