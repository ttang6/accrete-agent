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

from typing import Final, List, Tuple

from nanoagent.evolution.runtime_memory.schema import FailureEvent

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

    def __init__(self, repeated_threshold: int = 2):
        """repeated_threshold：同 (tool_key, args_hash) 出现 ≥ 阈值次数即判 repeated。
        默认 2——配合 FailureMemory 的 augment 触发点（2nd 次失败）。"""
        self._repeated_threshold = repeated_threshold

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
        if fe.tool_key and fe.args_hash:
            same_count = sum(
                1
                for other in episode_failures
                if other.tool_key == fe.tool_key
                and other.args_hash == fe.args_hash
            )
            if same_count >= self._repeated_threshold:
                return "repeated_same_args_failure"

        # 4. schema 类
        if fe.error_type == "schema_mismatch":
            return "schema_mismatch"

        # 5. 兜底
        return "tool_runtime_error"
