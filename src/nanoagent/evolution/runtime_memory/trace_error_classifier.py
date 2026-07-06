"""TraceErrorClassifier — FailureEvent + episode 上下文 → 单轨 failure_class。

键重构（刀3·3a）后的单轨:工具失败的 failure_class **只用召回时刻可观测的
base-3**（与热路径 `tool_failure.classify_tool_failure` 同源）:
- schema_mismatch
- transient
- unknown  (兜底)

外加两个 **turn 末质量时刻**的类（非工具失败、不进工具失败召回路径，见 3b）:
- soft_quality_issue  (evaluator 事件)
- semantic_failure    (skill script 结构化 semantic_failures 输出)

**same_args 不再是 failure_class**（原 `repeated_same_args_failure` 已删）。理由:
在线召回只在**首次失败**触发（`failure_memory.maybe_augment` 的 `entry is None`
分支），首次失败那一刻同参才失败 1 次、`same_args`(≥2) 还不成立——键里塞召回时
算不出的信息，正是旧 `_LABEL_BRIDGE` 承重墙的根因。现在 same_args 降为**派生传感器**:
熔断器读 `FailureMemory.entries[call_key].failure_count`、微反思读意图级计数触发，
classifier 只把同参计数写进 `fe.extras["same_args_count"]` 作 evidence 精化，不进键。

`fe.error_type` 对工具失败即 SSoT `classify_tool_failure` 的 base-3 输出
（EpisodeExtractor 已打标），本分类器直接读取、不重算——base 标签的单一来源仍是 SSoT。
"""

from __future__ import annotations

from typing import Final, List, Tuple

from nanoagent.evolution.runtime_memory.schema import FailureEvent

FAILURE_CLASSES: Final[Tuple[str, ...]] = (
    "schema_mismatch",
    "transient",
    "unknown",
    "soft_quality_issue",
    "semantic_failure",
)


class TraceErrorClassifier:
    """规则式分类器。无状态，可作单例使用。"""

    def __init__(self, repeated_threshold: int = 2):
        """repeated_threshold：同 (op, args_hash) 出现 ≥ 阈值次数即把
        `same_args_count` 写进 evidence（传感器精化，不改 failure_class）。默认 2——
        对齐 FailureMemory 的 augment 触发点（2nd 次失败）。"""
        self._repeated_threshold = repeated_threshold

    def classify(
        self, fe: FailureEvent, episode_failures: List[FailureEvent]
    ) -> str:
        # 1. turn 末质量时刻的类，已由 EpisodeExtractor 打标（非工具失败）
        if fe.error_type == "soft_quality":
            return "soft_quality_issue"

        # 2. 通用 semantic_failure 通道：skill script 输出 JSON 含 semantic_failures
        # 列表时已被 EpisodeExtractor 抽进 extras。质量时刻信号，非工具失败召回键。
        sf = (fe.extras or {}).get("semantic_failures") if fe.extras else None
        if isinstance(sf, list) and sf:
            return "semantic_failure"

        # 3. same_args 传感器精化：同参失败 ≥ 阈值 → 把计数写进 evidence（不进键）。
        # 熔断/微反思读各自的计数源触发，这里只留一条可供离线审计的 same_args_count。
        if fe.op and fe.args_hash:
            same_count = sum(
                1
                for other in episode_failures
                if other.op == fe.op
                and other.args_hash == fe.args_hash
            )
            if same_count >= self._repeated_threshold:
                if fe.extras is None:
                    fe.extras = {}
                fe.extras["same_args_count"] = same_count

        # 4. 工具失败 base 标签——SSoT 已在 EpisodeExtractor 打标进 error_type
        # （schema_mismatch / transient / unknown）。直接返回,不重算。
        return fe.error_type
