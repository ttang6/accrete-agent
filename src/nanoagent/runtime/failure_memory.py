"""FailureMemory — 失败计数 + lesson 召回模块（harness-recovery 已退役）。

跟踪 operation_key（意图级）的失败次数 + 错误类型 + 下一步建议。
计数 key = operation_key（skill 脚本=`skill_exec:skill/script`；fetch 按 host；其余=tool_name），
不含 args_hash（args_hash 降级为 entry 内 debug metadata）。
`[harness-recovery]` hint 已退役——2nd+ 次失败默认不注入提示文本；保留
failure_count（供 stop_condition.check_repeated_failure 防循环 loop guard 读）。
1st 次失败的 lesson 召回（[runtime-lesson]）保留——那是飞轮通道，不是 harness-recovery。
可选 `online_reflector`（在线微反思）：同意图（recall_key）第 2 次失败且飞轮
无可召回的 suggested_action 时，副 LLM 基于报错产一个 [online-reflector] 修复
假设注入——覆盖"飞轮产不出修复教训"的 hard-tail（默认不装配，开关见 main.py）。
触发计数特意用意图级而非 op_key（带 args-hash）：换参重试仍失败才最需要外脑。

入门友好视角：
- 一个 turn（MainLoop.run）创建一份 FailureMemory，run 结束就扔掉
- 只记录"失败"调用，成功的不占空间
- 判"失败"靠 tool_output 字符串里的前缀（`[skill_exec 错误]` 等），不依赖异常对象
- 判"错误类型"靠关键词扫描——schema_mismatch / transient / unknown 三档
- "建议下一步"是纯查表：schema_mismatch → describe_script 等

ReflexionStore 消费出口：
- `iter_records()`：把每条 entry 输出为 dict，便于 for-loop 转 ReflexionRecord
- `snapshot()`：整个 state 的深拷贝 dict，给 tracer 写 finish 时附加

lesson_retriever 注入：
- 可选传入 `lesson_retriever`——非 None 时，第 1 次失败也查 backend 中的
  promoted lesson；命中即注入 [runtime-lesson] 提示
- 不传 / 为 None 时，行为与不注入时完全一致（不影响既有调用方）

Not to be confused with `core/error_classifier.py`——那个分类异常对象，
这个分类 tool-output 字符串。两套标签命名有意不同。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Iterator, Optional, Tuple

# 失败签名 / 分类 / args_hash 三个常量与函数搬到 SSoT 模块。本文件通过 re-export
# 保留旧 import 路径（`from nanoagent.runtime.failure_memory import _is_tool_failure`
# 等），物理上 `failure_memory._FAILURE_SIGNATURES is tool_failure._FAILURE_SIGNATURES`
# 即可证明 drift 不可能再发生。详见 `runtime/tool_failure.py` 模块 docstring。
from nanoagent.runtime.tool_failure import (  # noqa: F401
    _FAILURE_SIGNATURES,
    _SCHEMA_MISMATCH_PATTERNS,
    _TRANSIENT_PATTERNS,
    _canonical_args_hash,
    _classify_tool_failure,
    _is_tool_failure,
)

if TYPE_CHECKING:
    # 避免运行时循环依赖：lesson_retriever 顶层 import schema/backend，
    # 而 failure_memory 是 runtime 顶层 module，不应被 evolution 依赖
    from nanoagent.evolution.runtime_memory.lesson_retriever import LessonRetriever
    from nanoagent.evolution.runtime_memory.online_reflector import (
        OnlineReflector,
        ReflectorSuggestion,
    )
    from nanoagent.evolution.runtime_memory.schema import RuntimeLesson


# 每 turn 在线微反思调用上限（计尝试次数，含 fail-open 返 None 的）
_REFLECTOR_MAX_CALLS_PER_TURN: Final[int] = 2


# ============================================================
# 分类常量（仅 failure_memory 用，不属 SSoT 范围）
# ============================================================

_SUGGESTED_NEXT_ACTION: Final[dict[str, str]] = {
    "schema_mismatch": "describe_script",
    "transient": "retry_same_later",
    # 非 transient/非 schema 失败（含 4xx / API exception / 源不可用）实测最常靠换工具或换源恢复，
    # 而非改同一调用的参数
    "unknown": "switch_tool_or_source",
}


# ============================================================
# 纯函数 helpers（仅 failure_memory 内部用）
# ============================================================


def _url_host(url: str) -> str:
    """从 URL 取 host（含端口）；失败回退到截断字符串。用于 fetch 的意图级计数。"""
    from urllib.parse import urlparse
    try:
        return urlparse(url).netloc or (url[:40] if url else "?")
    except Exception:
        return url[:40] if url else "?"


def _operation_key(tool_name: str, kwargs: dict, raw_args: str) -> tuple[str, str]:
    """构造 (count_key, recall_key)。

    - count_key：失败计数键（意图级，不含 args）。skill 脚本=`skill_exec:skill/script`；
        fetch 按 host；其余=tool_name。
    - recall_key：lesson 召回匹配键，对齐 `episode_extractor._build_tool_key`。
    """
    if tool_name == "skill_exec" and isinstance(kwargs, dict):
        k = f"skill_exec:{kwargs.get('skill', '?')}/{kwargs.get('script', '?')}"
        return (k, k)
    if tool_name == "fetch" and isinstance(kwargs, dict):
        url = str(kwargs.get("url", "") or "")
        if url:
            return (f"fetch:{_url_host(url)}", "fetch")
    return (tool_name, tool_name)


# ============================================================
# FailureEntry / FailureMemory
# ============================================================


@dataclass
class FailureEntry:
    """一条失败记录。entries 的 key 是 operation_key（意图级）。
    last_args_hash 仅作 debug metadata（不进 key）。"""
    failure_count: int = 0
    last_error_type: str = "unknown"
    suggested_next_action: str = "switch_tool_or_source"
    last_args_hash: str = ""


@dataclass
class FailureMemory:
    """跟踪 operation_key（意图级）的失败次数 + 分类。

    2nd+ 次失败只累加 failure_count（供 stop_condition loop guard），不再注入
    harness-recovery hint（已退役）。可选 `lesson_retriever` 让 1st 次失败命中
    backend promoted lesson 时立即 augment——把跨 trace 经验接入 in-turn 决策。
    """

    entries: dict[str, FailureEntry] = field(default_factory=dict)  # key = operation_key
    # Optional["LessonRetriever"] 字面量化避免运行时 import 循环
    lesson_retriever: Optional["LessonRetriever"] = None
    # 在线微反思（可选）：同意图第 2 次失败且飞轮无 suggested_action 时产修复假设。
    # 触发计数用 recall_key（意图级，skill_exec:skill/script）而非 op_key（带
    # args-hash）——模型"换着参数重试仍失败"恰恰是最需要外脑的时刻，按 op_key
    # 计数会把每次变参当新 op、永远凑不满 2 次（冒烟实测踩到）。
    online_reflector: Optional["OnlineReflector"] = None
    reflector_calls: int = 0
    intent_failures: dict[str, int] = field(default_factory=dict)  # key = recall_key
    reflector_fired: set = field(default_factory=set)  # 已触发过的 recall_key
    # 最近一次注入的 suggestion，main_loop 读后清空（发 REFLECTOR_HINT trace 用；
    # 不改 maybe_augment 的 3 元组返回签名，调用方 / eval monkeypatch 无感）
    pending_reflector_event: Optional["ReflectorSuggestion"] = None

    def maybe_augment(
        self, tool_name: str, kwargs: dict, raw_args: str, result: str,
        *, op_key: Optional[str] = None,
    ) -> Tuple[str, Optional[FailureEntry], Optional["RuntimeLesson"]]:
        """观察一次 tool 输出，若是失败则尝试 augment。

        Returns: (augmented_result, triggered_entry, used_lesson)
          - 成功调用 → (result, None, None) 不动
          - 首次失败 + backend 命中 → (augmented_result, None, lesson) [runtime-lesson] 注入
          - 首次失败 + backend 未命中 → (result, None, None) 只记录不 augment
          - 2nd+ 次失败 → (result, entry, None) 累加 failure_count
            （harness-recovery 已退役；entry 非 None 仅供 main_loop 发
            FAILURE_RECOVERY_HINT 重复失败 telemetry trace）；同意图失败 ≥2 次且
            online_reflector 装配且飞轮无修复时，result 可能附 [online-reflector]
            假设（见 _maybe_reflect），suggestion 存 pending_reflector_event
            供 main_loop 读后清空

        计数键（count_key）：优先用工具自声明的 `op_key`（main_loop 从
        `tool.op_key(kwargs)` 取，是熔断器也会用的同一把键——单一计数源不漂移）；
        op_key 为 None 时退回 `_operation_key` 的默认投影（直接调用 / 单测路径）。
        lesson 召回键（recall_key）始终走 `_operation_key`，跟 episode_extractor
        的存储键对齐（与计数粒度有意不同）。

        设计取舍：
        - 唯一的 in-turn augment 通道是 1st 次失败的 backend lesson 召回
        - retriever 为 None 时退化到只记录不召回的原行为，已有调用方无感
        - failure_count 始终累加——stop_condition.check_repeated_failure 靠它防循环
        """
        if not _is_tool_failure(result):
            return (result, None, None)

        default_count_key, recall_key = _operation_key(tool_name, kwargs, raw_args)
        count_key = op_key if op_key is not None else default_count_key
        args_hash = _canonical_args_hash(raw_args)
        error_type = _classify_tool_failure(result)
        self.intent_failures[recall_key] = self.intent_failures.get(recall_key, 0) + 1
        entry = self.entries.get(count_key)

        if entry is None:
            # 首次失败（按 op_key）：记录 + 查 backend promoted lesson（用 recall_key）
            self.entries[count_key] = FailureEntry(
                failure_count=1,
                last_error_type=error_type,
                suggested_next_action=_SUGGESTED_NEXT_ACTION[error_type],
                last_args_hash=args_hash,
            )
            if self.lesson_retriever is not None:
                lesson = self.lesson_retriever.try_recall(
                    tool_key=recall_key, error_type=error_type
                )
                if lesson is not None:
                    return (
                        result + self.lesson_retriever.format_hint(lesson),
                        None,
                        lesson,
                    )
            # 换 args 重试仍失败 → op_key 是新 entry 但意图级已 ≥2 次，微反思照触发
            hint = self._maybe_reflect(recall_key, count_key, error_type, raw_args, result)
            return (result + hint if hint else result, None, None)

        # 2nd+ 次失败（同 op_key）：累加 failure_count（供 stop_condition loop guard）。
        # harness-recovery hint 已退役；唯一可能的注入是在线微反思。
        entry.failure_count += 1
        entry.last_error_type = error_type
        entry.suggested_next_action = _SUGGESTED_NEXT_ACTION[error_type]
        entry.last_args_hash = args_hash
        hint = self._maybe_reflect(recall_key, count_key, error_type, raw_args, result)
        return (result + hint if hint else result, entry, None)

    def _maybe_reflect(
        self, recall_key: str, count_key: str, error_type: str,
        raw_args: str, result: str,
    ) -> Optional[str]:
        """微反思触发判定：满足全部条件时调副 LLM 产假设、返回 hint 文本。

        条件：装配了 reflector + 同意图（recall_key）失败 ≥2 次 + 该意图未触发过
        + 每 turn 调用上限未到 + 飞轮无可召回的 suggested_action。
        无论副 LLM 是否产出，触发即记入 fired/calls（失败的尝试不重试）。
        """
        if self.online_reflector is None:
            return None
        if self.intent_failures.get(recall_key, 0) < 2:
            return None
        if recall_key in self.reflector_fired:
            return None
        if self.reflector_calls >= _REFLECTOR_MAX_CALLS_PER_TURN:
            return None
        if self._flywheel_has_repair(recall_key, error_type):
            return None
        self.reflector_fired.add(recall_key)
        self.reflector_calls += 1
        suggestion = self.online_reflector.try_suggest(
            tool_key=count_key, raw_args=raw_args, error_output=result,
        )
        if suggestion is None:
            return None
        self.pending_reflector_event = suggestion
        return self.online_reflector.format_hint(suggestion)

    def _flywheel_has_repair(self, recall_key: str, error_type: str) -> bool:
        """backend 里是否已有带 suggested_action 的可召回 lesson。

        有 → 飞轮自己能给修复（首次失败已注入过），微反思不抢戏；
        无 / retriever 未装配 / lesson 只有泛建议没有修复 → 微反思上场。
        """
        if self.lesson_retriever is None:
            return False
        lesson = self.lesson_retriever.try_recall(
            tool_key=recall_key, error_type=error_type
        )
        return lesson is not None and lesson.suggested_action is not None

    # ============================================================
    # 给 ReflexionStore 消费的出口
    # ============================================================

    def iter_records(self) -> Iterator[dict]:
        """把每条 entry yield 成普通 dict（直接 for-loop 消费）。

        shape 与 ReflexionRecord 对齐方便未来 `.to_reflexion_record(trace_id)` 一行转换：
          {tool_key, args_hash, failure_count, last_error_type, suggested_next_action}
        """
        for tool_key, entry in self.entries.items():
            yield {
                "tool_key": tool_key,
                "args_hash": entry.last_args_hash,
                "failure_count": entry.failure_count,
                "last_error_type": entry.last_error_type,
                "suggested_next_action": entry.suggested_next_action,
            }

    def snapshot(self) -> dict:
        """整个 state 的深拷贝 dict（frozen），给 tracer 写 finish 时附加。

        后续修改 entries 不会影响已返回的 snapshot——便于观察 run 结束时
        的"最终态"而非"运行时态"。
        """
        return {
            "entries": [copy.deepcopy(rec) for rec in self.iter_records()],
            "total_failures": sum(e.failure_count for e in self.entries.values()),
        }

    def total_failure_count(self) -> int:
        """所有 entries 的 failure_count 之和——便于 stop_condition 判定。"""
        return sum(e.failure_count for e in self.entries.values())

    def max_failure_count(self) -> int:
        """单 entry 的最高 failure_count——用于 repeated_failure 判定。"""
        if not self.entries:
            return 0
        return max(e.failure_count for e in self.entries.values())

    def entries_above_threshold(self, threshold: int) -> list[dict]:
        """返回 failure_count >= threshold 的 entries（dict 格式）。

        check_repeated_failure 直接消费这个方法。
        """
        hits = []
        for rec in self.iter_records():
            if rec["failure_count"] >= threshold:
                hits.append(rec)
        return hits
