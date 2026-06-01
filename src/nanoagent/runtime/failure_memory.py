"""FailureMemory — 失败记忆 recovery 模块。

跟踪 `(tool_name, args_hash)` 的失败次数 + 错误类型 + 下一步建议。
2nd 次同参数失败时在 tool_result 末尾追加 `[harness-recovery]` 提示，
让 LLM 下一轮必然看到，避免无限 identical_retry。

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
    from nanoagent.evolution.runtime_memory.schema import RuntimeLesson


# ============================================================
# 分类常量（仅 failure_memory 用，不属 SSoT 范围）
# ============================================================

_SUGGESTED_NEXT_ACTION: Final[dict[str, str]] = {
    "schema_mismatch": "describe_script",
    "transient": "retry_same_later",
    "unknown": "change_params",
}


# ============================================================
# 纯函数 helpers（仅 failure_memory 内部用）
# ============================================================


def _failure_key(tool_name: str, kwargs: dict, raw_args: str) -> tuple[str, str]:
    """构造 FailureMemory 的 key。

    tool_name == "skill_exec" 时，用 "skill_exec:<skill>/<script>" 作 name
    前缀（trace 可读性）；其他 tool 直接用 tool_name。args_hash 部分统一走
    canonical。
    """
    if tool_name == "skill_exec" and isinstance(kwargs, dict):
        skill = kwargs.get("skill", "?")
        script = kwargs.get("script", "?")
        prefix = f"skill_exec:{skill}/{script}"
    else:
        prefix = tool_name
    return (prefix, _canonical_args_hash(raw_args))


# ============================================================
# FailureEntry / FailureMemory
# ============================================================


@dataclass
class FailureEntry:
    """一条失败记录。key 是 (tool_key, args_hash) 在 FailureMemory.entries 里。"""
    failure_count: int = 0
    last_error_type: str = "unknown"
    suggested_next_action: str = "change_params"


@dataclass
class FailureMemory:
    """跟踪 (tool_name, args_hash) 的失败次数 + 分类，2nd 次触发 augment。

    可选 `lesson_retriever` 让 1st 次失败也能命中 backend 中的
    promoted lesson 立即 augment——把跨 trace 经验接入 in-turn 决策。
    """

    entries: dict[tuple[str, str], FailureEntry] = field(default_factory=dict)
    # Optional["LessonRetriever"] 字面量化避免运行时 import 循环
    lesson_retriever: Optional["LessonRetriever"] = None

    def maybe_augment(
        self, tool_name: str, kwargs: dict, raw_args: str, result: str
    ) -> Tuple[str, Optional[FailureEntry], Optional["RuntimeLesson"]]:
        """观察一次 tool 输出，若是失败则尝试 augment。

        Returns: (augmented_result, triggered_entry, used_lesson)
          - 成功调用 → (result, None, None) 不动
          - 首次失败 + backend 命中 → (augmented_result, None, lesson) [runtime-lesson] 注入
          - 首次失败 + backend 未命中 → (result, None, None) 只记录不 augment
          - 2nd+ 次失败 → (augmented_result, entry, None) [harness-recovery] 注入

        设计取舍：
        - 第 1 次和第 2nd+ 次的 augment 通道不重叠（一次失败最多一种 hint）
        - 优先级：backend lesson 优先于 in-turn 重复（1st 次能拉回就不等 2nd）
        - retriever 为 None 时退化到只记录不召回的原行为，已有调用方无感
        """
        if not _is_tool_failure(result):
            return (result, None, None)

        key = _failure_key(tool_name, kwargs, raw_args)
        error_type = _classify_tool_failure(result)
        entry = self.entries.get(key)

        if entry is None:
            # 首次失败：先记录
            self.entries[key] = FailureEntry(
                failure_count=1,
                last_error_type=error_type,
                suggested_next_action=_SUGGESTED_NEXT_ACTION[error_type],
            )
            # 再问 backend 有没有匹配的 promoted lesson
            if self.lesson_retriever is not None:
                lesson = self.lesson_retriever.try_recall(
                    tool_key=key[0], error_type=error_type
                )
                if lesson is not None:
                    return (
                        result + self.lesson_retriever.format_hint(lesson),
                        None,
                        lesson,
                    )
            return (result, None, None)

        # 2nd+ 次失败：更新状态并 augment
        entry.failure_count += 1
        entry.last_error_type = error_type
        entry.suggested_next_action = _SUGGESTED_NEXT_ACTION[error_type]
        hint = (
            f"\n\n[harness-recovery] 同参数已连续失败 {entry.failure_count} 次 "
            f"(last_error_type={entry.last_error_type})。"
            f"建议: {entry.suggested_next_action}"
        )
        if entry.suggested_next_action == "describe_script" and tool_name == "skill_exec":
            skill = kwargs.get("skill", "?")
            script = kwargs.get("script", "?")
            hint += f'（调用 describe_script(skill="{skill}", script="{script}") 先查 schema）'
        return (result + hint, entry, None)

    # ============================================================
    # 给 ReflexionStore 消费的出口
    # ============================================================

    def iter_records(self) -> Iterator[dict]:
        """把每条 entry yield 成普通 dict（直接 for-loop 消费）。

        shape 与 ReflexionRecord 对齐方便未来 `.to_reflexion_record(trace_id)` 一行转换：
          {tool_key, args_hash, failure_count, last_error_type, suggested_next_action}
        """
        for (tool_key, args_hash), entry in self.entries.items():
            yield {
                "tool_key": tool_key,
                "args_hash": args_hash,
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
