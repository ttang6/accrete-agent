"""LessonRetriever — backend lesson 的极简检索接口。

只解决一件事：当一次 tool 失败发生，看看 backend 里有没有匹配的 promoted
lesson 立刻可用——这样不必等到第二次失败 FailureMemory 才能 augment。

设计取舍（减法）：
- 只一个查询入口 `try_recall(op, error_type)`——按精确 (tool, failure_class)
  匹配；不做语义检索 / 模糊匹配
- **桥已拆**（键重构·刀3·3c）:热路径 `error_type`（base-3:schema_mismatch/
  transient/unknown）与入库的 `trigger.failure_class` 现在同一套标签,直接精确匹配、
  不再 OR-展开。旧 `_LABEL_BRIDGE` 是"键含召回时算不出的信息"（repeated 要计数）
  的补偿器,键收敛到首次可观测的 base-3 后自然失业。
- 默认只取 status=PROMOTED 的 lesson；candidate 不算数（避免未验证经验污染）
- 多条 hit 取净有效次数（helped_count - hurt_count）最高的 1 条；其他丢弃
- backend 异常吞掉返回 None（fail-open，不阻断 main loop）
- format_hint 对 recommendation 做硬上限（300 char），防止长 lesson 绕过
  MainLoop 的 max_tool_output_chars 截断
"""

from __future__ import annotations

import json
from typing import Optional

from accrete.core.logger import get_logger
from accrete.evolution.runtime_memory.backend import MemoryBackend
from accrete.evolution.runtime_memory.lesson_generator import _condition_hint
from accrete.evolution.runtime_memory.lesson_score import compute_score, is_active
from accrete.evolution.runtime_memory.schema import RuntimeLesson
from accrete.runtime.context_sources import (
    MARKER_LEARNED_EXAMPLE,
    MARKER_LEARNED_LESSON,
)

_logger = get_logger("lesson_retriever")

# 单条 [learned-lesson] 块的 recommendation 上限。超过会硬截断 + "..."
# 防止 lesson 绕过 main_loop 的 max_tool_output_chars 截断膨胀 context。
_HINT_MAX_RECOMMENDATION_CHARS = 300


class LessonRetriever:
    """backend 上一层的薄包装。减法版只暴露一个方法。"""

    def __init__(self, backend: MemoryBackend):
        self._backend = backend

    def try_recall(
        self, op: str, error_type: str
    ) -> Optional[RuntimeLesson]:
        """按 (op, error_type) 找一条 promoted 且未过期的 lesson。

        - op：与 RuntimeLesson.trigger.op 精确比对
                    （e.g. "skill_exec:ai-digest/dup_check"）
        - error_type：热路径 base 标签（schema_mismatch/transient/unknown）,与入库的
                    `trigger.failure_class` 同一套标签,直接精确匹配（桥已拆,见模块 docstring）。

        Returns:
            最佳匹配 lesson（active 中按 score 降序取首条）；无匹配 / backend 异常 → None
        """
        if not op or not error_type:
            return None

        # 刀4:注入闸从"status in {PROMOTED,PROBATION}"换成"score ≥ T"。按 (op,
        # failure_class) 取全部候选（不再按 status 预切——status 已无存储态），
        # 客户端算派生分、留 active（score≥T）、按 score 降序。候选集本就极小（同
        # (op,class) 没几条,limit=20 兜底）,客户端过滤零成本。
        try:
            candidates = self._backend.search_lessons(
                filters={
                    "AND": [
                        {"trigger.op": op},
                        {"trigger.failure_class": error_type},
                    ]
                },
                limit=20,
            )
        except Exception as e:
            _logger.warning(
                f"[lesson_retriever] backend search failed (fail-open): {e}"
            )
            return None

        active = [l for l in candidates if is_active(l)]
        if not active:
            return None

        # score 已含净有效次数 + 次线性复活项；同分按 updated_at 新优先
        active.sort(key=lambda l: (compute_score(l), l.updated_at), reverse=True)
        return active[0]

    def format_hint(self, lesson: RuntimeLesson) -> str:
        """把 lesson 拼成可附加到 tool_result 末尾的 hint 字符串。

        格式：
            [learned-lesson] {advice} (lesson_id=xxx)
            [适用条件] {现渲染：适用于 X 的 Y 类失败（特征 Z）}     ← 基本恒有
            [learned-example]（示例标注） {json}     ← 可选

        advice 硬截断到 `_HINT_MAX_RECOMMENDATION_CHARS`（300 char），防止超长
        lesson 绕过 MainLoop 的 max_tool_output_chars 把 context 撑爆。

        [适用条件] 行从 trigger 的 failure_class+op+failure_reason **现渲染**
        （condition_hint 字段已删——它 100% 派生自这三个输入，不再存储）。

        lesson.example 非 None 时附加结构化修复块。示例来自历史失败现场，参数值
        是当时任务的——标注语提醒 LLM 结构照用、值按当前任务替换（防照抄旧实例值
        帮倒忙）。只在失败现场召回时注入（FailureMemory.maybe_augment 路径），不进
        常驻 system prompt——避免长 JSON 污染 prompt cache。
        """
        rec = lesson.advice or ""
        if len(rec) > _HINT_MAX_RECOMMENDATION_CHARS:
            rec = rec[:_HINT_MAX_RECOMMENDATION_CHARS] + "...(截断)"
        parts = [
            f"\n\n{MARKER_LEARNED_LESSON} {rec} "
            f"(lesson_id={lesson.lesson_id})"
        ]
        condition = _condition_hint(
            lesson.trigger.op, lesson.trigger.failure_class,
            lesson.trigger.failure_reason,
        ).strip()
        if condition:
            parts.append(f"\n[适用条件] {condition[:200]}")
        if lesson.example is not None:
            try:
                action_json = json.dumps(
                    lesson.example, ensure_ascii=False, indent=2
                )
            except (TypeError, ValueError):
                action_json = ""
            if action_json:
                parts.append(
                    f"\n{MARKER_LEARNED_EXAMPLE}"
                    "（历史同类失败的修复示例：结构照用，具体参数值按当前任务替换）"
                    f"\n{action_json}"
                )
        return "".join(parts)
