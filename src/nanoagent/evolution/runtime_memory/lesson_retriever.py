"""LessonRetriever — backend lesson 的极简检索接口。

只解决一件事：当一次 tool 失败发生，看看 backend 里有没有匹配的 promoted
lesson 立刻可用——这样不必等到第二次失败 FailureMemory 才能 augment。

设计取舍（减法）：
- 只一个查询入口 `try_recall(tool_key, error_type)`——按精确 (tool, error_class)
  匹配；不做语义检索 / 模糊匹配
- **标签桥接**：FailureMemory 的 schema_mismatch/transient/unknown 三档标签
  与 TraceErrorClassifier 的 5 类 error_class 不完全对齐——尤其
  classifier 把"同参数失败 ≥2 次"升级为 `repeated_same_args_failure`。
  retriever 内部用 `_LABEL_BRIDGE` 把 FailureMemory 标签 OR-展开到多个候选
  error_class，避免真实 pipeline 产出的 lesson 召不回。
- 默认只取 status=PROMOTED 的 lesson；candidate 不算数（避免未验证经验污染）
- TTL 过滤客户端做（filter grammar 没有 lt 运算符），跨 backend 行为一致
- 多条 hit 取 confidence 最高的 1 条；其他丢弃
- backend 异常吞掉返回 None（fail-open，不阻断 main loop）
- format_hint 对 recommendation 做硬上限（300 char），防止长 lesson 绕过
  MainLoop 的 max_tool_output_chars 截断
"""

from __future__ import annotations

import datetime
import json
from typing import Dict, List, Optional, Tuple

from nanoagent.core.logger import get_logger
from nanoagent.evolution.runtime_memory.backend import MemoryBackend
from nanoagent.evolution.runtime_memory.schema import LessonStatus, RuntimeLesson

_logger = get_logger("lesson_retriever")

# 单条 [runtime-lesson] 块的 recommendation 上限。超过会硬截断 + "..."
# 防止 lesson 绕过 main_loop 的 max_tool_output_chars 截断膨胀 context。
_HINT_MAX_RECOMMENDATION_CHARS = 300

# 标签桥接表：FailureMemory 标签（_classify_tool_failure 输出）
# → TraceErrorClassifier 的 error_class 候选集。
# 用于 try_recall 的 OR-匹配，覆盖真实 pipeline 产出的 lesson。
#
# 桥接逻辑：
# - schema_mismatch（FM）→ schema_mismatch + repeated_same_args_failure
#   理由：classifier 在同参数 ≥2 次失败时把 schema_mismatch 升级为 repeated 类，
#   但 FM 在第 1 次失败发生的瞬间分类只能是 schema_mismatch
# - transient（FM）→ tool_runtime_error + repeated_same_args_failure
#   理由：classifier 没有 transient 类；transient 错误归到
#   tool_runtime_error；同参数重复 transient 也走 repeated 类
# - unknown（FM）→ tool_runtime_error + repeated_same_args_failure
#   理由：兜底匹配；任何 tool 错误的兜底也是 tool_runtime_error
_LABEL_BRIDGE: Dict[str, Tuple[str, ...]] = {
    "schema_mismatch": ("schema_mismatch", "repeated_same_args_failure"),
    "transient": ("tool_runtime_error", "repeated_same_args_failure"),
    "unknown": ("tool_runtime_error", "repeated_same_args_failure"),
}


def _expand_error_type(error_type: str) -> List[str]:
    """把单个 FailureMemory 标签桥接成候选 error_class 列表。
    未注册的标签按原样返回（不做误展开）。"""
    bridged = _LABEL_BRIDGE.get(error_type)
    return list(bridged) if bridged else [error_type]


class LessonRetriever:
    """backend 上一层的薄包装。减法版只暴露一个方法。"""

    def __init__(self, backend: MemoryBackend):
        self._backend = backend

    def try_recall(
        self, tool_key: str, error_type: str
    ) -> Optional[RuntimeLesson]:
        """按 (tool_key, error_type) 找一条 promoted 且未过期的 lesson。

        - tool_key：与 RuntimeLesson.trigger.tool_name 精确比对
                    （e.g. "skill_exec:ai-digest/dup_check"）
        - error_type：FailureMemory 标签（schema_mismatch/transient/unknown）；
                    内部通过 `_LABEL_BRIDGE` OR-展开到多个候选 error_class，
                    保证真实 pipeline 产出的 lesson（如 repeated_same_args_failure）
                    在第 1 次失败时也能被召回。

        Returns:
            最佳匹配 lesson（按 confidence 降序取首条）；无匹配 / backend 异常 → None
        """
        if not tool_key or not error_type:
            return None

        # 标签桥接：把 FailureMemory 标签展开成候选 error_class 集合
        candidate_classes = _expand_error_type(error_type)
        # PROBATION 也参与召回（试用层），让试用 lesson 收到真实 helped/hurt
        # stats，PromotionGate 才能据此决定升 PROMOTED 或回 CANDIDATE。
        # confidence 排序时 PROMOTED 通常更高（已经被验证），自然优先。
        try:
            candidates = self._backend.search_lessons(
                filters={
                    "AND": [
                        {"status": {"in": [
                            LessonStatus.PROMOTED.value,
                            LessonStatus.PROBATION.value,
                        ]}},
                        {"trigger.tool_name": tool_key},
                        {
                            "OR": [
                                {"trigger.error_class": ec}
                                for ec in candidate_classes
                            ]
                        },
                    ]
                },
                limit=20,  # 同 (tool, error_class) 一般没几条；20 安全上限
            )
        except Exception as e:
            _logger.warning(
                f"[lesson_retriever] backend search failed (fail-open): {e}"
            )
            return None

        # 客户端 TTL 过滤（filter grammar 没 lt 运算符；保持行为跨 backend 一致）
        today = datetime.date.today().isoformat()
        fresh = [l for l in candidates if l.expires_on >= today]
        if not fresh:
            return None

        # 取 confidence 最高的；同分按 updated_at 新优先
        fresh.sort(
            key=lambda l: (l.confidence, l.updated_at),
            reverse=True,
        )
        return fresh[0]

    def format_hint(self, lesson: RuntimeLesson) -> str:
        """把 lesson 拼成可附加到 tool_result 末尾的 hint 字符串。

        格式与现有 [harness-recovery] 通道平行：
            [runtime-lesson] {recommendation} (lesson_id=xxx, conf=0.75)
            [structured-repair-hint] {json}     ← 可选

        recommendation 硬截断到 `_HINT_MAX_RECOMMENDATION_CHARS`（300 char），
        防止超长 lesson 绕过 MainLoop 的 max_tool_output_chars 把 context 撑爆。
        MainLoop 在 augment 之前已对 raw tool_result 截断；这里负责把追加部分
        也限制在受控范围内。

        lesson.suggested_action 非 None 时附加结构化修复块。
        只在失败现场召回时注入（FailureMemory.maybe_augment 路径），不进
        常驻 system prompt——避免长 JSON 污染 prompt cache。
        """
        rec = lesson.recommendation or ""
        if len(rec) > _HINT_MAX_RECOMMENDATION_CHARS:
            rec = rec[:_HINT_MAX_RECOMMENDATION_CHARS] + "...(截断)"
        parts = [
            f"\n\n[runtime-lesson] {rec} "
            f"(lesson_id={lesson.lesson_id}, conf={lesson.confidence:.2f})"
        ]
        if lesson.suggested_action is not None:
            try:
                action_json = json.dumps(
                    lesson.suggested_action, ensure_ascii=False, indent=2
                )
            except (TypeError, ValueError):
                action_json = ""
            if action_json:
                parts.append(f"\n[structured-repair-hint]\n{action_json}")
        return "".join(parts)
