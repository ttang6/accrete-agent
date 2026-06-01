"""Context source 分类——把 messages 按"token 来源"打标签。

Context Hygiene Foundation 的最小契约层：6 个字符串常量 + 1 个分类函数。
对齐 `core/trace_schema.py` 风格（Final[str] 常量 + frozenset 聚合 + is_known
helper），不引 Enum——保持和现有 trace_schema action 的字面量风格统一。

入门视角：
- 每条 message 在传给 LLM 时都贡献 token；不分类就不知道 token 都花在哪
- 6 个 source 覆盖现有所有注入路径（system / history / tool / 4 类 hint）
- runtime hint / lesson / evaluator hint 都是 user-role 消息但内容前缀不同，
  靠前缀模式区分，匹配 main_loop.py 和 harness.py 里现有的字面量

为什么不做 Enum：
- trace_schema.py 已经选了 Final[str] 字面量风格，本文件保持一致
- 字符串常量序列化进 trace 文件直接可读，不需 .value 解包
- 后续如果要加新 source 直接加常量 + 改 classify_message，无 enum 迁移负担

下游消费者：
- TokenCounter.count_by_source(messages) → dict[source, tokens]
- MainLoop.finalize_trace 写入 run_summary.tokens.by_source
"""

from __future__ import annotations

from typing import Final


# ============================================================
# Source 常量（字符串字面量，不是 Enum）
# ============================================================

SOURCE_SYSTEM: Final[str] = "system"
"""role=='system' 的消息——base_identity + skill body + user_facts + datetime。"""

SOURCE_HISTORY: Final[str] = "history"
"""role 为 user/assistant 的普通对话消息（去除特殊前缀的 hint 后的剩余）。"""

SOURCE_TOOL_OUTPUT: Final[str] = "tool_output"
"""role=='tool' 的消息——tool_call 返回结果（含 ToolOutputStore 落盘后的 preview）。"""

SOURCE_RUNTIME_HINT: Final[str] = "runtime_hint"
"""role=='user' 但被 harness 注入的硬保护 hint：
  - [harness-coverage] (CoverageChecker 覆盖不达标)
  - [harness-recovery] (FailureMemory 同参重复失败)
"""

SOURCE_LESSON: Final[str] = "lesson"
"""role=='user' 但被 LessonRetriever 注入的跨 trace 经验：
  - [runtime-lesson] (第 1 次失败时 backend 命中 promoted lesson)
"""

SOURCE_EVALUATOR_HINT: Final[str] = "evaluator_hint"
"""role=='user' 但被 Harness DigestEvaluator 注入的副链路 LLM 反馈：
  - [evaluator] (evaluator retry 触发的 hint)
"""


ALL_SOURCES: Final[frozenset[str]] = frozenset({
    SOURCE_SYSTEM,
    SOURCE_HISTORY,
    SOURCE_TOOL_OUTPUT,
    SOURCE_RUNTIME_HINT,
    SOURCE_LESSON,
    SOURCE_EVALUATOR_HINT,
})


# ============================================================
# 内容前缀 → source 的映射
# ============================================================

# user-role hint 前缀 → source。匹配 main_loop.py 的 coverage hint（user msg）
# 和 harness.py 的 evaluator hint（user msg）。
# Note: [harness-recovery] / [runtime-lesson] 同 prefix 也可能 *追加在 tool message
# 末尾*（failure_memory.py:214, lesson_retriever.py:140），见 _TOOL_APPENDED_HINT_TAGS。
_USER_HINT_PREFIXES: Final[tuple[tuple[str, str], ...]] = (
    ("[harness-coverage]", SOURCE_RUNTIME_HINT),
    ("[harness-recovery]", SOURCE_RUNTIME_HINT),
    ("[runtime-lesson]", SOURCE_LESSON),
    ("[evaluator]", SOURCE_EVALUATOR_HINT),
)


# 追加到 tool message 末尾的 hint 标签（不是 user-role）——failure_memory 和
# lesson_retriever 用 `\n\n[<tag>] ...` 格式 append 在原 tool 输出后面。
# count_by_source 用此表把 tool content 切成 "原 tool 部分 + hint 部分"，
# 避免 lesson / recovery 的 token 全被算进 SOURCE_TOOL_OUTPUT。
# 顺序：lesson 在前——同条 message 不会同时出现两种 hint（failure_memory
# 单失败一种 hint 的契约），但稳健起见仍按出现先后做单次切分。
_TOOL_APPENDED_HINT_TAGS: Final[tuple[tuple[str, str], ...]] = (
    ("\n\n[runtime-lesson]", SOURCE_LESSON),
    ("\n\n[harness-recovery]", SOURCE_RUNTIME_HINT),
)


def split_tool_content_by_hint(content: str) -> list[tuple[str, str]]:
    """把 tool message 的 content 按追加的 hint 切成 [(source, text), ...]。

    用例：
        in:  "tool result text\n\n[runtime-lesson] hint body"
        out: [(SOURCE_TOOL_OUTPUT, "tool result text"),
              (SOURCE_LESSON, "\n\n[runtime-lesson] hint body")]

    无 hint 标签时返回 `[(SOURCE_TOOL_OUTPUT, content)]` 单段。
    多种 hint 同时存在（理论上不会，但稳健处理）：取**最早**出现的 hint 作为
    切点——保证 hint 段从分隔符开始连续到 content 末尾。
    """
    if not isinstance(content, str) or not content:
        return [(SOURCE_TOOL_OUTPUT, content or "")]

    earliest_idx = -1
    earliest_source = SOURCE_TOOL_OUTPUT
    for tag, source in _TOOL_APPENDED_HINT_TAGS:
        idx = content.find(tag)
        if idx != -1 and (earliest_idx == -1 or idx < earliest_idx):
            earliest_idx = idx
            earliest_source = source

    if earliest_idx == -1:
        return [(SOURCE_TOOL_OUTPUT, content)]

    return [
        (SOURCE_TOOL_OUTPUT, content[:earliest_idx]),
        (earliest_source, content[earliest_idx:]),
    ]


# ============================================================
# 分类函数
# ============================================================


def classify_message(msg: dict) -> str:
    """把一条 OpenAI 格式 dict 消息分类到 6 个 source 之一。

    分类顺序（短路）：
    1. role=='system' → SOURCE_SYSTEM
    2. role=='tool' → SOURCE_TOOL_OUTPUT
    3. role in ('user', 'assistant')：
       a. content 以已知 hint 前缀开头 → 对应 source
       b. 否则 → SOURCE_HISTORY

    Args:
        msg: OpenAI 格式 dict（至少含 'role' 键，可选 'content' 键）。
            content 为非 str（None / list 多模态块）时按 SOURCE_HISTORY 处理——
            非文本块的 token 仍归 history（避免误分类）。

    Returns:
        6 个 SOURCE_ 常量之一。未知 role 也归 SOURCE_HISTORY 兜底。
    """
    role = msg.get("role", "")
    if role == "system":
        return SOURCE_SYSTEM
    if role == "tool":
        return SOURCE_TOOL_OUTPUT

    # user / assistant：靠 content 前缀识别 hint
    content = msg.get("content", "")
    if isinstance(content, str):
        # strip 容忍前置空白（虽然现有 hint 都不带前置 \n，保险起见）
        head = content.lstrip()
        for prefix, source in _USER_HINT_PREFIXES:
            if head.startswith(prefix):
                return source

    return SOURCE_HISTORY


def is_known_source(source: str) -> bool:
    """外部 consumer 校验 source 名是否在本 schema 里。"""
    return source in ALL_SOURCES
