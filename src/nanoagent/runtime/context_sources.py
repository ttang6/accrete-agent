"""Context source 分类 + 内联 marker registry——把 messages 按"token 来源"
打标签，并集中托管框架注入到 LLM 上下文里的内联 marker 字符串。

两件事住一个文件的理由（B 与 D 是同一批片段的两个标签）：
- **marker registry（B）**：框架注入给模型读的角色/可信度标签（`[learned-*]`/
  `[gate-*]`/`[review-*]`）。集中成 `MARKER_*` Final 常量，所有 emit 点与 match
  点都 import 同一常量，杜绝"原地改字面量"导致的 drift（历史上踩过两次）。
- **source 分类（D）**：token 计数器要知道每条 message 的 token 花在哪。来源
  **派生自 marker 的命名空间前缀**（`[<ns>-...]` 的 ns），不再维护独立子串表。

命名轴 = **对模型的可信度**，不绑 emitter：
- **learned**：飞轮召回·概率性·可掂量（lesson / example / online-fix）。
- **gate**：框架硬要求·拦截·须照办（coverage / required-action / invalid-call /
  circuit-open）。
- **review**：判断·反馈·参考（quality / repeat-failure）。

为什么不做 Enum：trace_schema.py 已选 Final[str] 字面量风格，本文件保持一致；
字符串常量序列化进 trace 直接可读，不需 .value 解包。

下游消费者：
- TokenCounter.count_by_source(messages) → dict[source, tokens]
- MainLoop.finalize_trace 写入 run_summary.tokens.by_source
"""

from __future__ import annotations

import re
from typing import Final, Optional


# ============================================================
# Source 常量（字符串字面量，不是 Enum）
# ============================================================

SOURCE_SYSTEM: Final[str] = "system"
"""role=='system' 的消息——base_identity + skill body + user_facts + datetime。"""

SOURCE_HISTORY: Final[str] = "history"
"""role 为 user/assistant 的普通对话消息（去除特殊前缀的 hint 后的剩余）。"""

SOURCE_TOOL_OUTPUT: Final[str] = "tool_output"
"""role=='tool' 的消息——tool_call 返回结果（含 ToolOutputStore 落盘后的 preview）。"""

SOURCE_LEARNED: Final[str] = "learned"
"""飞轮召回注入的概率性经验（`[learned-*]`）：
  - [learned-lesson]  (backend 命中 promoted lesson)
  - [learned-example] (结构化修复示范)
  - [learned-fix]     (OnlineReflector 修复假设)
"""

SOURCE_GATE: Final[str] = "gate"
"""框架硬要求 / 拦截类注入（`[gate-*]`）：
  - [gate-coverage]        (CoverageChecker 覆盖不达标；v3-2b 后 synthetic 注入退役、保留观测)
  - [gate-required-action] (RequiredActionGate；v3-2b 整条退役，marker 保留词表占位)
  - [gate-invalid-call]    (RepairGate schema 校验拦下)
  - [gate-circuit-open]    (per-op 熔断)
"""

SOURCE_REVIEW: Final[str] = "review"
"""判断 / 反馈 / 参考类注入（`[review-*]`）：
  - [review-quality]        (Critic 质量反馈，原 DigestEvaluator)
  - [review-repeat-failure] (重复失败提示，原 harness-recovery)
"""


ALL_SOURCES: Final[frozenset[str]] = frozenset({
    SOURCE_SYSTEM,
    SOURCE_HISTORY,
    SOURCE_TOOL_OUTPUT,
    SOURCE_LEARNED,
    SOURCE_GATE,
    SOURCE_REVIEW,
})


# ============================================================
# Marker registry（B：框架注入给模型读的内联角色/可信度标签）
# ============================================================
#
# 顺序铁律：所有 emit 点 + 所有 match 点都 import 这些常量；改名时只改这里的
# 值，绝不在 emit/match 点原地改字面量。

# --- learned（飞轮召回·概率性）---
MARKER_LEARNED_LESSON: Final[str] = "[learned-lesson]"
MARKER_LEARNED_EXAMPLE: Final[str] = "[learned-example]"
MARKER_LEARNED_FIX: Final[str] = "[learned-fix]"
MARKER_LEARNED_HYPOTHESIS: Final[str] = "[learned-hypothesis]"

# --- gate（框架硬要求·拦截）---
MARKER_GATE_COVERAGE: Final[str] = "[gate-coverage]"
# REQUIRED_ACTION：RequiredActionGate 退役（v3-2b）后已无 emitter，保留占位。
MARKER_GATE_REQUIRED_ACTION: Final[str] = "[gate-required-action]"
MARKER_GATE_INVALID_CALL: Final[str] = "[gate-invalid-call]"
MARKER_GATE_CIRCUIT_OPEN: Final[str] = "[gate-circuit-open]"

# --- review（判断·反馈·参考）---
MARKER_REVIEW_QUALITY: Final[str] = "[review-quality]"
MARKER_REVIEW_REPEAT_FAILURE: Final[str] = "[review-repeat-failure]"


ALL_MARKERS: Final[frozenset[str]] = frozenset({
    MARKER_LEARNED_LESSON,
    MARKER_LEARNED_EXAMPLE,
    MARKER_LEARNED_FIX,
    MARKER_LEARNED_HYPOTHESIS,
    MARKER_GATE_COVERAGE,
    MARKER_GATE_REQUIRED_ACTION,
    MARKER_GATE_INVALID_CALL,
    MARKER_GATE_CIRCUIT_OPEN,
    MARKER_REVIEW_QUALITY,
    MARKER_REVIEW_REPEAT_FAILURE,
})


def is_known_marker(marker: str) -> bool:
    """外部 consumer 校验 marker 名是否在本 registry 里。"""
    return marker in ALL_MARKERS


# ============================================================
# 命名空间前缀 → source 的派生（D：不再维护独立子串表）
# ============================================================

# marker 命名空间（`[<ns>-<name>]` 的 ns）→ token source。
_NAMESPACE_SOURCE: Final[dict[str, str]] = {
    "learned": SOURCE_LEARNED,
    "gate": SOURCE_GATE,
    "review": SOURCE_REVIEW,
}

# 匹配开头的命名空间 marker `[learned-...]` / `[gate-...]` / `[review-...]`。
_NS_MARKER_RE: Final[re.Pattern] = re.compile(r"\[(learned|gate|review)-[a-z-]+\]")


def _hint_source_at(text: str) -> Optional[str]:
    """text 以已知命名空间 marker 开头时返回对应 source，否则 None。

    取代旧的两张子串表（_USER_HINT_PREFIXES / _TOOL_APPENDED_HINT_TAGS）：
    来源直接读 marker 的 ns 前缀，新增 marker 只要落在 learned/gate/review
    命名空间即自动归类，无需再维护表。
    """
    m = _NS_MARKER_RE.match(text)
    if m is None:
        return None
    return _NAMESPACE_SOURCE.get(m.group(1))


def split_tool_content_by_hint(content: str) -> list[tuple[str, str]]:
    """把 tool message 的 content 按追加的 hint 切成 [(source, text), ...]。

    用例：
        in:  "tool result text\n\n[learned-lesson] hint body"
        out: [(SOURCE_TOOL_OUTPUT, "tool result text"),
              (SOURCE_LEARNED, "\n\n[learned-lesson] hint body")]

    无 hint 标签时返回 `[(SOURCE_TOOL_OUTPUT, content)]` 单段。
    框架把 hint 以 `\n\n[<ns>-...]` 追加在原 tool 输出后面；这里找**最早**出现
    的命名空间 marker 作为切点，hint 段从分隔符连续到 content 末尾。
    （通用 `\n\n[<ns>-` 扫描——`[learned-fix]`(原 online-reflector) /
    `[gate-circuit-open]`(原熔断) 这类以前被算进 tool_output 的也会切出来归类。）
    """
    if not isinstance(content, str) or not content:
        return [(SOURCE_TOOL_OUTPUT, content or "")]

    m = re.search(r"\n\n(\[(?:learned|gate|review)-[a-z-]+\])", content)
    if m is None:
        return [(SOURCE_TOOL_OUTPUT, content)]

    idx = m.start()
    source = _hint_source_at(content[idx + 2:]) or SOURCE_TOOL_OUTPUT
    return [
        (SOURCE_TOOL_OUTPUT, content[:idx]),
        (source, content[idx:]),
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
       a. content 以已知命名空间 marker 开头 → 对应 source
       b. 否则 → SOURCE_HISTORY

    Args:
        msg: OpenAI 格式 dict（至少含 'role' 键，可选 'content' 键）。
            content 为非 str（None / list 多模态块）时按 SOURCE_HISTORY 处理。

    Returns:
        6 个 SOURCE_ 常量之一。未知 role 也归 SOURCE_HISTORY 兜底。
    """
    role = msg.get("role", "")
    if role == "system":
        return SOURCE_SYSTEM
    if role == "tool":
        return SOURCE_TOOL_OUTPUT

    # user / assistant：靠 content 的 marker 命名空间前缀识别 hint
    content = msg.get("content", "")
    if isinstance(content, str):
        # lstrip 容忍前置空白（虽然现有 hint 都不带前置 \n，保险起见）
        source = _hint_source_at(content.lstrip())
        if source is not None:
            return source

    return SOURCE_HISTORY


def is_known_source(source: str) -> bool:
    """外部 consumer 校验 source 名是否在本 schema 里。"""
    return source in ALL_SOURCES
