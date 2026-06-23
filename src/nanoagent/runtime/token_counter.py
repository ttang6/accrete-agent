"""TokenCounter——tiktoken 计量 + chars/4 fallback。

替代 MainLoop 内联的 `chars // 4` 粗估（main_loop.py:217-223），让
`max_context_tokens` 警戒线真实可用。

入门视角：
- 优先用 tiktoken cl100k_base（OpenAI 4o / qwen / Claude 都近似可用，偏离 5-15%）
- tiktoken 加载失败 / encode 异常时 fallback 到 chars/4，单 run 内只 logger.warning
  一次（避免日志噪音），通过 `is_fallback` 暴露状态
- per-message overhead：固定 +4 token（消息框架开销，参考 nanobot helpers.py:324
  和 OpenAI cookbook 的 chat 消息估算公式）
- tools schema serialize 成 JSON 计入（tool_choice='auto' 模式下 tools 也吃 token）

为什么 cl100k_base 不是 o200k_base：
- cl100k 覆盖 OpenAI 老模型 + Anthropic + qwen + 多数 OSS 模型
- o200k 是 4o 专用，对其他模型偏离更大
- 简历项目用 qwen-plus（dashscope）+ openai/* 兼容，cl100k 是更稳的默认

不做的事：
- 不做 provider-perfect 计数（nanobot 有 estimate_prompt_tokens_chain 用 provider
  counter，那是 ~30 行额外复杂度，本项目暂不需要）
- 不做模型→encoding 自动映射（caller 显式传 encoding_name 即可）
- 不持有 RunTracer（warn 走 module logger，不污染 trace 字段）

下游：
- MainLoop._run_inner 每 iter 头估算 context tokens（替换 chars/4）
- ContextBudget 用 count_text 估单 tool output 是否超 max_single_tokens
- ToolOutputStore 用 count_text 估 preview/full 文件 token 数
- finalize_trace 用 count_by_source 写 run_summary.tokens.by_source
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from nanoagent.runtime.context_sources import (
    ALL_SOURCES,
    SOURCE_HISTORY,
    SOURCE_TOOL_OUTPUT,
    classify_message,
    split_tool_content_by_hint,
)

_logger = logging.getLogger("nanoagent.token_counter")


# 单消息 token overhead——OpenAI chat 模型每条消息固定加 ~4 token
# (role/name/content 框架开销)。和 nanobot utils/helpers.py:324 对齐。
_PER_MESSAGE_OVERHEAD = 4


class TokenCounter:
    """tiktoken 计量器，附带 chars/4 fallback。

    线程安全：tiktoken 编码器本身线程安全（内部 BPE 表只读）。本类无可变
    状态除了 _fallback_warned（race condition 至多多打几条 warning，不致命）。

    生命周期：可被多个 MainLoop / ToolOutputStore / ContextBudget 共享同一
    实例（推荐 main.py 装配区构造一份共用），tiktoken cache 自然命中。
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        self._encoding_name = encoding_name
        self._enc = self._try_load(encoding_name)
        self._fallback_warned = False

    @staticmethod
    def _try_load(encoding_name: str):
        """尝试加载 tiktoken 编码器，失败返回 None（caller 会 fallback）。"""
        try:
            import tiktoken
            return tiktoken.get_encoding(encoding_name)
        except Exception as e:
            _logger.warning(
                f"tiktoken 加载失败（encoding={encoding_name}, err={type(e).__name__}），"
                f"将退化到 chars/4 估算"
            )
            return None

    @property
    def is_fallback(self) -> bool:
        """tiktoken 不可用返回 True；caller 可据此判断是否要在 trace 标注。"""
        return self._enc is None

    def count_text(self, text: str) -> int:
        """单段文本的 token 数。空字符串返回 0。"""
        if not text:
            return 0
        if self._enc is not None:
            try:
                return len(self._enc.encode(text))
            except Exception:
                self._warn_fallback_once("encode 异常")
                # 单条 encode 失败 → 整个 encoder 不可信，降级到永久 fallback
                # 与 _warn_fallback_once 文案"后续退化到 chars/4"保持一致
                self._enc = None
                return self._fallback_count(text)
        return self._fallback_count(text)

    def count_messages(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
    ) -> int:
        """消息列表的总 token 数（含 tools schema 序列化后的 token）。

        计入：
        - 每条消息的 content（str / list[block] 的 text 块）
        - tool_calls / tool_call_id / name / reasoning_content（若有）
        - 每条消息 +4 token framing overhead
        - tools 列表（serialize JSON）一次性加在末尾

        排除：
        - role 字段本身（已在 +4 overhead 里）
        - dict 中 token 不可见的元字段（id / type 等）

        参考：nanobot utils/helpers.py:285 estimate_prompt_tokens 的同款实现。
        """
        if not messages:
            return self.count_text(json.dumps(tools, ensure_ascii=False)) if tools else 0

        total = 0
        for msg in messages:
            total += self._count_one_message(msg) + _PER_MESSAGE_OVERHEAD

        if tools:
            try:
                tools_json = json.dumps(tools, ensure_ascii=False)
                total += self.count_text(tools_json)
            except (TypeError, ValueError):
                # tools schema 含不可序列化对象（罕见）——保守加估算
                total += sum(self.count_text(str(t)) for t in tools)

        return total

    def count_by_source(self, messages: list[dict]) -> dict[str, int]:
        """按 context_sources.classify_message 分组求和。

        Returns:
            dict[source_name, tokens]，6 个 SOURCE_ 全部出现（无消息的 source
            归 0），便于 trace 和报告对齐结构。**包含 +4 per-message overhead**——
            这样 sum(by_source.values()) 与 count_messages(messages, tools=None)
            一致（不算 tools 部分，tools 在 count_messages 里单独算）。

        特殊处理 role=='tool'：
            lesson_retriever / online_reflector 等把 [learned-*] hint 追加到
            tool message 末尾，如果直接按 role 归 SOURCE_TOOL_OUTPUT，learned /
            gate / review 的 token 成本会在 by_source 报告里失踪。
            split_tool_content_by_hint() 按追加 hint 的命名空间前缀把 tool content
            切成 "原 tool 部分 + hint 部分"，分别归到 TOOL_OUTPUT / learned / gate /
            review。+4 framing overhead 留给 SOURCE_TOOL_OUTPUT 段（每条 message
            只有一份 framing，归到主 source 是合理的）。
        """
        result: dict[str, int] = {s: 0 for s in ALL_SOURCES}
        for msg in messages:
            role = msg.get("role", "")
            if role == "tool":
                # 特殊处理：可能在末尾追加了 hint 段
                content = msg.get("content", "")
                if isinstance(content, str):
                    segments = split_tool_content_by_hint(content)
                else:
                    # list/None content（多模态 / 空）—— 整条按 SOURCE_TOOL_OUTPUT
                    segments = [(SOURCE_TOOL_OUTPUT, "")]

                # 其他字段（tool_call_id / name / 等）一并放在 SOURCE_TOOL_OUTPUT
                # 段中处理：构造一个简化 msg 走 _count_one_message 拿到 framing 部分
                # 之外的非 content 字段开销
                non_content_msg = {k: v for k, v in msg.items() if k != "content"}
                non_content_tokens = self._count_one_message(
                    {**non_content_msg, "content": ""}
                )

                for src, segment in segments:
                    seg_tokens = self.count_text(segment)
                    if src == SOURCE_TOOL_OUTPUT:
                        # 主段承接 +4 framing + 非 content 字段开销
                        result[src] += seg_tokens + non_content_tokens + _PER_MESSAGE_OVERHEAD
                    else:
                        result[src] += seg_tokens
                continue

            # 非 tool message：按 classify_message 整体归类
            source = classify_message(msg)
            tokens = self._count_one_message(msg) + _PER_MESSAGE_OVERHEAD
            # classify_message 兜底返回 SOURCE_HISTORY，理论上 source 必在 ALL_SOURCES
            result[source if source in result else SOURCE_HISTORY] += tokens
        return result

    # ============================================================
    # 内部
    # ============================================================

    def _count_one_message(self, msg: dict) -> int:
        """单条消息的"内容"token（不含 +4 framing overhead）。

        和 nanobot estimate_message_tokens 同款拆分：
        - content 是 str：直接 count
        - content 是 list[block]：text 块 .text 计入；非 text 块 JSON dump 兜底
        - tool_calls / reasoning_content / name / tool_call_id 都计入
        """
        parts: list[str] = []
        content = msg.get("content")
        if isinstance(content, str):
            if content:
                parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text = block.get("text", "")
                        if text:
                            parts.append(text)
                    else:
                        # 非 text 块（image_url 等）—— JSON 序列化兜底
                        try:
                            parts.append(json.dumps(block, ensure_ascii=False))
                        except (TypeError, ValueError):
                            parts.append(str(block))

        # 其他字段
        tc = msg.get("tool_calls")
        if tc:
            try:
                parts.append(json.dumps(tc, ensure_ascii=False))
            except (TypeError, ValueError):
                parts.append(str(tc))

        rc = msg.get("reasoning_content")
        if isinstance(rc, str) and rc:
            parts.append(rc)

        for key in ("name", "tool_call_id"):
            value = msg.get(key)
            if isinstance(value, str) and value:
                parts.append(value)

        if not parts:
            return 0
        return self.count_text("\n".join(parts))

    def _fallback_count(self, text: str) -> int:
        """chars/4 退化估算。中文每字 ≈ 1 token，英文 ≈ 0.25 token——chars/4
        对中文严重低估、对英文略高估。但作为 fallback 已够"比没有好"。"""
        return max(1, len(text) // 4)

    def _warn_fallback_once(self, reason: str) -> None:
        """单实例只 warn 一次 encode 异常，避免 log 刷屏。"""
        if not self._fallback_warned:
            self._fallback_warned = True
            _logger.warning(
                f"TokenCounter 单条 encode 失败（{reason}），后续退化到 chars/4。"
            )
