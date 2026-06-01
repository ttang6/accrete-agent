"""DigestEvaluator — 结构化 Evaluator-Optimizer 的副 LLM 实现。

作用：主 LLM 生成日报后，用副 LLM（建议 qwen-plus via dashscope）走
function calling schema 判定是否需要再跑一轮补齐。输出直接映射下一步
action（不是描述性建议），Harness 按 recommended_action 决定是否 retry。

设计要点：
- function calling schema（非 prompt 级 JSON）—— 4o-mini-class 弱模型
  对 schema 级约束 adherence 远高于 prompt
- `recommended_action` enum 直接映射到可执行 tool name，而非 "建议多查论文" 文本
- Fail-open：任何异常（dashscope down / schema 解析失败 / 超时）→ 返回
  `recommended_action="finalize"`，不阻塞用户
- 不持久化状态，stateless 每次 evaluate 独立调用

evaluator 必须输出可执行 action，不是建议文本——否则
`issues: '论文内容不足'` 这种泛话闭环就弱了。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Final, Optional


EVALUATE_DIGEST_SCHEMA: Final[dict] = {
    "type": "function",
    "function": {
        "name": "evaluate_digest",
        "description": (
            "审查刚生成的 AI 日报，判断覆盖度 / 质量问题 / 下一步动作。"
            "recommended_action 必填，直接映射到下一步 tool：fetch_hf / "
            "fetch_github / fetch_rss / finalize。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "coverage_ok": {
                    "type": "boolean",
                    "description": "三维度（论文/开源/行业动态）覆盖是否都达标",
                },
                "missing": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["paper", "oss", "news"]},
                    "description": "缺失的维度，每项对应应补的类别",
                },
                "soft_issues": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["redundancy_high", "source_single", "topic_imbalance"],
                    },
                    "description": (
                        "软质量问题：redundancy_high=条目重复过多 / "
                        "source_single=来源单一 / topic_imbalance=热点失衡"
                    ),
                },
                "recommended_action": {
                    "type": "string",
                    "enum": ["fetch_hf", "fetch_github", "fetch_rss", "finalize"],
                    "description": (
                        "下一步动作。finalize=接受当前日报；其他 enum 值直接"
                        "对应 ai-digest 的 fetch_* script 名（Harness 会注入 hint 让主 LLM 调）"
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "简短判定理由，≤200 字",
                },
            },
            "required": ["coverage_ok", "recommended_action"],
        },
    },
}


_EVALUATOR_SYSTEM_PROMPT: Final[str] = """你是 AI 日报质量评审副助手。

任务：审查主助手生成的 AI 日报草稿，判断是否需要主助手补充后再发布。

判定维度：
1. **硬覆盖**：三段是否齐全（论文速览 ≥ 3 篇 / 开源与工程 ≥ 2 条 / 行业动态 ≥ 1 条）
2. **软质量**：条目重复、来源单一、某段明显不成比例

输出要求：
- 必须调用 `evaluate_digest` function（不要直接回复文字）
- `recommended_action` 必须是可执行的下一步：
  - `fetch_hf` / `fetch_github` / `fetch_rss` 三选一（对应硬覆盖缺的维度）
  - `finalize` 当日报已足够好时
- 如果三段都齐但软质量一般，倾向 `finalize`——不追求完美
- 如果某段为空或严重不足，优先选对应 fetch_*

禁止：写"可以更全面"之类的泛话。必须落到具体的 recommended_action。"""


@dataclass
class EvaluatorDecision:
    """evaluate() 返回结构。Harness 直接消费 recommended_action 决定 retry / finalize。"""

    coverage_ok: bool
    recommended_action: str  # enum: fetch_hf / fetch_github / fetch_rss / finalize
    missing: list[str] = field(default_factory=list)
    soft_issues: list[str] = field(default_factory=list)
    reason: str = ""
    fail_open: bool = False  # True = 异常降级，caller 可据此记日志

    def should_retry(self) -> bool:
        """是否应触发 Harness 层的 retry."""
        return self.recommended_action != "finalize" and not self.fail_open


class DigestEvaluator:
    """副 LLM 评审日报草稿。stateless，每次 evaluate 独立。

    用法：
        eval_llm = LLMClient(model="qwen-plus", provider="dashscope", timeout=15)
        evaluator = DigestEvaluator(llm=eval_llm)
        decision = evaluator.evaluate(digest_text, history)
        if decision.should_retry():
            ... inject hint, rerun main loop ...
    """

    def __init__(self, llm, system_prompt: Optional[str] = None):
        self._llm = llm
        self._system_prompt = system_prompt or _EVALUATOR_SYSTEM_PROMPT

    def evaluate(self, answer: str, history: Optional[list[dict]] = None) -> EvaluatorDecision:
        """评审日报 answer，返回结构化 decision。

        Args:
            answer: 主 LLM 刚生成的日报正文
            history: 主循环对话历史（可选，用于 evaluator 看 fetch 结果等上下文）

        Returns:
            EvaluatorDecision。任何异常路径 → fail_open=True, recommended_action=finalize。
        """
        try:
            return self._evaluate_impl(answer, history or [])
        except Exception as e:
            return EvaluatorDecision(
                coverage_ok=True,
                recommended_action="finalize",
                reason=f"evaluator_unavailable: {type(e).__name__}",
                fail_open=True,
            )

    def _evaluate_impl(self, answer: str, history: list[dict]) -> EvaluatorDecision:
        """核心评审逻辑。异常向上传给 evaluate() 的 try/except。"""
        # 构造 evaluator 看到的 messages：system + 精简 context + user(答案)
        # history 传给 evaluator 做参考，但不塞入整个主循环历史（会超 context）。
        # 简化：只拼一条 user 消息，含日报原文。若 history 非空，末尾附简短摘要。
        user_content = f"请评审以下 AI 日报草稿：\n\n---\n{answer}\n---"
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        response = self._llm.think_with_tools(
            messages=messages,
            tools=[EVALUATE_DIGEST_SCHEMA],
            temperature=0,
        )

        # 解析 tool_calls。若 evaluator 没调 function（返回纯文本）→ 视为无决断 → finalize
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return EvaluatorDecision(
                coverage_ok=True,
                recommended_action="finalize",
                reason="evaluator_no_tool_call",
                fail_open=True,
            )

        # 只取第一个 tool_call（应该只调一次 evaluate_digest）
        tc = tool_calls[0]
        if tc.function.name != "evaluate_digest":
            return EvaluatorDecision(
                coverage_ok=True,
                recommended_action="finalize",
                reason=f"unexpected_tool: {tc.function.name}",
                fail_open=True,
            )

        args = json.loads(tc.function.arguments)
        return EvaluatorDecision(
            coverage_ok=bool(args.get("coverage_ok", True)),
            recommended_action=args.get("recommended_action", "finalize"),
            missing=list(args.get("missing", [])),
            soft_issues=list(args.get("soft_issues", [])),
            reason=str(args.get("reason", ""))[:200],
        )
