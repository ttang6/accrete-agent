"""Critic —— 发布流程里的质量评审子 agent（替代旧 DigestEvaluator）。

与旧 evaluator 的两点本质区别，都是这轮讨论定下的：
- **返回自然语言批评，不返回可执行枚举**。旧 evaluator 让副 LLM 出
  `recommended_action=fetch_hf` 这种枚举、还兼着"数够不够 3 篇"——那是把确定性的
  覆盖计数交给概率裁判（覆盖度已交确定性的 CoverageChecker / body 方法论）。critic
  只判"质量"（写得糊 / 重复 / 失衡 / 跑偏），用散文说哪不合格，主 LLM 自己据此修。
- **非阻断**。critic 尽力改（最多触发一轮 revise），但判不合格**也照常发布**——
  verdict 只记录、不 gate。它是 reflection 的"挑错"半，不是收尾硬闸。

隔离 + 用完即弃：单次 think 调用、不带工具、不写 lesson、不落 trajectory、不进飞轮。
fail-open：任何异常 → 当作合格（passed=True、空批评），不阻塞发布。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

from nanoagent.core.logger import get_logger
from nanoagent.core.prompt_assets import load_prompt

_logger = get_logger("critic")

# 批评文本上限，防超长批评把 revise 轮的 context 撑大。
_MAX_CRITIQUE_CHARS: Final[int] = 800

_CRITIC_SYSTEM_PROMPT: Final[str] = """你是日报质量评审副助手。通读下面这份已生成的日报，只判**质量**，不数数量。

判什么（质量维度）：
- 条目是否重复啰嗦、来源是否过于单一、某段是否明显失衡跑偏
- 中文事实描述是否清楚、有没有空话/套话/夸大
- 标题/链接格式是否规范

不判什么：覆盖够不够篇数（那由别处确定性核对，不归你）。

输出要求：
- 若整体合格、没有值得返工的硬伤 → **只回两个字：合格**
- 否则 → 逐条列出具体硬伤（每条一行，指明在哪、问题是什么），不要写"可以更全面"这类空话"""


@dataclass
class CriticVerdict:
    """review 返回结构。passed=True 表示无需返工；critique 是给主 LLM 看的批评文本。"""

    passed: bool
    critique: str = ""


class Critic:
    """副 LLM 质量评审。stateless，每次 review 独立、单次 think 调用。

    用法（装配见 main.py）：
        critic = Critic(llm=critic_llm)
        verdict = critic.review(digest_text)
        if not verdict.passed:
            ... 注入 verdict.critique，主 LLM revise 一轮（封顶），然后照常发布 ...
    """

    def __init__(self, llm, system_prompt: Optional[str] = None):
        self._llm = llm
        self._system_prompt = system_prompt or load_prompt("critic", _CRITIC_SYSTEM_PROMPT)

    def review(self, answer: str) -> CriticVerdict:
        """评审日报，返回 NL 批评。任何异常 → passed=True（fail-open，不阻塞发布）。"""
        try:
            messages = [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": f"待评审日报：\n\n---\n{answer}\n---"},
            ]
            text = (self._llm.think(messages=messages, temperature=0) or "").strip()
        except Exception as e:
            _logger.warning(f"[critic] review 异常（已 fail-open 当合格）: {type(e).__name__}: {e}")
            return CriticVerdict(passed=True)

        # 合格判定宽松：以"合格"开头且没多少别的内容即算过；否则当批评。
        # 非阻断，判松判紧最坏只是多/少一轮 revise，都照常发布，无害。
        if text.startswith("合格") and len(text) <= 8:
            return CriticVerdict(passed=True)
        return CriticVerdict(passed=False, critique=text[:_MAX_CRITIQUE_CHARS])
