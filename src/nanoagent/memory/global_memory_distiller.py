"""GlobalMemoryDistiller：副 LLM 从对话蒸全局长期记忆（Dream-lite）。

对标 nanobot 的 Dream（MemoryConsolidator）：喂[当前长期记忆 + 一段新增对话] →
强制副 LLM **整块重写** → 覆盖写 GlobalMemory。

保守取舍（有意为之、跟 nanobot 同款）：
  - 无 eval / 无 confidence 门 / 无来源标（物理分文件即隔离）/ 无遗忘修剪；
  - 只在上下文压力下触发（撑爆 / 切换），**"填不满就不蒸"是 Dream 的固有性质**，
    一段没大到触发阈值就结束的会话不会被蒸——这是有出处、站得住的取舍，不是 bug。

触发与游标（last_distilled）在 Harness 一侧；本模块只管"给我一段新增对话，我蒸"。
"""

from typing import List, Optional

from nanoagent.core.logger import get_logger
from nanoagent.core.prompt_assets import load_prompt
from nanoagent.memory.global_memory import GlobalMemory
from nanoagent.runtime.subagent import SubAgentContextBuilder, SubAgentRunner

_logger = get_logger("global_memory_distiller")

_DISTILL_SYSTEM_PROMPT = """你是一个长期记忆整理助手。读「当前长期记忆」+「新增对话」，输出**整块更新后**的长期记忆（Markdown）。

# 只记什么
只记**关于这个用户本人、换个用户就会变**的全局事实与偏好：
- 身份 / 背景（角色、领域、技术栈、所在地、语言）
- 长期偏好（沟通风格、输出格式、口味、约束）
- 稳定的长期目标 / 项目背景

**不记**：一次性的任务细节、临时上下文、与"这个人是谁"无关的内容。

# 怎么写
- **保留当前长期记忆里的所有条目**，只增量补充 / 修正，绝不无故删除既有事实。
- 简洁的 Markdown 列表，按主题分组（如「身份」「偏好」「目标」）。
- 信息不足时宁可不写，不编造。
- 没有任何新的全局事实时，**原样返回当前长期记忆**。

直接输出整块 Markdown，不要任何前后缀说明。
"""


class GlobalMemoryDistiller:
    """喂一段新增对话 → 副 LLM 整块重写长期记忆 → 覆盖写。fail-open。"""

    def __init__(
        self,
        memory: GlobalMemory,
        runner: SubAgentRunner,
        *,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self._memory = memory
        self._runner = runner
        self._model = model
        self._provider = provider

    def distill(self, new_messages: List[dict]) -> bool:
        """喂[当前记忆 + 新增对话] → 整块重写 → 覆盖写。

        返回 True 表示写了新记忆；空对话 / 副 LLM 失败 / 空产出 → 返回 False 且不写。
        """
        transcript = _format_messages(new_messages)
        if not transcript.strip():
            return False
        current = self._memory.read()
        blocks = [
            ("当前长期记忆", current or "(空)"),
            ("新增对话", transcript),
        ]
        request = SubAgentContextBuilder.build(
            system_prompt=load_prompt("global_memory_distiller", _DISTILL_SYSTEM_PROMPT),
            blocks=blocks,
            allowed_tools=(),  # 纯蒸馏，零工具
            model=self._model,
            provider=self._provider,
            max_iterations=1,
        )
        result = self._runner.run(request)
        if not result.ok or not (result.text or "").strip():
            _logger.warning(f"[global-memory] distill 失败（已 fail-open）: {result.error}")
            return False
        self._memory.write(result.text.strip())
        return True


def _format_messages(messages: List[dict]) -> str:
    """history dict 列表 → `role: content` 纯文本 transcript（喂副 LLM）。"""
    lines: List[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        if not isinstance(content, str):
            content = str(content)
        content = content.strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)
