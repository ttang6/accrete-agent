"""五段上下文组装与保守压缩。"""

from dataclasses import dataclass

from .prompts import COMPACT_SUMMARIZER_PROMPT, DEFAULT_COMPACT_PROMPT
from .provider import LLMProvider
from .state import AgentState
from .types import Message, Usage


@dataclass
class CompactResult:
    """一次压缩的产出：新历史，以及调用方记账和留痕所需的全部信息。"""

    history: list[Message]
    usage: Usage
    degraded: bool
    before_tokens: int
    after_tokens: int


class ContextManager:
    """构建模型可见上下文，保护系统提示词与任务陈述。"""

    def __init__(self, system_prompt: str, compact_threshold_tokens: int,
                 keep_recent_tokens: int, max_input_tokens_per_call: int) -> None:
        if self.estimate_tokens(system_prompt) > 1500:
            raise ValueError("系统提示词保守估算超过 1500 tokens")
        self.system_prompt = system_prompt
        self.compact_threshold_tokens = compact_threshold_tokens
        self.keep_recent_tokens = keep_recent_tokens
        self.max_input_tokens_per_call = max_input_tokens_per_call
        self.summary: Message | None = None

    @staticmethod
    def estimate_tokens(value: str | list[Message]) -> int:
        """使用对非 ASCII 更保守的规则估算 token。"""
        text = value if isinstance(value, str) else "".join(item.content for item in value)
        ascii_chars = sum(char.isascii() for char in text)
        return (ascii_chars + 3) // 4 + (len(text) - ascii_chars)

    def build(self, history: list[Message], state: AgentState) -> list[Message]:
        """按固定顺序组装本轮上下文：系统提示词、压缩摘要（可选）、任务、最近消息、状态提示。

        顺序与分段是稳定契约：系统提示词和任务永远在场，压缩只会削减最近消息。
        history 的第一条必须是任务消息。

        Raises:
            ValueError: history 为空。
        """
        if not history:
            raise ValueError("上下文至少需要任务消息")
        task, recent = history[0], history[1:]
        messages = [Message("system", self.system_prompt)]
        if self.summary:
            messages.append(self.summary)
        messages.append(task)
        messages.extend(recent)
        messages.append(Message("system", state.render_system_hint()))
        return messages

    def context_too_large(self, history: list[Message], state: AgentState) -> bool:
        """判断当前可见上下文是否达到压缩阈值。"""
        return self.estimate_tokens(self.build(history, state)) > self.compact_threshold_tokens

    def compact(self, history: list[Message], state: AgentState,
                provider: LLMProvider) -> CompactResult:
        """压缩历史；摘要生成失败时确定性保留任务和最近消息。

        压缩本身是一次真实的模型调用，会消耗预算也可能失败，因此用量、
        降级标记和压缩前后的上下文规模一并随结果返回，由调用方记账和留痕。
        """
        before = self.estimate_tokens(self.build(history, state))
        usage = Usage()
        degraded = False
        task, candidates = history[0], history[1:]
        prompt = (DEFAULT_COMPACT_PROMPT + "\n\n".join(f"{item.role}: {item.content}" for item in candidates))
        try:
            response = provider.call([Message("system", COMPACT_SUMMARIZER_PROMPT),
                                      Message("user", prompt)])
            self.summary = Message("system", f"Compaction summary:\n{response.text}")
            usage = response.usage
        except Exception:
            # 摘要生成失败不终止运行，退化成只保留最近消息，由 degraded 暴露给轨迹。
            self.summary = None
            degraded = True
        # 从新到旧保留完整消息，不切开单条：切开会破坏 tool_call 与结果的配对。
        kept: list[Message] = []
        used = 0
        for message in reversed(candidates):
            size = self.estimate_tokens(message.content)
            if kept and used + size > self.keep_recent_tokens:
                break
            kept.append(message)
            used += size
        kept.reverse()
        compacted = [task, *kept]
        return CompactResult(compacted, usage, degraded,
                             before, self.estimate_tokens(self.build(compacted, state)))
