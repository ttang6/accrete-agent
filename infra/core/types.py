from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Role = Literal["system", "user", "assistant", "tool"]

ERROR_CLASSES = frozenset({
    "schema_error",
    "exec_error",
    "timeout",
    "permission",
    "not_found",
    "unsupported",
    "cancelled",
    "unknown",
})

ErrorClass = Literal[
    "schema_error",
    "exec_error",
    "timeout",
    "permission",
    "not_found",
    "unsupported",
    "cancelled",
    "unknown",
]

CompletionMode = Literal[
    "assistant_stop",   # 无 tool_calls 即完成，final 是那条 assistant 消息
    "submit_tool",      # 只有指定工具成功才算完成，final 是它的结果
    "submit_or_stop",   # 两者皆可
]

FinishReason = Literal[
    "completed",
    "max_turns",
    "cost_exceeded",            # 累计 CNY 成本达到 max_cost_cny
    "time_exceeded",
    "repeated_format_error",
    "repeated_tool_failure",
    "environment_failure",
    "run_error",
    "aborted",               # 合并 stop_condition / killed 的主动中止
    "context_exceed",        # 单次请求上下文窗口超额（原 context_overflow，改名对齐 cost_exceeded）
]


class EnvironmentFailure(Exception):
    """执行环境自身坏了——不是命令失败，是命令**没地方跑**。

    命令退出码非零是任务信息（测试没过、grep 没匹配），该喂回模型；环境死了喂回去只会
    让模型空转、把预算烧光。因此它是唯一不许被工具层翻译成 ToolResult 的
    异常，一路穿透 ToolRegistry.dispatch 到 loop，终结整个 run。定义在 core 而非
    environment，因为工具层的兜底转换和 loop 的收尾都要认识它。
    """


@dataclass
class FunctionCall:
    """OpenAI tool_calls[].function"""
    name: str
    arguments: str  # JSON string，与 OpenAI 一致


@dataclass
class ToolCall:
    """OpenAI message.tool_calls[] 元素"""
    id: str
    type: Literal["function"] = "function"
    function: FunctionCall = field(default_factory=lambda: FunctionCall("", "{}"))

    def to_openai(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "function": {
                "name": self.function.name,
                "arguments": self.function.arguments,
            },
        }


@dataclass
class Message:
    """OpenAI chat message 子集。"""
    role: Role
    content: str | None = None
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None  # role=tool 时必填
    name: str | None = None          # 可选：function/tool 名

    def to_openai(self) -> dict[str, Any]:
        msg: dict[str, Any] = {"role": self.role}
        if self.content is not None:
            msg["content"] = self.content
        if self.reasoning_content is not None:
            msg["reasoning_content"] = self.reasoning_content
        if self.tool_calls:
            msg["tool_calls"] = [tc.to_openai() for tc in self.tool_calls]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        if self.name is not None:
            msg["name"] = self.name
        return msg


@dataclass
class Usage:
    """一次或多次模型调用的 token 用量。

    `cached_input` 与 `reasoning` 分别是 `input`、`output` 的子集，不额外相加：
    前者是命中缓存因而计价更低的那部分输入，后者是模型的思考输出。端点不提供
    对应字段时为 0，与"确实是 0"不作区分——真要分辨去看 `LLMResponse.usage_raw`。

    字段名是与 `infra/pricing.py` 之间的契约：用量按 `asdict()` 原样写进轨迹，
    键一改，成本会静默算成 0 而不是报错。
    """

    input: int = 0
    output: int = 0
    complete: bool = True
    cached_input: int = 0
    reasoning: int = 0

    @property
    def total(self) -> int:
        """计入预算的总量；cached_input 与 reasoning 已含在两侧内，不重复计。"""
        return self.input + self.output

    def add(self, other: Usage) -> Usage:
        """合并两笔用量；任一侧不完整，总账就不完整。"""
        return Usage(
            input=self.input + other.input,
            output=self.output + other.output,
            complete=self.complete and other.complete,
            cached_input=self.cached_input + other.cached_input,
            reasoning=self.reasoning + other.reasoning,
        )


@dataclass(frozen=True)
class Cost:
    """一次模型调用按本地价格表估算的金额。"""

    amount: float
    currency: str
    confidence: str

    def to_dict(self) -> dict[str, Any]:
        """返回适合写入 JSONL 轨迹和账本的值。"""
        return {
            "amount": self.amount,
            "currency": self.currency,
            "confidence": self.confidence,
        }


@dataclass
class LLMResponse:
    """一次模型响应。

    `usage_raw` 是端点原样返回的用量对象。各家在标准字段之外还有自己的名字
    （DeepSeek 的 `prompt_cache_hit_tokens` 等），原样留一份才不必为了记账去
    维护一张模型清单。

    `thinking` 是模型这一轮的思考文本。各端点的字段名不一样，由 provider 归一到这里；
    端点不返回思考文本时为 None——**这与"模型没思考"不是一回事**，例如 OpenAI 的
    chat.completions 只给思考的 token 数，不给文本。它不进上下文，只用于留痕。
    """

    message: Message          # role=assistant
    usage: Usage
    finish_reason: str | None = None  # stop | tool_calls | length | ...
    model: str | None = None
    raw: dict[str, Any] | None = None
    usage_raw: dict[str, Any] | None = None
    cost: Cost | None = None
    thinking: str | None = None


@dataclass
class ToolResult:
    tool_call_id: str
    content: str
    is_error: bool = False
    # 不进 OpenAI message；仅供 hook/轨迹
    error_type: ErrorClass | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_message(self) -> Message:
        return Message(
            role="tool",
            content=self.content,
            tool_call_id=self.tool_call_id,
        )


@dataclass
class RunLimits:
    max_turns: int = 20
    # 单次请求的上下文窗口上限（字段名与 model_api_pricing.yaml 的 context_window_tokens
    # 键逐字一致）；当前请求 messages 的近似估算超过它即以 context_exceed 结束。
    # runtime 按架构不认识价格表，真实值由装配层从价格表对应模型读出后显式传入覆盖。
    context_window_tokens: int = 128_000
    # 护栏触发的余量：实际判定以 context_window_tokens - reserved_answer_tokens 为准，
    # 让 agent 在窗口将满未满时就被叫停，为最终答复留出空间。默认 0 表示不预留。
    reserved_answer_tokens: int = 0
    # 累计成本兑底上限（CNY）。None 表示不限成本；USD 金额不计入此上限。
    max_cost_cny: float | None = None
    wall_time_s: float = 600.0
    max_consecutive_schema_errors: int = 3
    max_repeated_tool_failures: int = 3


@dataclass
class RunResult:
    """一次运行的结局。

    `final_message` 是交出最终答复的那条消息本身：`assistant_stop` 下是模型回复，
    `submit_tool` 下是提交工具的结果消息（role=tool）。保留原消息而不是只留文本，
    调用方才能分辨最终答复从哪条路径来。
    """

    final_message: Message | None
    finish_reason: FinishReason
    turns: int
    usage: Usage
    error: str | None = None

    @property
    def final_text(self) -> str:
        """最终答复的文本；没有交出最终答复时为空串。"""
        return (self.final_message.content or "") if self.final_message else ""
