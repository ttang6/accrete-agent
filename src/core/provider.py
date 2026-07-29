"""OpenAI-compatible 的非流式模型提供者。"""

from abc import ABC, abstractmethod
import json

from openai import APIError, APIStatusError, OpenAI
from openai.types.chat import ChatCompletion

from .types import LLMResponse, Message, ToolCall, Usage


class ProviderError(RuntimeError):
    """不可恢复的模型提供者错误。"""


class LLMProvider(ABC):
    """运行循环使用的模型调用抽象。"""

    @abstractmethod
    def call(self, messages: list[Message], tools: list[dict] | None = None) -> LLMResponse:
        """同步调用模型并规整响应。"""


class OpenAICompatProvider(LLMProvider):
    """通过 openai SDK 调用兼容 OpenAI 的 Chat Completions 服务。

    职责边界：只保证协议本身能被正确解析，解析不了就抛 ProviderError。
    能解析但工具调用不合法或执行失败，由 Agent Loop 反馈给模型。
    SDK 异常一律在此收敛为 ProviderError，不向运行循环泄漏。
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout_s: float,
                 max_retries: int, strict_tool_schema: bool = False,
                 max_output_tokens: int | None = None) -> None:
        if not (base_url and api_key and model):
            raise ValueError("base_url、api_key、model 均不能为空")
        self.model = model
        self.strict_tool_schema = strict_tool_schema
        self.max_output_tokens = max_output_tokens or 0
        # 传输层抖动的重试交给 SDK：指数退避、抖动与 Retry-After 都由它处理。
        # 它只治网络与 429/5xx，业务错误直接抛出，不会把失败悄悄吞掉。
        self.client = OpenAI(base_url=base_url, api_key=api_key,
                             timeout=timeout_s, max_retries=max_retries)

    def call(self, messages: list[Message], tools: list[dict] | None = None) -> LLMResponse:
        """构造请求 → 调用模型 → 解析响应。"""
        payload = self._build_request(messages, tools)
        completion = self._create(payload)
        return self._parse_response(completion)

    def close(self) -> None:
        """释放底层 HTTP 连接。"""
        self.client.close()

    # --- 构造请求 ----------------------------------------------------------

    def _build_request(self, messages: list[Message], tools: list[dict] | None) -> dict:
        payload: dict = {"model": self.model,
                         "messages": [self._build_message(item) for item in messages]}
        if tools:
            payload["tools"] = [self._build_tool_schema(item) for item in tools]
        if self.max_output_tokens:
            payload["max_tokens"] = self.max_output_tokens
        return payload

    @staticmethod
    def _build_message(message: Message) -> dict:
        data = {"role": message.role, "content": message.content}
        if message.tool_call_id:
            data["tool_call_id"] = message.tool_call_id
        if message.tool_calls:
            data["tool_calls"] = [{"id": call.id, "type": "function", "function": {
                "name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False),
            }} for call in message.tool_calls]
        return data

    def _build_tool_schema(self, schema: dict) -> dict:
        schema = json.loads(json.dumps(schema))
        function = schema.get("function", {})
        compatible = function.pop("strict_compatible", True)
        if self.strict_tool_schema and compatible:
            function["strict"] = True
        return schema

    # --- 调用 --------------------------------------------------------------

    def _create(self, payload: dict) -> ChatCompletion:
        try:
            return self.client.chat.completions.create(**payload)
        except APIStatusError as exc:
            raise ProviderError(f"模型调用失败，HTTP {exc.status_code}: {exc.message}") from exc
        except APIError as exc:
            raise ProviderError(f"模型调用失败: {exc}") from exc

    # --- 解析响应 ----------------------------------------------------------

    def _parse_response(self, completion: ChatCompletion) -> LLMResponse:
        if not completion.choices:
            raise ProviderError("模型响应不含 choices")
        choice = completion.choices[0]
        usage = Usage(input=completion.usage.prompt_tokens,
                      output=completion.usage.completion_tokens
                      ) if completion.usage else Usage(complete=False)
        # model 取服务端实际回报的那个：它才是可复现的运行身份。
        return LLMResponse(choice.message.content or "",
                           self._parse_tool_calls(choice.message.tool_calls),
                           usage,
                           choice.finish_reason or "",
                           completion.model or self.model)

    @staticmethod
    def _parse_tool_calls(raw: list | None) -> list[ToolCall]:
        """解析 tool_calls，任一项不符合协议即抛 ProviderError。

        arguments 是模型逐 token 生成的 JSON 字符串，截断或转义错误都会让它
        解析失败。这里不兜底成 {}：那会让模型和 trace 把「JSON 坏了」读成
        「参数缺失」，得到错误的失败归因。
        """
        calls = []
        for index, item in enumerate(raw or []):
            if item.type != "function":
                raise ProviderError(f"tool_calls[{index}] 的类型 {item.type} 不受支持")
            name = item.function.name or ""
            try:
                arguments = json.loads(item.function.arguments or "{}")
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"tool_calls[{index}] ({name}) 的 arguments 不是合法 JSON: {exc}") from exc
            if not isinstance(arguments, dict):
                raise ProviderError(f"tool_calls[{index}] ({name}) 的 arguments 不是对象")
            calls.append(ToolCall(item.id, name, arguments))
        return calls
