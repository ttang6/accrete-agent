"""runtime/providers/openai_compat.py

最小 OpenAI-compatible Provider。
支持：
- GPT-4 / GPT-5 系列
- Qwen(3) ~ 3.7 Flash / Plus（DashScope compatible-mode）
- DeepSeek V4 Flash / Pro
- Grok 4.6（xAI compatible-mode）

API Key 从环境变量读取（配合 load_dotenv）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI
from yaml import YAMLError

# 假设你 core 里已有这些类型，按实际路径调整
from infra.core.types import Cost, Message, LLMResponse, Usage, ToolCall, FunctionCall
from infra.core.provider import LLMProvider   # 你的协议
from infra.pricing import cost_of, load_pricing


# ---------------------------------------------------------------------------
# 模型 → base_url / 默认 key 环境变量名
# ---------------------------------------------------------------------------

@dataclass
class Endpoint:
    base_url: str
    api_key_env: str


ENDPOINTS: dict[str, Endpoint] = {
    # OpenAI 官方
    "openai": Endpoint("https://api.openai.com/v1", "OPENAI_API_KEY"),
    # 通义 DashScope compatible-mode
    "dashscope": Endpoint(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "DASHSCOPE_API_KEY",
    ),
    # DeepSeek
    "deepseek": Endpoint("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    # xAI Grok
    "xai": Endpoint("https://api.x.ai/v1", "XAI_API_KEY"),
}


def _detect_vendor(model: str) -> str:
    m = model.lower()
    if m.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
        return "openai"
    if m.startswith(("qwen", "qwq")):
        return "dashscope"
    if m.startswith("deepseek"):
        return "deepseek"
    if m.startswith("grok"):
        return "xai"
    # 默认走 OpenAI
    return "openai"


# ---------------------------------------------------------------------------
# 参数漂移处理（集中在一个小函数里）
# ---------------------------------------------------------------------------

# 只有某一家认识的 extra_body 字段 → 它属于哪家。分析配置常在模型之间复用，
# 把 DeepSeek 的 thinking 原样发给 OpenAI 会换来 400 Unknown parameter，
# 而这个错误要等到真正发请求时才暴露。
VENDOR_ONLY_EXTRA_KEYS = {
    "thinking": "deepseek",
    "enable_thinking": "dashscope",
}

# 思考文本在响应里的字段名，各家不同，按顺序取第一个非空的。
# DeepSeek 与 DashScope 都用 reasoning_content；OpenAI 的 chat.completions
# 不返回思考文本（只在 usage 里给 token 数），所以这里取不到值是正常的。
# 端点换了名字就往这个元组里加一个，不用改别处。
THINKING_FIELDS = ("reasoning_content", "thinking", "reasoning")


def _extract_thinking(message: Any) -> str | None:
    """从响应消息里取出思考文本；端点不提供时返回 None。"""
    for field in THINKING_FIELDS:
        value = getattr(message, field, None)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _build_create_kwargs(
    model: str,
    messages: list[dict],
    tools: list[dict] | None,
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    reasoning_effort: str | None = None,
    enable_thinking: bool | None = None,
    extra: dict | None = None,
) -> dict[str, Any]:
    """把内部统一参数翻译成当前模型能接受的字段。

    只认某一家的 extra 字段会按当前模型的厂商过滤掉，与本函数对 temperature 的处理一致：
    配置可以跨模型复用，翻译由这里负责。

    """
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    # ---- max_tokens 漂移 ----
    # o 系列 / 部分新 GPT 只认 max_completion_tokens
    if max_tokens is not None:
        if any(x in model.lower() for x in ("o1", "o3", "o4", "gpt-5")):
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    # ---- temperature 拒绝 ----
    # 推理模型通常不接受 temperature
    rejects_temp = any(x in model.lower() for x in ("o1", "o3", "o4"))
    if temperature is not None and not rejects_temp:
        kwargs["temperature"] = temperature

    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort

    # ---- 厂商专属（Qwen enable_thinking）----
    # OpenAI SDK 对未知字段会报错，所以走 extra_body
    vendor = _detect_vendor(model)
    extra_body = {
        key: value for key, value in (extra or {}).items()
        if VENDOR_ONLY_EXTRA_KEYS.get(key, vendor) == vendor
    }
    if enable_thinking is not None and vendor == "dashscope":
        extra_body["enable_thinking"] = enable_thinking

    if extra_body:
        kwargs["extra_body"] = extra_body

    return kwargs


# ---------------------------------------------------------------------------
# Provider 实现
# ---------------------------------------------------------------------------

class OpenAICompatProvider:
    """满足 LLMProvider 协议的薄封装。"""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        default_max_tokens: int | None = 8192,
        default_temperature: float | None = None,
        default_reasoning_effort: str | None = None,
        default_extra: dict[str, Any] | None = None,
        pricing_path: Path | None = None,
        pricing_region: str = "cn",
        billing_root: Path | None = None,
        timeout_s: float | None = None,
        max_retries: int = 2,
    ) -> None:
        self.model = model
        vendor = _detect_vendor(model)
        ep = ENDPOINTS[vendor]

        key = api_key or os.getenv(ep.api_key_env)
        if not key:
            raise RuntimeError(f"缺少 API Key，请设置环境变量 {ep.api_key_env}")

        self.client = OpenAI(
            api_key=key,
            base_url=base_url or ep.base_url,
            timeout=timeout_s,
            max_retries=max_retries,
        )
        self.default_max_tokens = default_max_tokens
        self.default_temperature = default_temperature
        self.default_reasoning_effort = default_reasoning_effort
        # 创建 provider 时设置的厂商参数；单次调用可用 extra 覆盖。
        self.default_extra = dict(default_extra or {})
        self.pricing_region = pricing_region
        self.billing_root = billing_root or Path("artifacts") / "_billing"
        try:
            default_pricing_path = Path(__file__).resolve().parents[3] / "model_api_pricing.yaml"
            self._pricing = load_pricing(pricing_path or default_pricing_path)
        except (OSError, ValueError, YAMLError):
            # 计费只是观测；价格表不可用不能阻断模型调用。
            self._pricing = None

    def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
        enable_thinking: bool | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # 1. Message → OpenAI dict
        oai_messages = [m.to_openai() for m in messages]

        # 2. 组装参数（处理漂移）
        create_kwargs = _build_create_kwargs(
            self.model,
            oai_messages,
            tools,
            max_tokens=max_tokens if max_tokens is not None else self.default_max_tokens,
            temperature=temperature if temperature is not None else self.default_temperature,
            reasoning_effort=(
                reasoning_effort
                if reasoning_effort is not None
                else getattr(self, "default_reasoning_effort", None)
            ),
            enable_thinking=enable_thinking,
            extra={**self.default_extra, **(kwargs.get("extra") or {})},
        )

        # 3. 调用
        resp = self.client.chat.completions.create(**create_kwargs)

        # 4. 转回你的类型
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                ToolCall(
                    id=tc.id,
                    function=FunctionCall(
                        name=tc.function.name,
                        arguments=tc.function.arguments,
                    ),
                )
                for tc in msg.tool_calls
            ]

        usage = Usage(complete=False)
        usage_raw = None
        if resp.usage:
            # cached / reasoning 挂在 prompt/completion_tokens_details 下，
            # 不提供的端点没有这些属性，用 getattr 兜底成 0。
            prompt_details = getattr(resp.usage, "prompt_tokens_details", None)
            completion_details = getattr(resp.usage, "completion_tokens_details", None)
            usage = Usage(
                input=resp.usage.prompt_tokens or 0,
                output=resp.usage.completion_tokens or 0,
                complete=True,
                cached_input=(
                    getattr(prompt_details, "cached_tokens", 0) or 0
                ),
                reasoning=(
                    getattr(completion_details, "reasoning_tokens", 0) or 0
                ),
            )
            usage_raw = resp.usage.model_dump()

        cost = self._record_cost(usage)

        thinking = _extract_thinking(msg)
        return LLMResponse(
            message=Message(
                role="assistant",
                content=msg.content or "",
                reasoning_content=(
                    thinking
                    if tool_calls and _detect_vendor(self.model) == "deepseek"
                    else None
                ),
                tool_calls=tool_calls,
            ),
            usage=usage,
            finish_reason=choice.finish_reason,
            raw=resp.model_dump(),   # 转为 dict 以满足 raw: dict | None 的契约
            usage_raw=usage_raw,
            cost=cost,
            thinking=thinking,
        )

    def _record_cost(self, usage: Usage) -> Cost | None:
        """估算本次调用成本，并尽力追加到月度账本。"""
        if not usage.complete:
            return None
        try:
            cost = cost_of(self._pricing or {}, self.model, self.pricing_region, {
                "input": usage.input,
                "output": usage.output,
                "cached_input": usage.cached_input,
                "reasoning": usage.reasoning,
            })
            now = datetime.now(timezone.utc)
            entry = {
                "timestamp": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "model": self.model,
                "usage": {
                    "input": usage.input,
                    "output": usage.output,
                    "cached_input": usage.cached_input,
                    "reasoning": usage.reasoning,
                },
                "cost": cost.to_dict() if cost is not None else None,
            }
            path = self.billing_root / f"{now:%Y-%m}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as billing:
                billing.write(json.dumps(entry, ensure_ascii=False) + "\n")
            return cost
        except (OSError, TypeError, ValueError, KeyError):
            # 账本或价格表的问题不能影响一次已成功的模型响应。
            return None

    def close(self) -> None:
        """关闭底层 HTTP 客户端。"""
        self.client.close()
