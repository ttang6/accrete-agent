from __future__ import annotations

from types import SimpleNamespace

import pytest

from infra.core.types import Message
from infra.runtime.providers.openai_compat import (
    OpenAICompatProvider,
    _build_create_kwargs,
    _detect_vendor,
)


TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {}}}]


def test_extra_body_is_forwarded_to_openai_compatible_api() -> None:
    """厂商专属请求参数通过 extra_body 发送。"""
    kwargs = _build_create_kwargs(
        "deepseek-v4-flash",
        [{"role": "user", "content": "test"}],
        tools=None,
        extra={"thinking": {"type": "disabled"}},
    )

    assert kwargs["extra_body"] == {"thinking": {"type": "disabled"}}


def test_vendor_only_extra_is_dropped_for_other_vendors() -> None:
    """只有某一家认识的字段不会跟着配置漂到别家；通用字段照常透传。

    thinking 是 DeepSeek 的；原样发给 OpenAI 会换来 400 Unknown parameter。
    """
    kwargs = _build_create_kwargs(
        "gpt-5.4-mini",
        [{"role": "user", "content": "test"}],
        tools=None,
        extra={"thinking": {"type": "enabled"}, "reasoning_effort": "low"},
    )

    assert kwargs["extra_body"] == {"reasoning_effort": "low"}


def test_grok_models_route_to_xai() -> None:
    """grok 前缀走 xAI，不要落到默认的 OpenAI 端点。"""
    assert _detect_vendor("grok-4.6") == "xai"
    assert _detect_vendor("grok-4") == "xai"


def test_grok_uses_standard_chat_completion_fields() -> None:
    """Grok 4.6 走普通 chat.completions：max_tokens 和 tools，不借用别家 extra。"""
    kwargs = _build_create_kwargs(
        "grok-4.6",
        [{"role": "user", "content": "test"}],
        tools=TOOLS,
        max_tokens=128,
        extra={"thinking": {"type": "enabled"}},
    )

    assert kwargs["model"] == "grok-4.6"
    assert kwargs["max_tokens"] == 128
    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == "auto"
    assert "max_completion_tokens" not in kwargs
    assert "extra_body" not in kwargs


def test_grok_provider_points_at_xai_chat_completions(monkeypatch) -> None:
    """构造 grok provider 时使用 xAI 的兼容端点和 XAI_API_KEY。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("XAI_API_KEY", "test-xai-key")
    provider = OpenAICompatProvider("grok-4.6")
    try:
        assert str(provider.client.base_url).rstrip("/") == "https://api.x.ai/v1"
    finally:
        provider.close()


def test_gpt_5_6_allows_function_tools_without_extra_fields() -> None:
    """gpt-5.6 使用普通 Chat Completions 工具参数时不应被本地提前拒绝。"""
    kwargs = _build_create_kwargs(
        "gpt-5.6-luna",
        [{"role": "user", "content": "test"}],
        tools=TOOLS,
    )

    assert kwargs["tools"] == TOOLS
    assert kwargs["tool_choice"] == "auto"
    assert "extra_body" not in kwargs


def test_gpt_5_6_sends_reasoning_none_as_a_standard_field() -> None:
    """关闭 GPT-5.6 reasoning 不应借用厂商 extra_body。"""
    kwargs = _build_create_kwargs(
        "gpt-5.6-luna",
        [{"role": "user", "content": "test"}],
        tools=TOOLS,
        reasoning_effort="none",
    )

    assert kwargs["reasoning_effort"] == "none"
    assert "extra_body" not in kwargs


def test_gpt_5_6_without_tools_is_untouched() -> None:
    """不带工具时 gpt-5.6 在 chat.completions 上正常，别一并拦掉。"""
    kwargs = _build_create_kwargs(
        "gpt-5.6-luna",
        [{"role": "user", "content": "test"}],
        tools=None,
    )

    assert kwargs["model"] == "gpt-5.6-luna"
    assert "tools" not in kwargs


def test_earlier_gpt_generations_still_accept_tools() -> None:
    """这条限制只属于 gpt-5.6；5.4 带工具照常放行，不要一起拦掉。"""
    kwargs = _build_create_kwargs(
        "gpt-5.4-mini",
        [{"role": "user", "content": "test"}],
        tools=TOOLS,
    )

    assert kwargs["tools"] == TOOLS


def _fake_provider(message: SimpleNamespace) -> OpenAICompatProvider:
    """造一个只回一条预设消息的 provider，用来看响应解析。"""
    calls: list[dict] = []

    class FakeCompletions:
        def create(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=None,
                model_dump=lambda: {},
            )

    provider = object.__new__(OpenAICompatProvider)
    provider.model = "deepseek-v4-flash"
    provider.default_max_tokens = None
    provider.default_temperature = None
    provider.default_reasoning_effort = None
    provider.default_extra = {}
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )
    provider.create_calls = calls
    return provider


def test_provider_uses_its_default_reasoning_effort() -> None:
    """运行装配可固定 GPT-5.6 工具调用所需的 none。"""
    provider = _fake_provider(SimpleNamespace(content="", tool_calls=None))
    provider.model = "gpt-5.6-terra"
    provider.default_reasoning_effort = "none"

    provider.complete([Message(role="user", content="test")], tools=TOOLS)

    assert provider.create_calls[0]["reasoning_effort"] == "none"


def test_thinking_text_is_lifted_out_of_the_vendor_field() -> None:
    """思考文本各家字段名不同，统一归到 LLMResponse.thinking，不混进 message。"""
    provider = _fake_provider(SimpleNamespace(
        content="答案", tool_calls=None, reasoning_content="先算 37×41"))

    response = provider.complete([Message(role="user", content="test")])

    assert response.thinking == "先算 37×41"
    # 普通回复也保留正文，供下一轮上下文回灌。
    assert response.message.content == "答案"


def test_deepseek_tool_call_preserves_reasoning_content_for_the_next_turn() -> None:
    """DeepSeek 思考模式的工具回合必须把推理原样回传。"""
    response = _fake_provider(SimpleNamespace(
        content="", reasoning_content="先读文件", tool_calls=[
            SimpleNamespace(
                id="call-1", function=SimpleNamespace(name="read", arguments="{}")
            )
        ],
    )).complete([Message(role="user", content="test")])

    assert response.message.to_openai()["reasoning_content"] == "先读文件"


def test_thinking_is_none_when_the_endpoint_does_not_return_it() -> None:
    """端点不返回思考文本时是 None——这不等于模型没思考，例如 OpenAI 只给 token 数。"""
    provider = _fake_provider(SimpleNamespace(content="答案", tool_calls=None))

    assert provider.complete([Message(role="user", content="test")]).thinking is None


def test_provider_default_extra_is_used_and_call_extra_can_override() -> None:
    """默认厂商参数适用于每次调用，单次参数优先。"""
    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=None,
                model_dump=lambda: {},
            )

    provider = object.__new__(OpenAICompatProvider)
    provider.model = "deepseek-v4-flash"
    provider.default_max_tokens = None
    provider.default_temperature = None
    provider.default_extra = {"thinking": {"type": "disabled"}, "tag": "default"}
    provider.client = SimpleNamespace(
        chat=SimpleNamespace(completions=FakeCompletions())
    )

    provider.complete(
        [Message(role="user", content="test")],
        extra={"tag": "per-call"},
    )

    assert captured["extra_body"] == {
        "thinking": {"type": "disabled"},
        "tag": "per-call",
    }


def test_provider_records_estimated_cost_in_monthly_billing_file(tmp_path) -> None:
    """成功响应的 token 用量会生成成本，并写入本地月度账本。"""
    class FakeCompletions:
        def create(self, **kwargs):
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="ok", tool_calls=None),
                        finish_reason="stop",
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=1_000_000,
                    completion_tokens=100_000,
                    prompt_tokens_details=None,
                    completion_tokens_details=None,
                    model_dump=lambda: {},
                ),
                model_dump=lambda: {},
            )

    provider = object.__new__(OpenAICompatProvider)
    provider.model = "test-model"
    provider.default_max_tokens = None
    provider.default_temperature = None
    provider.default_extra = {}
    provider.pricing_region = "cn"
    provider.billing_root = tmp_path / "billing"
    provider._pricing = {
        "models": {
            "test-model": {
                "prices": {
                    "cn": {
                        "currency": "CNY",
                        "input_per_1m": 1.0,
                        "cached_input_per_1m": None,
                        "output_per_1m": 10.0,
                        "confidence": "confirmed",
                    }
                }
            }
        }
    }
    provider.client = SimpleNamespace(chat=SimpleNamespace(completions=FakeCompletions()))

    response = provider.complete([Message(role="user", content="test")])

    assert response.cost is not None
    assert response.cost.amount == 2.0
    entries = list(provider.billing_root.glob("*.jsonl"))
    assert len(entries) == 1
    assert '"amount": 2.0' in entries[0].read_text(encoding="utf-8")
