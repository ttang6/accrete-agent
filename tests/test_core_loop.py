"""新 core 主循环的完成条件、环境故障通道与用量记账。

避免误判任务完成、混淆环境故障与能力错误，以及因用量键名不一致导致成本漏记。
"""

from dataclasses import asdict

import pytest

from infra.core.loop import (DEFAULT_MAX_TURNS_WRAP_UP_PROMPT, DEFAULT_NO_TOOL_CALL_REMINDER,
                             DEFAULT_OUTPUT_TRUNCATED_REMINDER, Agent)
from infra.core.events import AgentEvent
from infra.core.hooks import HookManager
from infra.core.tracing import write_tool_state
from infra.core.tools import ToolRegistry
from infra.core.types import (Cost, EnvironmentFailure, FunctionCall, LLMResponse, Message,
                              RunLimits, ToolCall, ToolResult, Usage)
from infra.pricing import cost_of


class _StubProvider:
    """按顺序吐出预设响应；用完之后一直重复最后一条。"""

    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[list[Message]] = []

    def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.calls.append(list(messages))
        index = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[index]


class _StubTool:
    """按名字返回固定结果的工具；`raises` 用来模拟工具执行期抛异常。"""

    description = "stub"
    parameters: dict = {"type": "object", "properties": {}}

    def __init__(self, name: str, content: str = "ok", *, is_error: bool = False,
                 raises: BaseException | None = None) -> None:
        self.name = name
        self.content = content
        self.is_error = is_error
        self.raises = raises

    def execute(self, arguments: dict) -> ToolResult:
        if self.raises is not None:
            raise self.raises
        return ToolResult(tool_call_id="", content=self.content, is_error=self.is_error,
                          error_type="exec_error" if self.is_error else None)


def _text(content: str, usage: Usage | None = None, *, finish_reason: str | None = None) -> LLMResponse:
    return LLMResponse(message=Message(role="assistant", content=content),
                       usage=usage or Usage(), finish_reason=finish_reason)


def _call(tool: str, content: str = "", usage: Usage | None = None) -> LLMResponse:
    return LLMResponse(
        message=Message(role="assistant", content=content,
                        tool_calls=[ToolCall(id="c1", function=FunctionCall(tool, "{}"))]),
        usage=usage or Usage())


def _registry(*tools: _StubTool) -> ToolRegistry:
    registry = ToolRegistry()
    for tool in tools:
        registry.register(tool)
    return registry


# --- 完成条件 ------------------------------------------------------------


def test_assistant_stop_treats_a_text_reply_as_delivery():
    """交互式场景：不发工具调用就是给出了答案。"""
    agent = Agent(_StubProvider([_text("答案")]), _registry())

    result = agent.run("任务")

    assert result.finish_reason == "completed"
    assert result.final_text == "答案"
    assert result.turns == 1


def test_assistant_stop_reprompts_after_output_length_truncation():
    """被服务端截断的文本不是完成，应带着续写提示进入下一轮。"""
    provider = _StubProvider([
        _text("半截答复", finish_reason="length"),
        _text("完整答复", finish_reason="stop"),
    ])
    agent = Agent(provider, _registry())

    result = agent.run("任务")

    assert result.finish_reason == "completed"
    assert result.final_text == "完整答复"
    assert result.turns == 2
    assert provider.calls[1][-1].content == DEFAULT_OUTPUT_TRUNCATED_REMINDER


def test_max_turns_requests_one_tool_free_wrap_up():
    """达到轮数上限后，用完整已有上下文额外请求一次用户可读的收尾答复。"""
    provider = _StubProvider([_call("noop"), _text("收尾答复")])
    agent = Agent(provider, _registry(_StubTool("noop")))

    result = agent.run("任务", RunLimits(max_turns=1))

    assert result.finish_reason == "max_turns"
    assert result.final_text == "收尾答复"
    assert result.turns == 1
    assert len(provider.calls) == 2
    assert provider.calls[1][-1].content == DEFAULT_MAX_TURNS_WRAP_UP_PROMPT


def test_submit_tool_rejects_a_text_reply_and_reprompts():
    """批跑场景：纯文本是协议违规，回灌提醒后重来，不算完成。"""
    provider = _StubProvider([_text("我做完了"), _call("submit")])
    agent = Agent(provider, _registry(_StubTool("submit", "最终答复")),
                  completion_mode="submit_tool", submit_tool="submit")

    result = agent.run("任务")

    assert result.finish_reason == "completed"
    # final 取的是提交工具的结果，不是模型那句"我做完了"。
    assert result.final_text == "最终答复"
    assert result.turns == 2
    assert provider.calls[1][-1].content == DEFAULT_NO_TOOL_CALL_REMINDER


def test_submit_tool_gives_up_after_repeated_text_replies():
    """连续违规到阈值就放弃，避免模型光说不做把预算烧完。"""
    agent = Agent(_StubProvider([_text("我做完了")]), _registry(_StubTool("submit")),
                  completion_mode="submit_tool", submit_tool="submit")

    result = agent.run("任务", RunLimits(max_consecutive_schema_errors=3))

    assert result.finish_reason == "repeated_format_error"
    assert result.turns == 3


def test_submit_tool_ignores_a_failed_submission():
    """提交工具失败不算交付：运行继续，直到真的提交成功。"""
    provider = _StubProvider([_call("submit"), _call("submit")])
    tool = _StubTool("submit", "炸了", is_error=True)
    agent = Agent(provider, _registry(tool), completion_mode="submit_tool",
                  submit_tool="submit")

    result = agent.run("任务", RunLimits(max_turns=2))

    assert result.finish_reason == "max_turns"
    assert result.final_message is None


def test_submit_or_stop_accepts_either_exit():
    """两者皆可：纯文本也算完成。"""
    agent = Agent(_StubProvider([_text("答案")]), _registry(_StubTool("submit")),
                  completion_mode="submit_or_stop", submit_tool="submit")

    result = agent.run("任务")

    assert result.finish_reason == "completed"
    assert result.final_text == "答案"


def test_submit_modes_require_a_submit_tool_name():
    """core 不认识任何具体工具名，缺了它属于装配期错误。"""
    with pytest.raises(ValueError):
        Agent(_StubProvider([]), _registry(), completion_mode="submit_tool")


# --- 环境故障 ------------------------------------------------------------


def test_environment_failure_is_not_translated_into_a_tool_error():
    """环境死了不能喂回模型：直接收尾，并与 run_error 分开记。"""
    tool = _StubTool("bash", raises=EnvironmentFailure("容器没了"))
    agent = Agent(_StubProvider([_call("bash")]), _registry(tool))

    result = agent.run("任务")

    assert result.finish_reason == "environment_failure"
    assert "容器没了" in (result.error or "")


def test_other_tool_exceptions_still_become_tool_errors():
    """普通工具异常仍然翻译成 ToolResult 喂回模型，让它自己纠正。"""
    tool = _StubTool("bash", raises=RuntimeError("命令挂了"))
    agent = Agent(_StubProvider([_call("bash")]), _registry(tool))

    result = agent.run("任务", RunLimits(max_turns=1))

    assert result.finish_reason == "max_turns"


def test_registry_rejects_invalid_tool_arguments_before_execution():
    """缺字段、额外字段和错类型都必须计为 schema_error，而非工具执行失败。"""
    class StrictTool(_StubTool):
        parameters = {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        }

    tool = StrictTool("read")
    registry = _registry(tool)

    result = registry.dispatch(ToolCall("c1", function=FunctionCall("read", '{"extra": 1}')))

    assert result.is_error is True
    assert result.error_type == "schema_error"
    assert "required property" in result.content


def test_registry_hides_and_rejects_disabled_tools():
    """禁用工具后既不再暴露给模型，也不能绕过清单直接执行。"""
    registry = _registry(_StubTool("read"), _StubTool("write"))

    registry.disable("read")

    assert registry.is_enabled("read") is False
    assert {item["function"]["name"] for item in registry.openai_tools()} == {"write"}
    result = registry.dispatch(ToolCall("c1", function=FunctionCall("read", "{}")))
    assert result.is_error is True
    assert result.error_type == "permission"

    registry.enable("read")

    assert registry.is_enabled("read") is True
    assert registry.dispatch(ToolCall("c2", function=FunctionCall("read", "{}"))).is_error is False


def test_registry_state_changes_require_a_registered_tool():
    """错误工具名必须立即暴露，不能静默留下无效禁用状态。"""
    registry = _registry()

    with pytest.raises(KeyError, match="未注册工具"):
        registry.disable("read")


def test_registry_defaults_unknown_tool_permissions_to_mutating():
    """旧测试替身和外部工具未声明权限时不能被宽松放行。"""
    registry = _registry(_StubTool("legacy"))

    assert registry.permission_group("legacy") == "mutating"


def test_registry_exposes_declared_tool_permission_group():
    """gate 通过 registry 读取工具权限，而不硬编码工具名。"""
    class NetworkTool(_StubTool):
        permission_group = "network"

    registry = _registry(NetworkTool("search"))

    assert registry.permission_group("search") == "network"


def test_before_tool_abort_is_traced_and_reaches_after_tool():
    """hook 拒绝也是一次可追溯工具调用，供 gate 留下拒绝依据。"""
    class RecordingTrajectory:
        run_id = "run-test"

        def __init__(self) -> None:
            self.entries = []

        def write(self, entry):
            self.entries.append(entry)
            return "event"

    trajectory = RecordingTrajectory()
    hooks = HookManager()
    after_results = []

    def reject(ctx) -> None:
        ctx.abort = True
        ctx.abort_reason = "需要 manifest"

    def record_after(ctx) -> None:
        after_results.append(ctx.tool_result)

    hooks.on(AgentEvent.BEFORE_TOOL, reject)
    hooks.on(AgentEvent.AFTER_TOOL, record_after)
    agent = Agent(
        _StubProvider([_call("write"), _text("停止")]),
        _registry(_StubTool("write")),
        hooks=hooks,
        trajectory=trajectory,
    )

    result = agent.run("任务")

    tool_entry = next(entry for entry in trajectory.entries if entry["type"] == "tool_exec")
    assert result.finish_reason == "completed"
    assert tool_entry["status"] == "error"
    assert tool_entry["error_class"] == "permission"
    assert tool_entry["output_bytes"] == len("需要 manifest".encode("utf-8"))
    assert len(after_results) == 1
    assert after_results[0].tool_call_id == "c1"


def test_before_tool_result_is_traced_and_reaches_after_tool():
    """hook 直接给出的工具结果也必须经过统一收尾路径。"""
    class RecordingTrajectory:
        run_id = "run-test"

        def __init__(self) -> None:
            self.entries = []

        def write(self, entry):
            self.entries.append(entry)
            return "event"

    trajectory = RecordingTrajectory()
    hooks = HookManager()
    after_results = []

    def shortcut(ctx) -> None:
        ctx.tool_result = ToolResult("", "hook result")

    hooks.on(AgentEvent.BEFORE_TOOL, shortcut)
    hooks.on(AgentEvent.AFTER_TOOL, lambda ctx: after_results.append(ctx.tool_result))
    agent = Agent(
        _StubProvider([_call("read"), _text("停止")]),
        _registry(_StubTool("read")),
        hooks=hooks,
        trajectory=trajectory,
    )

    agent.run("任务")

    tool_entry = next(entry for entry in trajectory.entries if entry["type"] == "tool_exec")
    assert tool_entry["status"] == "ok"
    assert tool_entry["output_bytes"] == len("hook result".encode("utf-8"))
    assert after_results[0].tool_call_id == "c1"


def test_tool_state_event_records_why_a_tool_was_disabled():
    """配额或熔断策略可把工具状态变化写入当前轨迹。"""
    class RecordingTrajectory:
        def __init__(self) -> None:
            self.entries = []

        def write(self, entry):
            self.entries.append(entry)
            return "event"

        def close(self) -> None:
            pass

    trajectory = RecordingTrajectory()

    write_tool_state(
        trajectory,
        tool_name="read",
        enabled=False,
        reason="read_budget_exhausted",
        turn=2,
    )

    assert trajectory.entries == [{
        "type": "tool_state",
        "tool_name": "read",
        "enabled": False,
        "reason": "read_budget_exhausted",
        "turn": 2,
    }]


# --- 用量记账 ------------------------------------------------------------


def test_usage_accumulates_across_turns_and_keeps_incompleteness():
    """两轮用量相加；任一侧不完整，总账就不完整。"""
    provider = _StubProvider([
        _call("noop", usage=Usage(input=100, output=10, cached_input=40)),
        _text("答案", usage=Usage(input=200, output=20, complete=False)),
    ])
    agent = Agent(provider, _registry(_StubTool("noop")))

    result = agent.run("任务")

    assert (result.usage.input, result.usage.output) == (300, 30)
    assert result.usage.cached_input == 40
    assert result.usage.complete is False
    assert result.usage.total == 330


def test_usage_keys_match_what_pricing_reads():
    """用量按 asdict 写进轨迹，键名与 pricing 对不上会让成本静默算成 0。"""
    pricing = {"models": {"m": {"prices": {"cn": {
        "input_per_1m": 1.0, "output_per_1m": 2.0,
        "cached_input_per_1m": 0.1, "currency": "CNY"}}}}}
    agent = Agent(_StubProvider([_text("答案", usage=Usage(input=1_000_000,
                                                          output=1_000_000,
                                                          cached_input=1_000_000))]),
                  _registry())

    cost = cost_of(pricing, "m", "cn", asdict(agent.run("任务").usage))

    # 一百万输入全部命中缓存，按缓存价 0.1 计；输出一百万按 2.0 计。
    assert cost is not None
    assert cost.amount == pytest.approx(2.1)


def test_history_is_prepended_but_run_budget_starts_fresh():
    """resume 只恢复模型上下文，不复用上次 run 的轮数或预算。"""
    provider = _StubProvider([_text("答案")])
    agent = Agent(provider, _registry())
    history = [Message("user", "旧任务"), Message("assistant", "旧答复")]

    result = agent.run("新任务", history=history)

    sent = provider.calls[0]
    assert sent[0].role == "system" and "# Environment" in sent[0].content
    assert [message.content for message in sent[1:]] == ["旧任务", "旧答复", "新任务"]
    assert result.turns == 1


def test_core_writes_but_never_closes_an_injected_trajectory():
    """trajectory 的生命周期属于 runner，Core 只写本次运行事件。"""
    class RecordingTrajectory:
        run_id = "run-test"

        def __init__(self) -> None:
            self.entries = []
            self.closed = False

        def write(self, entry):
            self.entries.append(entry)
            return "event"

        def close(self) -> None:
            self.closed = True

    trajectory = RecordingTrajectory()
    agent = Agent(_StubProvider([_text("答案")]), _registry(), trajectory=trajectory)

    agent.run("任务")

    assert trajectory.closed is False
    assert [entry["type"] for entry in trajectory.entries] == ["message", "finish"]


def test_thinking_is_recorded_in_the_trace_but_not_in_the_message():
    """开了思考的模型，思考文本要留在本地 trace 里；没有就不写这个字段。"""
    class RecordingTrajectory:
        run_id = "run-test"

        def __init__(self) -> None:
            self.entries = []

        def write(self, entry):
            self.entries.append(entry)
            return "event"

        def close(self) -> None:
            return None

    thought = LLMResponse(message=Message(role="assistant", content="1517"),
                          usage=Usage(), thinking="37×41 = 1517")
    trajectory = RecordingTrajectory()
    Agent(_StubProvider([thought]), _registry(), trajectory=trajectory).run("任务")

    message_entry = next(e for e in trajectory.entries if e["type"] == "message")
    assert message_entry["thinking"] == "37×41 = 1517"
    # 思考不进 message 本体：那份要能原样回灌给端点。
    assert "thinking" not in message_entry["message"]

    plain = RecordingTrajectory()
    Agent(_StubProvider([_text("1517")]), _registry(), trajectory=plain).run("任务")

    assert "thinking" not in next(e for e in plain.entries if e["type"] == "message")


def test_context_window_exceed_stops_before_llm():
    """单次请求估算超过 context_window_tokens 即以 context_exceed 结束，不进入模型调用。"""
    provider = _StubProvider([_text("完成")])
    agent = Agent(provider, _registry())

    result = agent.run("x" * 10_000, RunLimits(context_window_tokens=100))

    assert result.finish_reason == "context_exceed"
    # 护栏在 provider.complete 之前拦截：stub 未被调用。
    assert provider.calls == []


def test_default_context_window_allows_short_run():
    """默认窗口足够大时，短对话不触发护栏，正常完成。"""
    provider = _StubProvider([_text("答案")])
    agent = Agent(provider, _registry())

    result = agent.run("任务")

    assert result.finish_reason == "completed"
    assert len(provider.calls) == 1


def test_context_window_reserves_answer_token_headroom():
    """预留空间让护栏在窗口将满未满时就触发，为最终答复留出余量。"""
    # task "x"*2000 估算约 500，加 system 块约 500~700：窗口 1000 无预时不超，
    # 扣 500 预留后阈值 500，即超。对比两处可说明预留确实生效。
    bare = Agent(_StubProvider([_text("完成")]), _registry())
    budgeted = Agent(_StubProvider([_text("完成")]), _registry())

    without_reserved = bare.run("x" * 2_000, RunLimits(context_window_tokens=1_000))
    with_reserved = budgeted.run(
        "x" * 2_000,
        RunLimits(context_window_tokens=1_000, reserved_answer_tokens=500),
    )

    assert without_reserved.finish_reason == "completed"
    assert with_reserved.finish_reason == "context_exceed"


def test_cost_exceed_stops_when_cny_budget_reached():
    """累计 CNY 成本达到 max_cost_cny 即 cost_exceeded 结束。"""
    pricey = LLMResponse(
        message=Message(role="assistant", content="",
                        tool_calls=[ToolCall(id="c1", function=FunctionCall("noop", "{}"))]),
        usage=Usage(input=1_000, output=1_000, complete=True),
        cost=Cost(amount=2.0, currency="CNY", confidence="confirmed"),
    )
    provider = _StubProvider([pricey, _text("不应到达")])
    agent = Agent(provider, _registry(_StubTool("noop")))

    result = agent.run("任务", RunLimits(max_cost_cny=1.0))

    assert result.finish_reason == "cost_exceeded"
    # 第一轮累计 2.0 CNY 后，第二轮顶部判定超额，没再发请求。
    assert len(provider.calls) == 1
