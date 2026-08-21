"""最小 runner 的资源收尾与会话接续。"""

from pathlib import Path

import pytest

from infra.core.tools import ToolRegistry
from infra.core.events import AgentEvent
from infra.core.hooks import HookManager
from infra.core.types import Cost, FunctionCall, LLMResponse, Message, ToolCall, ToolResult, Usage
from infra.runtime.runner import run_agent
from infra.runtime.session import list_sessions, resume_session


class StubProvider:
    def __init__(self) -> None:
        self.closed = False
        self.calls: list[list[Message]] = []

    def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
        self.calls.append(list(messages))
        return LLMResponse(Message("assistant", "答复"), Usage())

    def close(self) -> None:
        self.closed = True


class StubEnvironment:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


def test_runner_records_session_and_closes_owned_resources(tmp_path: Path):
    provider = StubProvider()
    environment = StubEnvironment()

    result = run_agent(
        "第一问",
        provider=provider,
        tools=ToolRegistry(),
        workdir="/workspace",
        artifacts_root=tmp_path,
        environment=environment,
    )

    assert result.final_text == "答复"
    session = list_sessions(tmp_path / "general")[0]
    assert session["status"] == "completed"
    assert provider.closed is True
    assert environment.closed is True
    trace = next(resume_session(session["session_id"], root=tmp_path / "general").directory.glob("run_*/trace.jsonl"))
    assert '"type": "session"' in trace.read_text(encoding="utf-8")
    assert '"type": "finish"' in trace.read_text(encoding="utf-8")


def test_runner_exposes_the_current_trajectory_to_run_hooks(tmp_path: Path):
    """target 级 hook 可在 turn 结束时写入与本次运行相同的轨迹。"""
    class ToolProvider(StubProvider):
        def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                return LLMResponse(
                    Message("assistant", tool_calls=[ToolCall(
                        "call-1", function=FunctionCall("noop", "{}"),
                    )]),
                    Usage(),
                )
            return LLMResponse(Message("assistant", "答案"), Usage())

    class NoopTool:
        name = "noop"
        description = "noop"
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

        def execute(self, arguments) -> ToolResult:
            return ToolResult("", "ok")

    seen = []
    hooks = HookManager()

    def record_trajectory(ctx) -> None:
        seen.append(ctx.trajectory)

    hooks.on(AgentEvent.TURN_END, record_trajectory)
    tools = ToolRegistry()
    tools.register(NoopTool())
    run_agent(
        "任务",
        provider=ToolProvider(),
        tools=tools,
        workdir="/workspace",
        artifacts_root=tmp_path,
        run_hooks=hooks,
    )

    assert len(seen) == 1
    assert seen[0] is not None


def test_runner_resumes_clean_history(tmp_path: Path):
    first = StubProvider()
    first_result = run_agent(
        "第一问",
        provider=first,
        tools=ToolRegistry(),
        workdir="/workspace",
        artifacts_root=tmp_path,
    )
    session_id = list_sessions(tmp_path / "general")[0]["session_id"]
    second = StubProvider()

    run_agent(
        "第二问",
        provider=second,
        tools=ToolRegistry(),
        workdir="/workspace",
        artifacts_root=tmp_path,
        session_id=session_id,
    )

    assert first_result.final_text == "答复"
    sent = second.calls[0]
    assert sent[0].role == "system" and "# Environment" in sent[0].content
    assert [message.content for message in sent[1:]] == ["第一问", "答复", "第二问"]


def test_runner_persists_the_protocol_reminder(tmp_path: Path):
    class SequenceProvider(StubProvider):
        def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
            self.calls.append(list(messages))
            if len(self.calls) == 1:
                return LLMResponse(Message("assistant", "我做完了"), Usage())
            return LLMResponse(
                Message(
                    "assistant",
                    tool_calls=[ToolCall("submit-1", function=FunctionCall("submit", "{}"))],
                ),
                Usage(),
            )

    class SubmitTool:
        name = "submit"
        description = "提交"
        parameters = {"type": "object", "properties": {}, "additionalProperties": False}

        def execute(self, arguments) -> ToolResult:
            return ToolResult("", "最终答复")

    registry = ToolRegistry()
    registry.register(SubmitTool())
    run_agent(
        "任务",
        provider=SequenceProvider(),
        tools=registry,
        workdir="/workspace",
        artifacts_root=tmp_path,
        completion_mode="submit_tool",
        submit_tool="submit",
    )

    session = resume_session(list_sessions(tmp_path / "general")[0]["session_id"], root=tmp_path / "general")
    assert [message.role for message in session.restore_history()] == [
        "user", "assistant", "user", "assistant", "tool",
    ]


def test_runner_closes_resources_when_session_open_fails(tmp_path: Path):
    provider = StubProvider()
    environment = StubEnvironment()

    with pytest.raises(FileNotFoundError):
        run_agent(
            "任务",
            provider=provider,
            tools=ToolRegistry(),
            workdir="/workspace",
            artifacts_root=tmp_path,
            session_id="missing",
            environment=environment,
        )

    assert provider.closed is True
    assert environment.closed is True


def test_repl_can_keep_resources_open_between_runs(tmp_path: Path):
    provider = StubProvider()
    environment = StubEnvironment()

    run_agent(
        "任务",
        provider=provider,
        tools=ToolRegistry(),
        workdir="/workspace",
        artifacts_root=tmp_path,
        environment=environment,
        close_resources=False,
    )

    assert provider.closed is False
    assert environment.closed is False


def test_runner_writes_response_cost_to_trace(tmp_path: Path):
    """单次调用的成本应随同 assistant message 留在原始轨迹中。"""
    class CostedProvider(StubProvider):
        def complete(self, messages, tools=None, **kwargs) -> LLMResponse:
            return LLMResponse(
                Message("assistant", "答复"),
                Usage(input=100, output=10),
                cost=Cost(0.001, "CNY", "confirmed"),
            )

    run_agent(
        "任务",
        provider=CostedProvider(),
        tools=ToolRegistry(),
        workdir="/workspace",
        artifacts_root=tmp_path,
    )

    session = list_sessions(tmp_path / "general")[0]
    trace = next(
        resume_session(session["session_id"], root=tmp_path / "general").directory.glob(
            "run_*/trace.jsonl"
        )
    )
    assert '"cost": {"amount": 0.001, "currency": "CNY", "confidence": "confirmed"}' in trace.read_text(encoding="utf-8")
