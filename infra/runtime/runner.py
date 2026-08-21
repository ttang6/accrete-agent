"""把会话、运行资源与 Core 连接成一次可恢复运行。"""

from __future__ import annotations

import uuid
from contextlib import ExitStack
from pathlib import Path
from typing import Any

from infra.core.events import AgentEvent
from infra.core.hooks import HookContext, HookManager
from infra.core.loop import Agent
from infra.core.context import ContextBuilder
from infra.core.provider import LLMProvider
from infra.core.tools import ToolRegistry
from infra.core.tracing import JsonlTrajectory, write_session_header
from infra.core.types import CompletionMode, RunLimits, RunResult

from .session import Session, create_session, resolve_artifacts_root, resume_session


def run_agent(
    task: str,
    *,
    provider: LLMProvider,
    tools: ToolRegistry,
    workdir: str,
    session_id: str | None = None,
    artifacts_root: Path | None = None,
    artifact_namespace: str = "general",
    system_prompt: str | None = None,
    completion_mode: CompletionMode = "assistant_stop",
    submit_tool: str | None = None,
    limits: RunLimits | None = None,
    environment: Any | None = None,
    close_resources: bool = True,
    run_hooks: HookManager | None = None,
    debug_hooks: HookManager | None = None,
    context_builder: ContextBuilder | None = None,
) -> RunResult:
    """运行一次任务，并负责会话、轨迹和外部资源的收尾。

    Args:
        provider: 本次运行专用的模型连接；本函数结束后会尝试关闭它。
        tools: 已由调用方按 target 装配好的工具注册表。runtime 不导入具体 target。
        environment: 本次运行专用的执行环境；本函数结束后会尝试关闭它。
        session_id: 给出时恢复该会话，否则创建新会话。
        artifacts_root: artifact 的基础目录；未给出时使用 ``artifacts/``。
        artifact_namespace: 基础目录下的运行分区。默认 ``general``，不同分区
            分别维护自己的 session 索引。
        close_resources: 是否由本函数关闭 provider 和 environment；REPL 等长生命周期调用方传入 ``False`` 并自行关闭。
        debug_hooks: 仅 REPL 调试使用的额外 hook（repl_debug.py）；正式调用方不要传。
        context_builder: 覆盖默认上下文构造；容器运行可用它提供与宿主不同的环境事实。
    """
    with ExitStack() as resources:
        # 先登记外部资源：后续任一装配步骤失败，仍会按相反顺序释放全部资源。
        if close_resources:
            resources.callback(_close, environment)
            resources.callback(_close, provider)
        session_root = resolve_artifacts_root(artifacts_root, artifact_namespace)
        session = (
            resume_session(session_id, root=session_root)
            if session_id is not None
            else create_session(task=task, root=session_root)
        )
        resources.callback(session.close)
        run_id = "run_" + uuid.uuid4().hex[:12]
        trace_path = session.directory / run_id / "trace.jsonl"
        trajectory = JsonlTrajectory(trace_path, run_id=run_id)
        resources.callback(trajectory.close)
        write_session_header(
            trajectory,
            cwd=workdir,
            task=task,
            extra={"session_id": session.session_id},
        )
        hooks = _session_hooks(session)
        if run_hooks is not None:
            hooks.extend(run_hooks)
        # [DEBUG] REPL 调试打印；正式调用方不传，本行可随 debug_hooks 一起移除。
        if debug_hooks is not None:
            hooks.extend(debug_hooks)

        try:
            agent = Agent(
                provider,
                tools,
                hooks=hooks,
                trajectory=trajectory,
                system_prompt=system_prompt,
                context_builder=context_builder,
                completion_mode=completion_mode,
                submit_tool=submit_tool,
            )
            result = agent.run(
                task,
                limits,
                history=session.restore_history(),
                workdir=workdir,
                run_id=run_id,
            )
            session.update(
                status="completed" if result.finish_reason == "completed" else "stopped",
                last_run_id=run_id,
                last_finish_reason=result.finish_reason,
                last_error=result.error,
            )
            return result
        except Exception as error:
            session.update(
                status="failed",
                last_run_id=run_id,
                last_finish_reason="runner_error",
                last_error=str(error),
            )
            raise


def _session_hooks(session: Session) -> HookManager:
    """把模型可见消息写入会话日志；半截工具回合由恢复端清理。"""
    hooks = HookManager()
    written_messages = 0

    def record_user(ctx: HookContext) -> None:
        nonlocal written_messages
        session.append_message(ctx.state.messages[-1])
        written_messages = len(ctx.state.messages)

    def record_assistant(ctx: HookContext) -> None:
        nonlocal written_messages
        if ctx.llm_response is not None:
            session.append_message(ctx.llm_response.message)
            written_messages += 1

    def record_tool(ctx: HookContext) -> None:
        nonlocal written_messages
        if ctx.tool_result is not None:
            session.append_message(ctx.tool_result.to_message())
            written_messages += 1

    def record_remaining_messages(ctx: HookContext) -> None:
        """记录 Core 追加的协议提醒等非工具消息。"""
        nonlocal written_messages
        for message in ctx.state.messages[written_messages:]:
            session.append_message(message)
        written_messages = len(ctx.state.messages)

    hooks.on(AgentEvent.RUN_START, record_user)
    hooks.on(AgentEvent.AFTER_LLM, record_assistant)
    hooks.on(AgentEvent.AFTER_TOOL, record_tool)
    hooks.on(AgentEvent.TURN_END, record_remaining_messages)
    return hooks


def _close(resource: Any | None) -> None:
    """关闭可关闭资源；没有 close 方法的测试替身无需特殊处理。"""
    close = getattr(resource, "close", None)
    if callable(close):
        close()
