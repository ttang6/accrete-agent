from __future__ import annotations

import time
import uuid
from dataclasses import asdict
from typing import Any

from .context import ContextBuilder, DefaultContextBuilder
from .events import AgentEvent
from .hooks import HookContext, HookManager
from .provider import LLMProvider
from .state import RunBudget, RunState, estimate_tokens
from .tools import ToolRegistry
from .tracing import Trajectory, write_message, write_tool_exec
from .types import (
    CompletionMode,
    EnvironmentFailure,
    FinishReason,
    Message,
    RunLimits,
    RunResult,
    ToolCall,
    ToolResult,
)


# 模型没发工具调用时回灌的纠正。不复述系统提示词的全部规则，只点明这一次错在哪、
# 下一步该发什么，因为它会紧跟在违规回复之后出现。
DEFAULT_NO_TOOL_CALL_REMINDER = (
    "你上一条回复没有调用任何工具，因此什么都没有发生。"
    "继续工作请调用相应工具；任务已完成或无法完成，请调用提交工具交出最终答复。"
)

DEFAULT_OUTPUT_TRUNCATED_REMINDER = (
    "你上一条回复因输出长度限制被截断。请从中断处继续，并完成对用户的答复。"
)

DEFAULT_MAX_TURNS_WRAP_UP_PROMPT = (
    "运行已达到最大轮数，不能再调用工具。请根据此前完整对话、工具结果和已完成的工作，"
    "直接给用户一份简洁的收尾答复：说明已完成的内容、依据和仍未解决的限制。"
    "不要声称尚未完成的工作已经完成。"
)


class Agent:
    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        *,
        context_builder: ContextBuilder | None = None,
        hooks: HookManager | None = None,
        trajectory: Trajectory | None = None,
        system_prompt: str | None = None,
        completion_mode: CompletionMode = "assistant_stop",
        submit_tool: str | None = None,
        no_tool_call_reminder: str = DEFAULT_NO_TOOL_CALL_REMINDER,
    ) -> None:
        """装配一次运行所需的组件与完成条件。

        Args:
            completion_mode: 什么算"交付完毕"。`assistant_stop` 下模型不发工具调用
                即完成，适合交互式对话；`submit_tool` 下只有提交工具成功才算完成，
                纯文本回复视为协议违规并回灌提醒，适合评测与批跑；`submit_or_stop` 两者皆可。
            submit_tool: 提交工具的名字，由装配层给出；只在需要提交的两种模式下必填。
            no_tool_call_reminder: 协议违规时回灌的纠正文本。

        Raises:
            ValueError: 需要提交工具的模式却没给出工具名。
        """
        if completion_mode in ("submit_tool", "submit_or_stop") and not submit_tool:
            raise ValueError(f"completion_mode={completion_mode} 必须给出 submit_tool")
        self.provider = provider
        self.tools = tools
        self.context_builder = context_builder or DefaultContextBuilder(system_prompt)
        self.hooks = hooks or HookManager()
        self.trajectory = trajectory
        self.completion_mode = completion_mode
        self.submit_tool = submit_tool
        self.no_tool_call_reminder = no_tool_call_reminder

    def run(
        self,
        task: str,
        limits: RunLimits | None = None,
        *,
        history: list[Message] | None = None,
        **run_attrs: Any,
    ) -> RunResult:
        """执行一次新任务，可在已验证的会话历史之后继续。

        Args:
            history: 已完成工具回合组成的模型可见历史。它只恢复上下文，不恢复上一次
                run 的轮数、用量或止损计数；本次运行仍使用一套新的预算。
        """
        limits = limits or RunLimits()
        run_id = str(run_attrs.get("run_id") or getattr(self.trajectory, "run_id", "") or uuid.uuid4().hex)
        started = time.time()
        state = RunState(
            run_id=run_id,
            task=task,
            workdir=str(run_attrs.get("workdir", ".")),
            started_at=started,
            messages=[*(history or []), Message(role="user", content=task)],
        )
        budget = RunBudget(limits, started)
        # 运行期窗口护栏阈值：先在装配处取一次，护栏处不再各处理想 budget.limits。
        # 已扣除预留余量，让 agent 在窗口将满未满时就被叫停，为最终答复留出空间。
        context_window = limits.context_window_tokens - limits.reserved_answer_tokens
        trajectory = self.trajectory

        self.hooks.emit(HookContext(event=AgentEvent.RUN_START, state=state))

        final: Message | None = None
        finish: FinishReason = "completed"
        error: str | None = None

        try:
            while True:
                reason = budget.check_turn(state.turn)
                if reason:
                    finish = reason
                    break
                reason = budget.check_cost(state.total_cost_cny)
                if reason:
                    finish = reason
                    break

                state.turn += 1
                self.hooks.emit(
                    HookContext(event=AgentEvent.TURN_START, state=state, turn=state.turn)
                )

                # --- context ---
                ctx = self.hooks.emit(
                    HookContext(
                        event=AgentEvent.BEFORE_CONTEXT,
                        state=state,
                        turn=state.turn,
                        messages=list(state.messages),
                    )
                )
                if ctx.abort:
                    finish, error = "aborted", ctx.abort_reason
                    break
                if ctx.messages is not None:
                    state.messages = ctx.messages

                messages = self.context_builder.build(state)
                ctx = self.hooks.emit(
                    HookContext(
                        event=AgentEvent.AFTER_CONTEXT,
                        state=state,
                        turn=state.turn,
                        messages=messages,
                    )
                )
                messages = ctx.messages or messages

                # --- context window guard ---
                # AFTER_CONTEXT 可能改写过 messages，故在这里对定稿后的请求做窗口护栏：
                # 近似估算超过运行期阈值 context_window 即结束，不进入 LLM
                # （压缩/裁剪后续经 hook 接，本轮不做）。
                if estimate_tokens(messages) > context_window:
                    finish = "context_exceed"
                    break

                # --- llm ---
                ctx = self.hooks.emit(
                    HookContext(
                        event=AgentEvent.BEFORE_LLM,
                        state=state,
                        turn=state.turn,
                        messages=messages,
                    )
                )
                if ctx.abort:
                    finish, error = "aborted", ctx.abort_reason
                    break
                messages = ctx.messages or messages

                llm_started = time.time()
                try:
                    response = self.provider.complete(
                        messages, tools=self.tools.openai_tools()
                    )
                except Exception as e:
                    self.hooks.emit(
                        HookContext(
                            event=AgentEvent.ON_ERROR,
                            state=state,
                            turn=state.turn,
                            error=e,
                        )
                    )
                    finish, error = "run_error", str(e)
                    break

                if trajectory is not None:
                    write_message(
                        trajectory,
                        response.message,
                        turn=state.turn,
                        model=response.model,
                        usage=asdict(response.usage),
                        cost=response.cost,
                        stop_reason=response.finish_reason,
                        latency_ms=round((time.time() - llm_started) * 1000, 3),
                        thinking=response.thinking,
                    )

                state.usage = state.usage.add(response.usage)
                # 成本兑底只认 CNY；USD 金额不计入（与 max_cost_cny 约定一致）。
                state.total_cost_cny += (
                    response.cost.amount
                    if response.cost is not None and response.cost.currency == "CNY"
                    else 0.0
                )
                state.messages.append(response.message)

                ctx = self.hooks.emit(
                    HookContext(
                        event=AgentEvent.AFTER_LLM,
                        state=state,
                        turn=state.turn,
                        llm_response=response,
                        messages=state.messages,
                    )
                )
                if ctx.abort:
                    finish, error = "aborted", ctx.abort_reason
                    break

                tool_calls = response.message.tool_calls or []
                if not tool_calls:
                    if response.finish_reason == "length":
                        state.messages.append(
                            Message(role="user", content=DEFAULT_OUTPUT_TRUNCATED_REMINDER)
                        )
                        self.hooks.emit(
                            HookContext(
                                event=AgentEvent.TURN_END,
                                state=state,
                                turn=state.turn,
                                trajectory=trajectory,
                            )
                        )
                        continue
                    if self.completion_mode != "submit_tool":
                        final = response.message
                        finish = "completed"
                        break
                    # 需要提交工具时，纯文本不是完成而是协议违规：留痕、计数、回灌重来。
                    stop = budget.record_format_error()
                    state.messages.append(
                        Message(role="user", content=self.no_tool_call_reminder)
                    )
                    if stop:
                        finish = stop
                        break
                    self.hooks.emit(
                        HookContext(
                            event=AgentEvent.TURN_END,
                            state=state,
                            turn=state.turn,
                            trajectory=trajectory,
                        )
                    )
                    continue

                # --- tools ---
                submitted: ToolResult | None = None
                stop = None
                for call in tool_calls:
                    result = self._run_one_tool(call, state, trajectory)
                    state.messages.append(result.to_message())
                    name = call.function.name
                    state.tool_call_counts[name] = state.tool_call_counts.get(name, 0) + 1
                    if name == self.submit_tool and not result.is_error:
                        submitted = result
                    stop = budget.record_tool(call, result)
                    if stop:
                        break
                if stop:
                    finish = stop
                    break
                # assistant_stop 下即使注册了同名工具也不认它的提交，模式必须唯一决定出口。
                if submitted is not None and self.completion_mode != "assistant_stop":
                    final = submitted.to_message()
                    finish = "completed"
                    break
                self.hooks.emit(
                    HookContext(
                        event=AgentEvent.TURN_END,
                        state=state,
                        turn=state.turn,
                        trajectory=trajectory,
                    )
                )

        # 环境没了就没有下一轮可言：不重试、不回灌给模型，直接收尾。仍然产出 RunResult，
        # 因为"环境死了"是这次 run 的一种结局，不是没有结局。它与 run_error 分开记，
        # 否则基础设施故障会在失败分桶里伪装成能力不足。
        except EnvironmentFailure as e:
            finish, error = "environment_failure", str(e)
            self.hooks.emit(
                HookContext(event=AgentEvent.ON_ERROR, state=state, error=e)
            )
        except Exception as e:
            finish, error = "run_error", str(e)
            self.hooks.emit(
                HookContext(event=AgentEvent.ON_ERROR, state=state, error=e)
            )

        if finish == "max_turns" and self.completion_mode == "assistant_stop":
            # 先广播“agent 即将终结”，让外部 hook 有机会把收尾引导换成自己的
            # 要求(如：必须按给定 schema 交 JSON)；未注册时用默认收尾提示词。
            wrap = self.hooks.emit(
                HookContext(event=AgentEvent.ON_AGENT_END, state=state, turn=state.turn)
            )
            prompt = wrap.data.get("wrap_up_prompt") or DEFAULT_MAX_TURNS_WRAP_UP_PROMPT
            state.messages.append(Message(role="user", content=prompt))
            self.hooks.emit(
                HookContext(
                    event=AgentEvent.TURN_END,
                    state=state,
                    turn=state.turn,
                    trajectory=trajectory,
                )
            )
            messages = self.context_builder.build(state)
            llm_started = time.time()
            try:
                response = self.provider.complete(messages, tools=None)
            except Exception as e:
                finish, error = "run_error", str(e)
                self.hooks.emit(
                    HookContext(event=AgentEvent.ON_ERROR, state=state, error=e)
                )
            else:
                if trajectory is not None:
                    write_message(
                        trajectory,
                        response.message,
                        model=response.model,
                        usage=asdict(response.usage),
                        cost=response.cost,
                        stop_reason=response.finish_reason,
                        latency_ms=round((time.time() - llm_started) * 1000, 3),
                        thinking=response.thinking,
                    )
                state.usage = state.usage.add(response.usage)
                state.messages.append(response.message)
                self.hooks.emit(
                    HookContext(
                        event=AgentEvent.AFTER_LLM,
                        state=state,
                        llm_response=response,
                        messages=state.messages,
                    )
                )
                final = response.message
                if wrap.data.get("treat_as_completed"):
                    # 外部 hook 把这次强制收尾视作一次有效完成（例如 overview 的
                    # JSON 兜底）；上层据此按 completed 处理，而不是 max_turns。
                    finish = "completed"
                    error = None

        if trajectory is not None:
            trajectory.write(
                {
                    "type": "finish",
                    "finish_reason": finish,
                    "turns": state.turn,
                    "error": error,
                }
            )
        self.hooks.emit(
            HookContext(
                event=AgentEvent.RUN_END,
                state=state,
                data={"finish_reason": finish, "error": error},
            )
        )
        return RunResult(
            final_message=final,
            finish_reason=finish,
            turns=state.turn,
            usage=state.usage,
            error=error,
        )

    def _run_one_tool(
        self, call: ToolCall, state: RunState, trajectory: Trajectory | None
    ) -> ToolResult:
        ctx = self.hooks.emit(
            HookContext(
                event=AgentEvent.BEFORE_TOOL,
                state=state,
                turn=state.turn,
                tool_call=call,
            )
        )
        if ctx.abort:
            result = ToolResult(
                tool_call_id=call.id,
                content=ctx.abort_reason or "aborted by hook",
                is_error=True,
                error_type="permission",
            )
        elif ctx.tool_result is not None:
            result = ctx.tool_result
            result.tool_call_id = call.id
        else:
            result = self.tools.dispatch(call)
        if trajectory is not None:
            write_tool_exec(
                trajectory,
                tool_call_id=call.id,
                tool_name=call.function.name,
                arguments=call.function.arguments,
                status="error" if result.is_error else "ok",
                turn=state.turn,
                error_class=result.error_type,
                output=result.content,
            )

        self.hooks.emit(
            HookContext(
                event=AgentEvent.AFTER_TOOL,
                state=state,
                turn=state.turn,
                tool_call=call,
                tool_result=result,
            )
        )
        return result
