"""Agent 的同步最小执行循环。"""

import time

from .context import ContextManager
from .ports import NullSessionStore, NullTracer, SessionStore, Tracer
from .provider import LLMProvider
from .registry import ToolRegistry
from .state import AgentState, RunGuards, RunStopped, elapsed_ms
from .types import LLMResponse, Message, RunLimits, RunResult, ToolCall, ToolResult


class Agent:
    """编排上下文、模型、工具和轨迹。"""

    def __init__(self, provider: LLMProvider, registry: ToolRegistry, context: ContextManager,
                 state: AgentState, max_output_tokens: int, tracer: Tracer | None = None,
                 session: SessionStore | None = None) -> None:
        self.provider = provider
        self.registry = registry
        self.context = context
        self.state = state
        self.max_output_tokens = max_output_tokens
        self.tracer = tracer or NullTracer()
        self.session = session or NullSessionStore()

    def run(self, task: str, limits: RunLimits) -> RunResult:
        """执行一次任务：准备上下文 → 模型生成 → 记录响应 → 判断提交 → 执行工具。

        终止由 RunGuards 抛 RunStopped 表达；循环内其余未捕获异常转换成 run_error，
        以保证一次运行总能产出可归档的 RunResult。
        """
        history: list[Message] = [Message("user", task)]
        history.extend(self.session.load())
        guards = RunGuards(limits, self.state.started_at, self.max_output_tokens)
        try:
            while True:
                # 1. 准备上下文
                history, messages = self._prepare_context(history, guards)

                # 2. 模型生成
                response = self._call_model(messages, guards)

                # 3. 记录响应
                assistant = self._record_response(response, history)

                # 4. 判断是否提交
                if response.has_no_tool_calls():
                    self.session.append(assistant)
                    return self._finish(response.text, "completed", guards)

                # 5. 执行工具
                self._run_tool_turn(response.tool_calls, assistant, history, guards)
        except RunStopped as exc:
            self.tracer.emit("guard_tripped", reason=exc.reason, **exc.detail)
            return self._finish("", exc.reason, guards)
        except Exception as exc:
            self.tracer.emit("run_error", stage="loop", error_class=type(exc).__name__,
                             summary=str(exc)[:500])
            return self._finish("", "run_error", guards, str(exc))

    def _prepare_context(self, history: list[Message],
                         guards: RunGuards) -> tuple[list[Message], list[Message]]:
        """过前置限额、必要时压缩一次，并组装本轮模型上下文。

        压缩是一次真实的模型调用，用量在这里入账。

        Returns:
            二元组 (历史, 本轮上下文)。历史可能被压缩过，调用方必须用它替换原 history。
        """
        guards.before_turn()
        if self.context.context_too_large(history, self.state):
            started = time.perf_counter()
            result = self.context.compact(history, self.state, self.provider)
            guards.add_usage(result.usage)
            self.tracer.emit("context_compact", before_tokens=result.before_tokens,
                             after_tokens=result.after_tokens, degraded=result.degraded,
                             duration_ms=elapsed_ms(started))
            history = result.history
        messages = self.context.build(history, self.state)
        guards.before_call(self.context.estimate_tokens(messages),
                           self.context.max_input_tokens_per_call)
        return history, messages

    def _call_model(self, messages: list[Message], guards: RunGuards) -> LLMResponse:
        """调用模型并记录本次生成的用量和轨迹。"""
        started = time.perf_counter()
        response = self.provider.call(messages, self.registry.to_openai_schemas())
        guards.record_call(response.usage)
        self.tracer.emit("llm_call", iteration=guards.turns, model=response.model,
                         usage=response.usage.__dict__,
                         duration_ms=elapsed_ms(started),
                         has_tool_calls=bool(response.tool_calls))
        return response

    def _record_response(self, response: LLMResponse, history: list[Message]) -> Message:
        """将模型响应写入内存历史；工具回合的 Session 提交延后到工具跑完。"""
        assistant = Message("assistant", response.text, response.tool_calls or None)
        history.append(assistant)
        return assistant

    def _run_tool_turn(self, calls: list[ToolCall], assistant: Message,
                       history: list[Message], guards: RunGuards) -> None:
        """把本轮全部工具调用变成一个完整回合：执行、回填历史与 Session、检查止损。

        两个计数器各司其职：state.tool_call_counts 渲染进模型可见的提示，
        guards 的止损计数只给终止逻辑用，不会泄进上下文。
        止损检查放在 Session 提交之后，确保即使触发终止，本回合也已落盘。

        Raises:
            RunStopped: 止损闸触发。
        """
        tool_messages = []
        for call in calls:
            result = self.registry.dispatch(call)
            self.state.tool_call_counts[call.name] = (
                self.state.tool_call_counts.get(call.name, 0) + 1)
            guards.record_tool(call, result)
            self.tracer.emit("tool_call", tool=call.name, args_summary=str(call.arguments),
                             status="success" if result.success else "failure",
                             error_class=result.error_class,
                             result_summary=(result.output or result.error or "")[:500],
                             duration_ms=result.meta.get("duration_ms"))
            tool_messages.append(Message("tool", self._tool_text(result), tool_call_id=call.id))
        history.extend(tool_messages)
        self.session.append(assistant)
        for message in tool_messages:
            self.session.append(message)
        guards.after_tools()

    # --- 收尾与纯工具函数 --------------------------------------------------

    def _finish(self, text: str, reason: str, guards: RunGuards,
                error: str | None = None) -> RunResult:
        """组装运行结果，并落一条 finish 轨迹。所有终止路径都必须经过这里。"""
        result = RunResult(text, reason, guards.turns, guards.usage, error)
        self.tracer.emit("finish", finish_reason=reason, turns=guards.turns,
                         final_text=text[:500])
        return result

    @staticmethod
    def _tool_text(result: ToolResult) -> str:
        """渲染回填给模型的工具结果；失败时带上错误分类，便于模型自行纠正。"""
        return result.output if result.success else f"ERROR [{result.error_class}]: {result.error}"

