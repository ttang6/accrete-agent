"""MainLoop：nanobot 风格的 LLM 驱动主循环。

用法：
    loop = MainLoop(name="assistant", llm=llm, tool_registry=registry)
    answer = loop.run([{"role": "user", "content": "..."}])  # one-shot 或 多轮
"""

import json
import re
from datetime import datetime
from typing import Optional

from nanoagent.core.agent import BaseAgent
from nanoagent.core.message import Message
from nanoagent.core.logger import RunTracer
from nanoagent.core import trace_schema as ts
from nanoagent.runtime.context_budget import ContextBudget, ContextBudgetConfig
from nanoagent.runtime.stop_condition import (
    DEFAULT_REPEAT_FAILURE_THRESHOLD,
    StopReason,
    check_repeated_failure,
    format_forced_stop_message,
)
from nanoagent.runtime.token_counter import TokenCounter
from nanoagent.runtime.tool_output_store import ToolOutputStore
from nanoagent.runtime.failure_memory import _is_tool_failure
from nanoagent.runtime.turn_context import TurnContext

# 从 [tool-call-repair-required] 文本里抽 example_call JSON 块。
# `_format_repair_text` 里固定写：
#   example_call:
#   { ... 4-space indented json ... }
# 接着是 next_call_must / do_not_repeat 行——所以 JSON 块以平衡花括号闭合，
# 然后下一非缩进行开始。这里保守 regex：找 `example_call:\n` 后第一个 `{` 开始
# 到第一个最外层 `}` 闭合的范围；json.loads 失败兜底 None（fail-open，调用方
# 直接传 None 进 trace 不影响主路径）。
_EXAMPLE_CALL_RE = re.compile(
    r"example_call:\s*\n(\{.*?\n\})", re.DOTALL
)


def _extract_repair_example(repair_text: str) -> Optional[dict]:
    if not repair_text:
        return None
    match = _EXAMPLE_CALL_RE.search(repair_text)
    if match is None:
        return None
    try:
        return json.loads(match.group(1))
    except (json.JSONDecodeError, ValueError):
        return None


# 默认 system prompt：只告诉 LLM 它是谁，不告诉它该怎么走。
# {current_datetime} 会在 run() 被替换成当前本地时间（含时区偏移）。
# 自定义 system_prompt 若包含此占位符也会被替换；不含则在末尾追加。
DEFAULT_SYSTEM_PROMPT = """你是一个智能助手，能够使用工具来完成用户的任务。

工具列表会通过 API 提供给你，你可以根据需要自由决定：
- 直接回答（如果你已经知道答案）
- 调用一个或多个工具获取信息
- 没有固定流程，根据任务实际需要灵活决定

回答要求：简洁、准确、结构化。

当前时间：{current_datetime}"""


class MainLoop(BaseAgent):
    """nanobot 风格主循环：LLM 全权决策，工具（含高阶能力）按需调用。"""

    def __init__(self, max_iterations: int = 20,
                 temperature: float = 0.0,
                 max_tool_output_chars: int = 5000,
                 max_context_tokens: int = 80_000,
                 coverage_thresholds: Optional[dict[str, int]] = None,
                 coverage_specs: Optional[list] = None,
                 repeat_failure_threshold: int = DEFAULT_REPEAT_FAILURE_THRESHOLD,
                 lesson_retriever=None,
                 token_counter: Optional[TokenCounter] = None,
                 tool_output_store: Optional[ToolOutputStore] = None,
                 context_budget_config: Optional[ContextBudgetConfig] = None,
                 **kwargs):
        """Args 说明仅列**非平凡**字段；其它字段名已自描述。

        max_tool_output_chars: 仅 `tool_output_store=None` 时生效（无 store 走 inline 截断）
        max_context_tokens: 仅 `context_budget_config=None` 时生效（fallback 用 chars/4）
        coverage_thresholds: None 用 specs 内置阈值；非 None 时按 name 覆盖（用于测试）
        coverage_specs: 装配层从 SkillContract.coverage_manifest 聚合的
            CoverageCategorySpec 列表（duck-typed 避免 main_loop 反向 import skills.contract）。
            None 时 CoverageChecker 退化为无 category 检查（framework 不内建 skill 知识）
        lesson_retriever: 非 None 时第 1 次失败也查 backend promoted lesson
        token_counter: None 时构造默认（cl100k_base + chars/4 fallback）
        tool_output_store: None 时退化到 inline `[:max_tool_output_chars]` 截断
        context_budget_config: None 时仅保留 chars/4 单点 warning，不做 budget 检查
        temperature: 透传给 think_with_tools 的采样温度（默认 0.0 确定性）
        **kwargs: 传给 BaseAgent（name, llm, tool_registry, system_prompt）
        """
        super().__init__(**kwargs)
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.max_tool_output_chars = max_tool_output_chars
        self.max_context_tokens = max_context_tokens
        self._coverage_thresholds = coverage_thresholds
        self._coverage_specs = coverage_specs
        self._repeat_failure_threshold = repeat_failure_threshold
        self._lesson_retriever = lesson_retriever
        # token_counter 默认构造一份（cl100k_base，tiktoken 不可用时自动 fallback）
        self._token_counter = token_counter if token_counter is not None else TokenCounter()
        self._tool_output_store = tool_output_store
        self._context_budget_config = context_budget_config
        self._turn_ctx: Optional[TurnContext] = None
        self._context_budget: Optional[ContextBudget] = None
        # 给 _token_dict() finalize 时算 by_source 用的最后一份 messages 快照
        self._last_messages_snapshot: Optional[list[dict]] = None
        # 单 turn delta 基准：run() 开头记，finalize 算差值
        self._usage_baseline = None  # type: Optional["TokenUsage"]
        if not self.system_prompt:
            self.system_prompt = DEFAULT_SYSTEM_PROMPT

    @staticmethod
    def _dict_to_message(d: dict) -> Message:
        role = d.get("role", "user")
        content = d.get("content", "") or ""
        if role == "user":
            return Message.user(content)
        if role == "assistant":
            return Message.assistant(content)
        return Message(role=role, content=content)  # type: ignore[arg-type]

    def _token_dict(self) -> dict:
        # 双轨字段：`total/in/out/calls` 是历史 alias 指向 session 累计，
        # `session_*` 是新名同义；`run_delta_*` 是单 turn 净增量。
        # 三轨并存是为了不破坏既有 trace 消费者，新代码该读 session_* / run_delta_*。
        u = self.llm.usage
        d = {
            "total": u.total_tokens,
            "in": u.prompt_tokens,
            "out": u.completion_tokens,
            "calls": u.call_count,
            "session_total": u.total_tokens,
            "session_in": u.prompt_tokens,
            "session_out": u.completion_tokens,
            "session_calls": u.call_count,
        }
        if self._usage_baseline is not None:
            b = self._usage_baseline
            d["run_delta_total"] = max(0, u.total_tokens - b.total_tokens)
            d["run_delta_in"] = max(0, u.prompt_tokens - b.prompt_tokens)
            d["run_delta_out"] = max(0, u.completion_tokens - b.completion_tokens)
            d["run_delta_calls"] = max(0, u.call_count - b.call_count)
        if self._last_messages_snapshot is not None:
            try:
                by_source = self._token_counter.count_by_source(self._last_messages_snapshot)
                d["by_source"] = by_source
                d["context_total"] = sum(by_source.values())
                d["counter_fallback"] = self._token_counter.is_fallback
            except Exception:
                pass
        if self._context_budget is not None:
            d["budget_snapshot"] = self._context_budget.snapshot()
        return d

    def _resolve_system_prompt(self) -> str:
        """渲染 system_prompt：含 {current_datetime} 占位符则替换，否则末尾追加。
        时间用本地时区 ISO（含偏移）精确到分钟，单次 run 内冻结。
        """
        now_iso = datetime.now().astimezone().isoformat(timespec="minutes")
        prompt = self.system_prompt
        if "{current_datetime}" in prompt:
            return prompt.replace("{current_datetime}", now_iso)
        return f"{prompt}\n\n当前时间：{now_iso}"

    def run(
        self,
        messages: list[dict],
        save_on_finish: bool = True,
        pending_action_contracts: Optional[list] = None,
        **kwargs,
    ) -> str:
        """跑一次主循环。messages 不含 system，MainLoop 自己拼。

        save_on_finish=False 时 tracer 保持 open，caller 必须手动 finalize_trace()
        ——evaluator retry 路径用此模式以让 run_continuation 复用同一 trace 文件。

        pending_action_contracts: RequiredActionGate 输入。Harness 算
        好用户本轮命中的 ActionContract 列表（duck-typed，main_loop 不 import
        skills.contract 避免反向依赖）。loop 内部传给 obligation_tracker.register。
        """
        log_input = messages[-1].get("content", "") if messages else ""
        convo: list[Message] = [Message.system(self._resolve_system_prompt())]
        for m in messages:
            convo.append(self._dict_to_message(m))

        self._tracer = RunTracer(self.name, user_input=log_input)
        self._turn_ctx = TurnContext.create(
            coverage_thresholds=self._coverage_thresholds,
            coverage_specs=self._coverage_specs,
            lesson_retriever=self._lesson_retriever,
        )
        if pending_action_contracts:
            self._turn_ctx.obligation_tracker.register(pending_action_contracts)
        if self._context_budget_config is not None:
            self._context_budget = ContextBudget(config=self._context_budget_config)
        else:
            self._context_budget = None
        self._last_messages_snapshot = None
        # MockUsage 等不实现 snapshot 时 baseline=None，_token_dict 自动跳过 delta 字段
        snapshot_fn = getattr(self.llm.usage, "snapshot", None)
        self._usage_baseline = snapshot_fn() if callable(snapshot_fn) else None
        self._log(f"Task: {log_input}")

        tool_schemas = [t.to_openai_schema() for t in self.tool_registry.get_all_tools()]

        try:
            result = self._run_inner(convo, tool_schemas)
        except Exception as e:
            self._tracer.step_error(
                e,
                action=ts.ACTION_RUN_ERROR,
                context_tags={"agent": self.name, "scope": "framework"},
            )
            self._tracer.set_token_summary(self._token_dict())
            self._tracer.save()
            raise

        if save_on_finish:
            self.finalize_trace()
        return result

    def run_continuation(
        self, messages: list[dict], tracer: RunTracer,
        save_on_finish: bool = True, **kwargs
    ) -> str:
        """复用上一轮的 tracer + turn_ctx 续跑（evaluator retry 路径）。
        system_prompt 仍重拼以让 {current_datetime} 刷新。caller 必须保证 tracer 未 .save()。
        """
        assert self._turn_ctx is not None, "run_continuation 必须在 run() 之后调"
        self._tracer = tracer

        convo: list[Message] = [Message.system(self._resolve_system_prompt())]
        for m in messages:
            convo.append(self._dict_to_message(m))

        tool_schemas = [t.to_openai_schema() for t in self.tool_registry.get_all_tools()]

        try:
            result = self._run_inner(convo, tool_schemas)
        except Exception as e:
            self._tracer.step_error(
                e,
                action=ts.ACTION_RUN_ERROR,
                context_tags={"agent": self.name, "scope": "framework", "continuation": True},
            )
            self._tracer.set_token_summary(self._token_dict())
            self._tracer.save()
            raise

        if save_on_finish:
            self.finalize_trace()
        return result

    def finalize_trace(self) -> None:
        """显式关闭 tracer 并写 token summary；多次调用安全。"""
        if self._tracer is None:
            return
        self._tracer.set_token_summary(self._token_dict())
        self._tracer.save()

    def _current_trace_id(self) -> str:
        """trace 文件 stem（'assistant_153012'），用作 ToolOutputStore 子目录名。
        tracer 未 open 时返回 'unknown_trace'，不阻塞落盘路径。
        """
        tracer = self._tracer
        if tracer is None:
            return "unknown_trace"
        path = getattr(tracer, "_trace_path", None)
        if path is None:
            return "unknown_trace"
        return path.stem

    def _estimate_context_tokens(
        self,
        messages: list[Message],
        tool_schemas: Optional[list[dict]] = None,
    ) -> int:
        """估算 messages + tools schema 总 token 数。

        tool_schemas 必须传——否则 budget 警戒线会系统性低估（项目挂了 9-10 个
        BaseTool，schema 占 prompt token 不小）。token_counter 注入则用 tiktoken
        真值，否则 chars/4 兜底。
        """
        if self._token_counter is not None:
            try:
                return self._token_counter.count_messages(
                    [m.to_dict() for m in messages],
                    tools=tool_schemas,
                )
            except Exception:
                pass
        # 防御性 fallback——counter 自身已有 chars/4 fallback，这里再兜一层防被注入坏 counter
        ctx_chars = sum(len(str(m.to_dict())) for m in messages)
        if tool_schemas:
            try:
                ctx_chars += len(json.dumps(tool_schemas, ensure_ascii=False))
            except (TypeError, ValueError):
                ctx_chars += sum(len(str(t)) for t in tool_schemas)
        return ctx_chars // 4

    # ============================================================
    # post-tool / pre-finish helpers（_run_inner 内联块拆出）
    #
    # 命名归并而非真正抽抽象：每个 helper 仍 mutate self._turn_ctx / self._tracer
    # 等既有属性，不引入 BaseGate / EventBus 抽象。helper 价值在 _run_inner
    # 读起来像循环。
    # ============================================================

    def _apply_repair_gate(self, c: dict, result: str, iteration: int) -> None:
        """ToolCallRepairGate trace：result 以 [tool-call-repair-required] 前缀
        开头时 emit 独立事件给 OutcomeTracker 判 ineffective_application。

        从 result 里 regex 抽 example_call JSON 块写进 repair_example
        字段，让飞轮下游 episode_extractor → lesson.suggested_action 拿到完整
        结构化修复示例（不被 fe.error_message[:120] 截断）。
        """
        if not (isinstance(result, str) and result.startswith("[tool-call-repair-required]")):
            return
        validation_error = ""
        for line in result.split("\n", 5)[1:6]:
            if line.startswith("validation_error:"):
                validation_error = line[len("validation_error:"):].strip()[:200]
                break
        repair_example = _extract_repair_example(result)
        self._trace(
            action=ts.ACTION_TOOL_CALL_REPAIR_REQUIRED, iteration=iteration,
            tool=c["name"],
            skill=c["kwargs"].get("skill", ""),
            script=c["kwargs"].get("script", ""),
            validation_error=validation_error,
            repair_example=repair_example,
        )

    def _augment_with_failure_memory(
        self, c: dict, result: str, iteration: int
    ) -> str:
        """同 (tool, args_hash) 连续失败 augment recovery hint；首次失败若 backend
        命中 promoted lesson 也 augment。返回（可能被增强的）result。"""
        augmented_result, failure_entry, used_lesson = (
            self._turn_ctx.failure_memory.maybe_augment(
                c["name"], c["kwargs"], c["raw_args"], result
            )
        )
        if failure_entry is not None:
            self._trace(
                action=ts.ACTION_FAILURE_RECOVERY_HINT, iteration=iteration,
                tool=c["name"],
                failure_count=failure_entry.failure_count,
                last_error_type=failure_entry.last_error_type,
                suggested_next_action=failure_entry.suggested_next_action,
            )
        if used_lesson is not None:
            self._trace(
                action=ts.ACTION_LESSON_USED, iteration=iteration,
                tool=c["name"],
                lesson_id=used_lesson.lesson_id,
                error_class=used_lesson.trigger.error_class,
                confidence=used_lesson.confidence,
            )
        return augmented_result

    def _observe_coverage(
        self, c: dict, augmented_result: str, iteration: int
    ) -> None:
        """硬覆盖计数 observe + ACTION_COVERAGE_CHECK trace。"""
        obs = self._turn_ctx.coverage.observe(c["name"], c["kwargs"], augmented_result)
        if obs is not None:
            self._trace(
                action=ts.ACTION_COVERAGE_CHECK, iteration=iteration,
                tool=c["name"], category=obs[0], running_max=obs[1],
            )

    def _track_obligation(self, c: dict, augmented_result: str) -> None:
        """RequiredActionGate post-tool 钩子：通知 ObligationTracker 检查是否
        满足某 obligation。失败的 call 不算 satisfy。"""
        output_is_failure = (
            isinstance(augmented_result, str)
            and _is_tool_failure(augmented_result)
        )
        self._turn_ctx.obligation_tracker.notice_tool_call(
            tool=c["name"], kwargs=c["kwargs"],
            output_is_failure=output_is_failure,
        )

    def _check_stop_condition(
        self, messages: list[Message], iteration: int
    ) -> Optional[str]:
        """同参数连续失败超阈值时构造强制终止消息并 emit STOP/FINISH trace；
        无需停止时返 None。返回非 None 时 caller 必须 return forced_msg。"""
        stop_decision = check_repeated_failure(
            self._turn_ctx.failure_memory,
            threshold=self._repeat_failure_threshold,
        )
        if not stop_decision.should_stop:
            return None
        forced_msg = format_forced_stop_message(stop_decision)
        self._log(
            f"[Iter {iteration}] → 强制终止 ({stop_decision.reason.value})",
            level="warning",
        )
        self._trace(
            action=ts.ACTION_STOP_CONDITION_MET, iteration=iteration,
            reason=stop_decision.reason.value,
            details=stop_decision.details,
        )
        self._trace(
            action=ts.ACTION_FINISH, iterations=iteration,
            reason=stop_decision.reason.value,
            output=forced_msg[:500],
        )
        self._last_messages_snapshot = [m.to_dict() for m in messages]
        return forced_msg

    def _apply_required_action_gate(
        self, messages: list[Message], response, iteration: int
    ) -> Optional[str]:
        """RequiredActionGate pre-finish 检查：未满足 obligation 时 inject 一次
        repair message 强制再一轮 LLM；已 repair 过仍不满足 → emit violation
        允许 finish（防死循环）。

        返回值：
        - "continue" → caller 应 continue（已注入 repair，messages 已 mutate）
        - None → caller 走 finish 流程（可能含 violation trace）
        """
        tracker = self._turn_ctx.obligation_tracker
        if tracker.has_unsatisfied():
            repair = tracker.build_repair_message()
            if repair is not None:
                ids = [o.id for o in tracker.unsatisfied()]
                tracker.mark_repair_emitted()
                # assistant 当前回答先入 history（LLM 看到自己说什么）
                messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                ))
                messages.append(Message.user(repair))
                self._trace(
                    action=ts.ACTION_OBLIGATION_REPAIR_INJECTED,
                    iteration=iteration,
                    obligation_ids=ids,
                )
                self._log(
                    f"[Iter {iteration}] → 注入 RequiredActionGate repair: {ids}",
                )
                return "continue"

        # 已 repair 过仍不满足 → 记 violation 但允许 finish（防死循环）
        still_open = tracker.all_unsatisfied_including_repaired()
        if still_open:
            self._trace(
                action=ts.ACTION_OBLIGATION_VIOLATION,
                iteration=iteration,
                obligation_ids=[o.id for o in still_open],
            )
            self._log(
                f"[Iter {iteration}] → obligation violation, finish anyway: "
                f"{[o.id for o in still_open]}",
                level="warning",
            )
        return None

    def _run_inner(
        self, messages: list[Message], tool_schemas: list[dict]
    ) -> str:
        """主循环本体。与 run() 分开是为了 top-level try/except 包住异常路径。"""
        last_content = ""

        for iteration in range(1, self.max_iterations + 1):
            # ---- context 膨胀监控 ----
            ctx_tokens_est = self._estimate_context_tokens(messages, tool_schemas)

            if self._context_budget is not None:
                level = self._context_budget.check_total_context(ctx_tokens_est)
                if level == "exceeded":
                    self._log(
                        f"[context-budget] ⚠ 总 context ~{ctx_tokens_est:,} tokens "
                        f"超 hard 阈值 {self._context_budget.config.hard_threshold:,}",
                        level="warning",
                    )
                    self._trace(
                        action=ts.ACTION_CONTEXT_BUDGET_EXCEEDED, iteration=iteration,
                        total_tokens=ctx_tokens_est,
                        hard_threshold=self._context_budget.config.hard_threshold,
                    )
                elif level == "warning":
                    self._log(
                        f"[context-budget] 总 context ~{ctx_tokens_est:,} tokens "
                        f"超 warn 阈值 {self._context_budget.config.warn_threshold:,}",
                        level="warning",
                    )
                    self._trace(
                        action=ts.ACTION_CONTEXT_BUDGET_WARNING, iteration=iteration,
                        total_tokens=ctx_tokens_est,
                        warn_threshold=self._context_budget.config.warn_threshold,
                    )
            else:
                # 无 budget 配置时退到单点 warning，不分级
                if ctx_tokens_est > self.max_context_tokens:
                    self._log(
                        f"[context] ⚠ 估算 context ~{ctx_tokens_est:,} tokens "
                        f"已超警戒线 {self.max_context_tokens:,}（Iter {iteration}）",
                        level="warning",
                    )

            # ---- 调用 LLM ----
            self._log(f"[Iter {iteration}] → calling LLM (messages={len(messages)}, ~{ctx_tokens_est:,} tokens)...")
            self._trace(action=ts.ACTION_LLM_CALL_START, iteration=iteration,
                        messages_count=len(messages))
            response = self.llm.think_with_tools(
                messages=[m.to_dict() for m in messages],
                tools=tool_schemas,
                temperature=self.temperature,
            )
            self._trace(action=ts.ACTION_LLM_CALL_END, iteration=iteration,
                        has_tool_calls=bool(response.tool_calls))

            # ---- 分支：LLM 决定调工具 vs 直接回答 ----
            if response.tool_calls:
                self._log(f"[Iter {iteration}] tool_calls: "
                          + ", ".join(tc.function.name for tc in response.tool_calls))

                messages.append(Message(
                    role="assistant",
                    content=response.content or "",
                    tool_calls=[
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in response.tool_calls
                    ],
                ))

                # ---- 解析所有 tool_call 参数，构造 call_plan ----
                # 保持 LLM 返回顺序；每项含 tc 对象 / 工具名 / raw_args / kwargs / 预设错误
                call_plan = []
                for tc in response.tool_calls:
                    tool_name = tc.function.name
                    raw_args = tc.function.arguments

                    try:
                        tool_kwargs = json.loads(raw_args)
                        if not isinstance(tool_kwargs, dict):
                            tool_kwargs = {"args": raw_args}
                    except json.JSONDecodeError:
                        tool_kwargs = {"args": raw_args}

                    prebuilt_error = None
                    if tool_name not in self.tool_registry:
                        prebuilt_error = (
                            f"错误: 工具 '{tool_name}' 不存在。"
                            f"可用: {self.tool_registry.list_tools()}"
                        )

                    call_plan.append({
                        "tc": tc,
                        "name": tool_name,
                        "raw_args": raw_args,
                        "kwargs": tool_kwargs,
                        "error": prebuilt_error,
                    })

                    self._log(f"  → {tool_name}({raw_args[:80]})")
                    self._trace(action=ts.ACTION_TOOL_CALL_START, iteration=iteration,
                                tool=tool_name, input=raw_args[:200])

                # ---- 执行：单个顺序，多个并发 ----
                runnable = [c for c in call_plan if c["error"] is None]
                results_by_index: dict[int, str] = {}

                if len(runnable) <= 1:
                    for c in runnable:
                        idx = call_plan.index(c)
                        results_by_index[idx] = self.tool_registry.execute(
                            c["name"], **c["kwargs"]
                        )
                else:
                    # execute_parallel 内部 asyncio + 线程池，已处理异常隔离
                    parallel_calls = [
                        {"name": c["name"], "kwargs": c["kwargs"]}
                        for c in runnable
                    ]
                    parallel_results = self.tool_registry.execute_parallel(parallel_calls)
                    for c, r in zip(runnable, parallel_results):
                        idx = call_plan.index(c)
                        results_by_index[idx] = r["result"]

                # ---- 组装每个工具结果，统一截断、trace end、追加到 messages ----
                for idx, c in enumerate(call_plan):
                    if c["error"] is not None:
                        result = c["error"]
                    else:
                        result = results_by_index[idx]

                    # 注入 store 走 token-based 截断 + 落盘；否则 inline char 截断
                    if self._tool_output_store is not None:
                        force_store = (
                            self._context_budget is not None
                            and self._context_budget.should_force_store()
                        )
                        stored = self._tool_output_store.store_if_needed(
                            tool_name=c["name"],
                            tool_call_id=c["tc"].id,
                            trace_id=self._current_trace_id(),
                            output=result,
                            force_store=force_store,
                        )
                        if stored.truncated:
                            # 仅落盘成功才发 SAVED——避免 trace 写"已保存"但文件不存在，
                            # 误导后续 EpisodeExtractor / 复盘工具
                            if stored.full_path is not None:
                                self._trace(
                                    action=ts.ACTION_TOOL_OUTPUT_SAVED, iteration=iteration,
                                    tool=c["name"], tool_call_id=c["tc"].id,
                                    original_tokens=stored.original_tokens,
                                    preview_tokens=stored.preview_tokens,
                                    full_path=str(stored.full_path),
                                    reason=stored.reason,
                                )
                            # TRUNCATED 不论落盘成功都发：prompt 总被截了
                            self._trace(
                                action=ts.ACTION_TOOL_OUTPUT_TRUNCATED, iteration=iteration,
                                tool=c["name"],
                                original_tokens=stored.original_tokens,
                                preview_tokens=stored.preview_tokens,
                                reason=stored.reason,
                            )
                        result = stored.preview
                        # preview_tokens 是真正进 prompt 的量
                        if self._context_budget is not None:
                            self._context_budget.record_tool_tokens(stored.preview_tokens)
                    elif len(result) > self.max_tool_output_chars:
                        result = result[:self.max_tool_output_chars] + "\n...(截断)"

                    self._apply_repair_gate(c, result, iteration)
                    augmented_result = self._augment_with_failure_memory(
                        c, result, iteration
                    )
                    self._observe_coverage(c, augmented_result, iteration)
                    self._track_obligation(c, augmented_result)

                    self._log(f"  {c['name']} → {augmented_result[:100]}", level="debug")
                    self._trace(action=ts.ACTION_TOOL_CALL_END, iteration=iteration,
                                tool=c["name"], input=c["raw_args"][:200],
                                output=augmented_result[:300])

                    messages.append(Message.tool_result(
                        content=augmented_result,
                        tool_call_id=c["tc"].id,
                        tool_name=c["name"],
                    ))

                # 本 iter tool 落完后若仍有 missing category，注入 synthetic user 提示。
                # counts 非空只在观察过成功 coverage tool 后——首次失败 / 非 ai-digest 都不触发
                if self._turn_ctx.coverage.counts:
                    hint = self._turn_ctx.coverage.build_hint()
                    if hint is not None:
                        messages.append(Message.user(hint))
                        self._trace(action=ts.ACTION_COVERAGE_HINT_INJECTED, iteration=iteration,
                                    missing=self._turn_ctx.coverage.missing_categories(),
                                    counts=dict(self._turn_ctx.coverage.counts),
                                    observed_empty=sorted(self._turn_ctx.coverage.observed_empty))

                # 结构化 stop：同参数连续失败超阈值强制退出。
                # 放在 coverage_hint 之后是为了让 hint 已注入到 trace（方便复盘判因果）
                forced_msg = self._check_stop_condition(messages, iteration)
                if forced_msg is not None:
                    return forced_msg

                continue

            else:
                # RequiredActionGate：LLM 想 finish 前检查未满足 obligation。
                # gate 返回 "continue" 时本 iter 已注入 repair → 走下一轮 LLM
                if self._apply_required_action_gate(messages, response, iteration) == "continue":
                    continue

                # LLM 直接回答 → 结束。tracer.save() 由 caller 负责（让 evaluator
                # retry 路径能复用同一个 tracer）
                last_content = response.content or ""
                self._log(f"[Iter {iteration}] → 直接回答 ({len(last_content)} 字)")
                self._trace(action=ts.ACTION_FINISH, iterations=iteration,
                            output=last_content[:500])
                self._last_messages_snapshot = [m.to_dict() for m in messages]
                return last_content

        # 安全兜底
        self._log(f"达到最大迭代 {self.max_iterations}，强制结束", level="warning")
        self._trace(action=ts.ACTION_FINISH, iterations=self.max_iterations,
                    reason="max_iterations", output=last_content[:500])
        self._last_messages_snapshot = [m.to_dict() for m in messages]
        return last_content or "达到最大迭代次数，未能完成任务。"
