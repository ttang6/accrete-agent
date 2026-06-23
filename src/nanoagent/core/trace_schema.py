"""Trace schema freeze —— 所有 tracer.step/action 的字符串契约集中点。

要把 trace 当 "source of truth" 消费（error_classifier → ReflexionStore →
promoted hints），必须先冻结 action 名 + 每 action 的预期字段，避免字符串拼错
或字段重命名导致 consumer 静默失败。

本文件职责（入门友好视角）：
- 每个 trace event 的 action 名 = 一个 `Final[str]` 常量
- 常量上方两行注释：**emit 位置 + 预期字段**
- `ALL_ACTIONS` frozenset 聚合全部——consumer 可用它做 sanity check
- 使用方式：`_trace(action=ACTION_COVERAGE_CHECK, ...)`（而不是字符串字面量）

命名约定：
- `ACTION_<VERB>_<NOUN>`: e.g. `ACTION_LLM_CALL_START`, `ACTION_TOOL_CALL_END`
- action 字符串本身沿用 snake_case（保持和现有 trace 输出一致）
- 不改任何现有 action 的字符串值，只是集中成常量以避免散落拼写错误

参考：`core/error_classifier.py` 的 Final 常量布局。
"""

from __future__ import annotations

from typing import Final


# ============================================================
# LLM 调用事件
# ============================================================

# emit: MainLoop._run_inner, 每 iter 开头调 think_with_tools 前
# fields: iteration(int), messages_count(int)
ACTION_LLM_CALL_START: Final[str] = "llm_call_start"

# emit: MainLoop._run_inner, think_with_tools 返回后
# fields: iteration(int), has_tool_calls(bool)
ACTION_LLM_CALL_END: Final[str] = "llm_call_end"


# ============================================================
# Tool 调用事件（v3 阶段四-4a：span-as-node，start/end 合一为单节点）
# ============================================================

# tool_call_start 已退役（无 emitter，常量保留占位/词表）：一次 tool 调用现在塌成
# 一个 tool_call_end 操作节点，不再 start+end 两事件靠 (iteration,tool) 配回去。
# episode_extractor 仍保留 START 分支作向后兼容兜底（读旧 trace），但新 trace 不产 START。
ACTION_TOOL_CALL_START: Final[str] = "tool_call_start"

# emit: MainLoop._emit_tool_op_node, tool execute + augment/breaker/coverage 之后（单节点）
# fields: iteration(int), tool(str), input(str, raw_args[:200]), output(str[:300]),
#         status("ok"|"failed"), op_key(str, 细键含 args-hash), klass(str, 失败时),
#         error_type(str, 粗 substring 分类), is_mutating(bool), duration_ms(int, 真耗时)
# 注：error_class(5 类)/cause_sig 是冷路径派生（需 episode 上下文），不在本节点。
ACTION_TOOL_CALL_END: Final[str] = "tool_call_end"


# ============================================================
# Coverage 事件
# ============================================================

# emit: MainLoop._run_inner, 成功 observe() 一次 ai-digest fetch_* 输出后
# fields: iteration(int), tool(str), category(enum:paper|news|oss), running_max(int)
ACTION_COVERAGE_CHECK: Final[str] = "coverage_check"

# emit: MainLoop._run_inner, 本 iter tool 循环结束后硬覆盖不达标时
# fields: iteration(int), missing(list[str]), counts(dict[str,int]),
#         observed_empty(list[str]) —— 事实空源已 confirmed 的 category，已从
#         missing 中排除；保持向后兼容时 caller 可忽略
ACTION_COVERAGE_HINT_INJECTED: Final[str] = "coverage_hint_injected"


# ============================================================
# Failure Recovery 事件
# ============================================================

# emit: MainLoop._run_inner, FailureMemory.maybe_augment 返回非 None entry 时
# fields: iteration(int), tool(str), failure_count(int), last_error_type(str),
#         suggested_next_action(str)
ACTION_FAILURE_RECOVERY_HINT: Final[str] = "failure_recovery_hint"


# ============================================================
# 结构化 Stop 事件
# ============================================================

# emit: MainLoop._run_inner, stop_condition.check_repeated_failure 返回 should_stop=True
# fields: iteration(int), reason(StopReason.value), details(dict)
ACTION_STOP_CONDITION_MET: Final[str] = "stop_condition_met"


# ============================================================
# Circuit breaker 事件（per-op 熔断，禁该 op 本轮、turn 继续）
# ============================================================

# emit: MainLoop._maybe_trip_breaker, 某 op_key 连续失败达 klass policy 阈值时
# fields: iteration(int), tool(str), op_key(str), klass(str), failure_count(int),
#         threshold(int), is_mutating(bool)
# 语义：该 op 本轮被禁，后续同 op 调用直接回 [gate-circuit-open] 消息不执行；turn 继续。
ACTION_CIRCUIT_OPEN: Final[str] = "circuit_open"


# ============================================================
# Loop 结束 / 错误
# ============================================================

# emit: MainLoop._run_inner, LLM 无 tool_calls 直接回答 / max_iter / forced stop
# fields: iterations(int), output(str, truncated to 500), reason(str, optional)
ACTION_FINISH: Final[str] = "finish"

# emit: MainLoop.run, top-level try/except 捕获未处理异常
# fields: error_class, is_transient, error_message, error_type, context_tags, action
ACTION_RUN_ERROR: Final[str] = "run_error"


# ============================================================
# Evaluator / Critic 事件（Harness 层）
# ============================================================
#
# 旧 DigestEvaluator（recommended_action 枚举 + 覆盖计数）已退役，发布流程改用
# Critic 子 agent（NL 批评、非阻断）。CALL_END 被 critic 复用记 verdict；CALL_START /
# RETRY_TRIGGERED 是旧 evaluator-optimizer 留下的占位常量，当前无 emitter（保留词表）。

# 当前无 emitter（旧 evaluator retry 路径已删）
# fields: attempt(int)
ACTION_EVALUATOR_CALL_START: Final[str] = "evaluator_call_start"

# emit: Harness._trace_critic, Critic.review() 返回后
# fields: passed(bool), critique(str, 截断 200)
ACTION_EVALUATOR_CALL_END: Final[str] = "evaluator_call_end"

# 当前无 emitter（旧 evaluator retry 路径已删）
# fields: attempt(int), recommended_action(str), missing(list)
ACTION_EVALUATOR_RETRY_TRIGGERED: Final[str] = "evaluator_retry_triggered"


# ============================================================
# 跨 trace lesson 注入事件
# ============================================================

# emit: MainLoop._run_inner, FailureMemory.maybe_augment 命中 backend lesson 时
# fields: iteration(int), tool(str), lesson_id(str), error_class(str), confidence(float)
ACTION_LESSON_USED: Final[str] = "lesson_used"

# emit: MainLoop._augment_with_failure_memory, FailureMemory 第 2 次同 op 失败
#       触发 OnlineReflector 且产出 suggestion 时
# fields: iteration(int), tool(str), diagnosis(str, 截断), has_suggested_args(bool)
# 语义：在线微反思注入了 [learned-fix] 修复假设（未验证；采纳与否由主 LLM
# 决定，下一次工具调用即环境验证）。复盘 / 小测用此事件量 hint→重试成功率。
ACTION_REFLECTOR_HINT: Final[str] = "reflector_hint"


# ============================================================
# ToolCallRepairGate 事件（schema 校验前置拦截）
# ============================================================

# emit: MainLoop._run_inner, 检测到 tool_result 以 [gate-invalid-call] 开头
# fields: iteration(int), tool(str), skill(str), script(str), validation_error(str)
# 语义：LLM 生成的 tool call args 在 subprocess 前被 jsonschema 拦截。
# OutcomeTracker 用此事件判 lesson "ineffective_application" —— 即 lesson
# 已召回但 LLM 仍生成同样的非法调用，说明文本建议未转化为合法 call。
ACTION_TOOL_CALL_REPAIR_REQUIRED: Final[str] = "tool_call_repair_required"


# ============================================================
# RequiredActionGate（v3 阶段三-2b 已退役，常量保留位置/格式）
# ============================================================
#
# RequiredActionGate / ObligationTracker 整条退役（发布流程改成确定性的 publish
# pipeline，mark 由副作用层自动写、不再靠 obligation 逼 LLM）。两个常量**刻意保留**：
# eval grader 仍读它们做计数（无 emitter → 恒为 0，不是 bug），且作为可观测信号
# 词表的占位、别被后续 action 复用冲掉。当前进程内无任何 emit 点。

# 历史 emit: MainLoop._apply_required_action_gate（已删）
# fields: iteration(int), obligation_ids(list[str])
ACTION_OBLIGATION_REPAIR_INJECTED: Final[str] = "obligation_repair_injected"

# 历史 emit: MainLoop._apply_required_action_gate（已删）
# fields: iteration(int), obligation_ids(list[str])
ACTION_OBLIGATION_VIOLATION: Final[str] = "obligation_violation"


# ============================================================
# D-context 事件：ToolOutputStore + ContextBudget
# ============================================================

# emit: MainLoop._run_inner, ToolOutputStore.store_if_needed 落盘成功后
# fields: iteration(int), tool(str), tool_call_id(str), original_tokens(int),
#         preview_tokens(int), full_path(str), reason(str: single_limit | total_limit)
ACTION_TOOL_OUTPUT_SAVED: Final[str] = "tool_output_saved"

# emit: MainLoop._run_inner, ToolOutputStore 决定 truncate 时（落盘并替换 preview 后）
# fields: iteration(int), tool(str), original_tokens(int), preview_tokens(int),
#         reason(str: single_limit | total_limit)
ACTION_TOOL_OUTPUT_TRUNCATED: Final[str] = "tool_output_truncated"

# emit: MainLoop._run_inner, ContextBudget.check_total_context 返回 'warning'
# fields: iteration(int), total_tokens(int), warn_threshold(int)
ACTION_CONTEXT_BUDGET_WARNING: Final[str] = "context_budget_warning"

# emit: MainLoop._run_inner, ContextBudget.check_total_context 返回 'exceeded'
# fields: iteration(int), total_tokens(int), hard_threshold(int)
ACTION_CONTEXT_BUDGET_EXCEEDED: Final[str] = "context_budget_exceeded"


# ============================================================
# PromotionGate 审计事件
# ============================================================

# emit: PromotionGate.sweep 内 apply 成功后（如果 caller 提供 audit_callback 写 trace）
# fields: lesson_id(str), from_status(str), to_status(str), reason(str),
#         evidence_episode_ids(list[str], 截断 ≤10)
# 语义：lesson 状态自动转移的审计记录。当前默认仅写 _logger.info；trace event
# wiring 留给 caller 选择（sweep 时 RunTracer 已 finalize，不能直接写）。
ACTION_LESSON_STATUS_CHANGED: Final[str] = "lesson_status_changed"


# ============================================================
# OutcomeTracker post-save 事件：让 lesson_helped/hurt/ineffective 在 trace
# 中可见，grader 才能量化飞轮真改善
# ============================================================

# emit: Harness._maybe_consume_trace_outcomes 在 OutcomeTracker.process_trace_path
# 后，针对每条 OutcomeUpdate 调 RunTracer.append_post_save 追加到 JSONL
# fields: lesson_id(str), outcome(str: helped|hurt|ineffective),
#         new_confidence(float), new_hit_count(int)
# 注意：emit 时机是 RunTracer.save() 之后——文件已关闭，需用专门的
# append_post_save 路径重新 open(append) 写入再 close。trace 末尾出现
# `outcome_update` 行属正常（grader 全量遍历 events 已支持）。
ACTION_OUTCOME_UPDATE: Final[str] = "outcome_update"


# ============================================================
# 聚合：所有已定义的 action 值
# ============================================================

ALL_ACTIONS: Final[frozenset[str]] = frozenset({
    ACTION_LLM_CALL_START,
    ACTION_LLM_CALL_END,
    ACTION_TOOL_CALL_START,
    ACTION_TOOL_CALL_END,
    ACTION_COVERAGE_CHECK,
    ACTION_COVERAGE_HINT_INJECTED,
    ACTION_FAILURE_RECOVERY_HINT,
    ACTION_STOP_CONDITION_MET,
    ACTION_CIRCUIT_OPEN,
    ACTION_FINISH,
    ACTION_RUN_ERROR,
    ACTION_EVALUATOR_CALL_START,
    ACTION_EVALUATOR_CALL_END,
    ACTION_EVALUATOR_RETRY_TRIGGERED,
    ACTION_LESSON_USED,
    ACTION_REFLECTOR_HINT,
    ACTION_TOOL_CALL_REPAIR_REQUIRED,
    ACTION_OBLIGATION_REPAIR_INJECTED,
    ACTION_OBLIGATION_VIOLATION,
    ACTION_TOOL_OUTPUT_SAVED,
    ACTION_TOOL_OUTPUT_TRUNCATED,
    ACTION_CONTEXT_BUDGET_WARNING,
    ACTION_CONTEXT_BUDGET_EXCEEDED,
    ACTION_LESSON_STATUS_CHANGED,
    ACTION_OUTCOME_UPDATE,
})


def is_known_action(action: str) -> bool:
    """consumer 可用来校验 trace 里的 action 名在本 schema 里。

    不在 `ALL_ACTIONS` 里的 action 要么是过时 trace（旧版本 log）要么拼写错误，
    consumer 应单独处理或丢弃。
    """
    return action in ALL_ACTIONS
