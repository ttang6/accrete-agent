"""TraceGrader — 单 trace JSONL → TaskScore。

纯函数，不调外部服务（不依赖 sqlite / LLM）。所有信号都从 trace 抽，
P0.2 ACTION_OUTCOME_UPDATE 事件让 lesson helped/hurt/ineffective 直接可见。

EpisodeExtractor 用 events[1:-1] 切 body（假设 summary 在末尾），但本 grader
直接全量遍历 events（含 header / summary / 后追加的 outcome_update），按
action 字段过滤——更鲁棒。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from nanoagent.core import trace_schema as ts
from nanoagent.runtime.tool_failure import _is_tool_failure

from evals.specs import TaskScore, TaskSpec


def grade_trace(
    spec: TaskSpec, trace_paths: Union[Path, List[Path]]
) -> TaskScore:
    """读 trace JSONL → TaskScore。

    trace_paths：
    - 单 Path（mock / single-turn）→ 当 [Path] 处理
    - List[Path]（multi-turn）→ 累加多份 trace 的信号

    累加策略：
    - 计数指标（lesson_uses/helped/hurt/ineffective、tool_calls/failures、
      obligation_violations/repair_injected、llm_calls、total_steps、duration_ms）
      → 全 trace 求和（多轮里飞轮在前轮命中也不丢）
    - expected_tool_calls：所有 trace events union 后匹配（任一 trace 命中即算 hit）
    - finish_reason / success / coverage_missing / final_answer_chars
      → 用最后一份 trace（task 最终状态）
    - trace_path 字段：最后一份（debug 入口）

    全部 trace 都读不到 / 损坏 → finish_reason="error"。
    """
    if isinstance(trace_paths, Path):
        trace_paths = [trace_paths]
    if not trace_paths:
        return _empty_score(spec, "", reason="error")

    # 读全部 trace，过滤掉损坏的
    valid: List[tuple[Path, List[Dict[str, Any]]]] = []
    for tp in trace_paths:
        events = _read_events(tp)
        if events is not None:
            valid.append((tp, events))
    if not valid:
        return _empty_score(spec, str(trace_paths[-1]), reason="error")

    last_path, last_events = valid[-1]

    # 最后一份 trace 决定 task 终态
    finish_reason = _extract_finish_reason(last_events)
    coverage_missing = _extract_last_coverage_missing(last_events)
    final_answer_text = _extract_final_answer_text(last_events)
    final_answer_chars = len(final_answer_text)

    # 累加全部 trace 的计数指标
    def _sum_action(action: str) -> int:
        return sum(_count_action(ev, action) for _, ev in valid)

    def _sum_outcome(outcome: str) -> int:
        return sum(_count_outcome(ev, outcome) for _, ev in valid)

    # expected_tool_calls 用所有 events union 匹配（任一 trace 命中即算 hit）
    union_events: List[Dict[str, Any]] = [e for _, ev in valid for e in ev]

    obligations_violations = _sum_action(ts.ACTION_OBLIGATION_VIOLATION)
    expected_tool_calls_hit = _count_expected_tool_calls(
        union_events, spec.expected_tool_calls
    )
    expected_tool_calls_required = len(spec.expected_tool_calls)

    # end-state scoring（看产物，不看路径）：success 不再只看"是否正常结束"，
    # 而是联合已有过程指标——必须正常 finish + 声明的 coverage 全达标 +
    # obligation 无违约 + 声明的必需能力全部成功命中。指标集不变，只是把过程
    # 指标提为成败门槛（doc §6 部分信用/看产物）。expected_* 为空的字段不约束。
    # H-2 防护（盲点研究 §H）：声明了 success_keywords 时，终答须命中至少一个才算成功
    # ——堵"空 action=mark 也能 finish+hit 但没真产出"的漏洞（看产物不看路径）。
    # 例外：R4_unrecoverable / observed_empty 的成功 = 优雅诚实收尾，不该靠精确关键词
    # 卡死（措辞太散，pilot 实测误杀了 finish 正常的降级卷）；这两类主信号本就是
    # finish_reason / 步数，故跳过关键词门。trace 终答截断到 500 字。
    if spec.recovery_type in ("R4_unrecoverable", "observed_empty"):
        keyword_ok = True
    else:
        keyword_ok = (not spec.success_keywords) or any(
            kw in final_answer_text for kw in spec.success_keywords
        )
    success = (
        finish_reason == "finish"
        and not coverage_missing
        and obligations_violations == 0
        and expected_tool_calls_hit == expected_tool_calls_required
        and keyword_ok
    )

    return TaskScore(
        task_id=spec.id,
        success=success,
        finish_reason=finish_reason,
        total_steps=sum(_extract_total_steps(ev) for _, ev in valid),
        llm_calls=_sum_action(ts.ACTION_LLM_CALL_END),
        tool_calls=_sum_action(ts.ACTION_TOOL_CALL_END),
        tool_failures=sum(_count_tool_failures(ev) for _, ev in valid),
        obligations_required=len(spec.expected_obligations),
        obligations_repair_injected=_sum_action(ts.ACTION_OBLIGATION_REPAIR_INJECTED),
        obligations_violations=obligations_violations,
        expected_tool_calls_hit=expected_tool_calls_hit,
        expected_tool_calls_required=expected_tool_calls_required,
        lesson_uses=_sum_action(ts.ACTION_LESSON_USED),
        lesson_helped=_sum_outcome("helped"),
        lesson_hurt=_sum_outcome("hurt"),
        lesson_ineffective=_sum_outcome("ineffective"),
        coverage_missing=coverage_missing,
        final_answer_chars=final_answer_chars,
        duration_ms=sum(_extract_duration(ev) for _, ev in valid),
        trace_path=str(last_path),
        total_tokens=sum(_extract_tokens(ev) for _, ev in valid),
        recovery_type=spec.recovery_type,
    )


# ============================================================
# Internal helpers
# ============================================================


def _read_events(trace_path: Path) -> Optional[List[Dict[str, Any]]]:
    """读 trace JSONL。所有损坏行跳过。"""
    if not trace_path.exists():
        return None
    events: List[Dict[str, Any]] = []
    try:
        with open(trace_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return None
    return events


def _empty_score(spec: TaskSpec, trace_path: str, reason: str) -> TaskScore:
    return TaskScore(
        task_id=spec.id,
        success=False,
        finish_reason=reason,
        total_steps=0,
        llm_calls=0,
        tool_calls=0,
        tool_failures=0,
        obligations_required=len(spec.expected_obligations),
        obligations_repair_injected=0,
        obligations_violations=0,
        expected_tool_calls_hit=0,
        expected_tool_calls_required=len(spec.expected_tool_calls),
        lesson_uses=0,
        lesson_helped=0,
        lesson_hurt=0,
        lesson_ineffective=0,
        coverage_missing=[],
        final_answer_chars=0,
        duration_ms=0,
        trace_path=trace_path,
        recovery_type=spec.recovery_type,
    )


def _extract_finish_reason(events: List[Dict[str, Any]]) -> str:
    """决定 task 终止原因：finish (正常) / max_iter / error / timeout / running。"""
    has_run_error = any(e.get("action") == ts.ACTION_RUN_ERROR for e in events)
    if has_run_error:
        return "error"
    finish_events = [e for e in events if e.get("action") == ts.ACTION_FINISH]
    if not finish_events:
        return "running"  # 没 finish event 说明跑半截被杀（subprocess timeout）
    last_finish = finish_events[-1]
    reason = last_finish.get("reason")
    if reason == "max_iterations":
        return "max_iter"
    if reason and reason != "":
        return reason  # stop_condition_met 等
    return "finish"


def _extract_total_steps(events: List[Dict[str, Any]]) -> int:
    """优先 run_summary.total_steps；缺则按 step 字段最大值兜底。"""
    summary = next(
        (e for e in events if e.get("type") == "run_summary"),
        None,
    )
    if summary and "total_steps" in summary:
        try:
            return int(summary["total_steps"])
        except (TypeError, ValueError):
            pass
    step_nums = [int(e["step"]) for e in events if isinstance(e.get("step"), int)]
    return max(step_nums) if step_nums else 0


def _count_action(events: List[Dict[str, Any]], action: str) -> int:
    return sum(1 for e in events if e.get("action") == action)


def _count_tool_failures(events: List[Dict[str, Any]]) -> int:
    """tool_call_end output 命中失败签名的次数。"""
    n = 0
    for e in events:
        if e.get("action") != ts.ACTION_TOOL_CALL_END:
            continue
        out = e.get("output") or ""
        if _is_tool_failure(out):
            n += 1
    return n


def _count_expected_tool_calls(
    events: List[Dict[str, Any]], expected: List[Dict[str, Any]]
) -> int:
    """命中 spec.expected_tool_calls 模式的去重计数（每模式至少 1 次**成功**调用算 1 命中）。

    模式字段：
    - tool: 必填，匹配 trace event 的 tool 字段
    - skill / script: 可选，对 skill_exec 调用解 input JSON 比对

    "成功"判定（end-state scoring，看产物）：output 命中失败签名的 tool_call_end
    不算命中。这样"必需能力"必须**至少成功执行一次**——
    - 堵 route-around：agent 用 fetch 绕过 fetch_github，则 fetch_github 永不命中；
    - 逼真正恢复：注入只 fail 首次调用，agent 必须重试拿到一次成功调用才算命中。
    被注入打掉的那次失败调用本身不计入命中。
    """
    if not expected:
        return 0
    hit_count = 0
    for pattern in expected:
        tool = pattern.get("tool")
        if not tool:
            continue
        skill = pattern.get("skill")
        script = pattern.get("script")
        for e in events:
            if e.get("action") != ts.ACTION_TOOL_CALL_END:
                continue
            if e.get("tool") != tool:
                continue
            if _is_tool_failure(e.get("output") or ""):
                continue  # 失败调用不算"成功命中"
            if skill is None and script is None:
                hit_count += 1
                break
            inp = e.get("input") or ""
            try:
                inp_args = json.loads(inp) if inp else {}
            except (json.JSONDecodeError, ValueError):
                inp_args = {}
            if skill is not None and inp_args.get("skill") != skill:
                continue
            if script is not None and inp_args.get("script") != script:
                continue
            hit_count += 1
            break
    return hit_count


def _count_outcome(events: List[Dict[str, Any]], outcome: str) -> int:
    """ACTION_OUTCOME_UPDATE 中 outcome 字段命中的次数。"""
    return sum(
        1
        for e in events
        if e.get("action") == ts.ACTION_OUTCOME_UPDATE and e.get("outcome") == outcome
    )


def _extract_last_coverage_missing(events: List[Dict[str, Any]]) -> List[str]:
    """trace 里最后一条 coverage_check 的 missing categories。空 list = 全达标。"""
    coverage_checks = [
        e for e in events if e.get("action") == ts.ACTION_COVERAGE_CHECK
    ]
    if not coverage_checks:
        return []
    last = coverage_checks[-1]
    missing = last.get("missing") or []
    return [str(m) for m in missing if isinstance(m, (str, int))]


def _extract_final_answer_text(events: List[Dict[str, Any]]) -> str:
    """ACTION_FINISH event 的 output 文本（trace 里截断到 500 字）。

    供 final_answer_chars（取 len）与 success_keywords 命中判定共用。
    >500 字的答案 trace 里只到 500，关键词须落在前 500 字内。
    """
    finish_events = [e for e in events if e.get("action") == ts.ACTION_FINISH]
    if not finish_events:
        return ""
    return str(finish_events[-1].get("output") or "")


def _extract_tokens(events: List[Dict[str, Any]]) -> int:
    """run_summary.tokens（本 trace 的 token 总量）；缺则 0。"""
    summary = next((e for e in events if e.get("type") == "run_summary"), None)
    if summary and "tokens" in summary:
        try:
            return int(summary["tokens"])
        except (TypeError, ValueError):
            return 0
    return 0


def _extract_duration(events: List[Dict[str, Any]]) -> int:
    """run_summary.total_duration_ms；缺则 sum(step.duration_ms) 兜底。"""
    summary = next(
        (e for e in events if e.get("type") == "run_summary"),
        None,
    )
    if summary and "total_duration_ms" in summary:
        try:
            return int(summary["total_duration_ms"])
        except (TypeError, ValueError):
            pass
    return sum(int(e.get("duration_ms", 0)) for e in events if isinstance(e.get("duration_ms"), int))
