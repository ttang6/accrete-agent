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
    success = finish_reason == "finish"
    coverage_missing = _extract_last_coverage_missing(last_events)
    final_answer_chars = _extract_final_answer_chars(last_events)

    # 累加全部 trace 的计数指标
    def _sum_action(action: str) -> int:
        return sum(_count_action(ev, action) for _, ev in valid)

    def _sum_outcome(outcome: str) -> int:
        return sum(_count_outcome(ev, outcome) for _, ev in valid)

    # expected_tool_calls 用所有 events union 匹配（任一 trace 命中即算 hit）
    union_events: List[Dict[str, Any]] = [e for _, ev in valid for e in ev]

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
        obligations_violations=_sum_action(ts.ACTION_OBLIGATION_VIOLATION),
        expected_tool_calls_hit=_count_expected_tool_calls(
            union_events, spec.expected_tool_calls
        ),
        expected_tool_calls_required=len(spec.expected_tool_calls),
        lesson_uses=_sum_action(ts.ACTION_LESSON_USED),
        lesson_helped=_sum_outcome("helped"),
        lesson_hurt=_sum_outcome("hurt"),
        lesson_ineffective=_sum_outcome("ineffective"),
        coverage_missing=coverage_missing,
        final_answer_chars=final_answer_chars,
        duration_ms=sum(_extract_duration(ev) for _, ev in valid),
        trace_path=str(last_path),
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
    """命中 spec.expected_tool_calls 模式的去重计数（每模式至少 1 次算 1 命中）。

    模式字段：
    - tool: 必填，匹配 trace event 的 tool 字段
    - skill / script: 可选，对 skill_exec 调用解 input JSON 比对
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


def _extract_final_answer_chars(events: List[Dict[str, Any]]) -> int:
    """ACTION_FINISH event 的 output 字段长度（trace 里截断到 500，所以上限 500）。

    >500 答案在 trace 里看不到完整长度。eval 用 500 作为"内容相关"信号阈值即可。
    """
    finish_events = [e for e in events if e.get("action") == ts.ACTION_FINISH]
    if not finish_events:
        return 0
    return len(finish_events[-1].get("output") or "")


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
