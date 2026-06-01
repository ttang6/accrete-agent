"""EpisodeExtractor — trace JSONL → RuntimeEpisode（飞轮管道入口）。

读 `data/logs/traces/<date>/<runner>_HHMMSS.jsonl`，单遍扫，配对
tool_call_start/end 的 args；用 `runtime/tool_failure` SSoT 的失败签名
判定 tool 失败（共用 SSoT 而非各自重声明，防签名 drift）。

输出：
- 无 failure 的 trace → None（无须建 episode）
- 有 failure → RuntimeEpisode，failures 列表至少 1 条

failure 来源：
1. tool_call_end 且 output 命中 `_FAILURE_SIGNATURES` → tool 调用失败
2. coverage_hint_injected → 硬覆盖不达标，error_type="coverage_gap"
3. evaluator_call_end 且 recommended_action != "finalize"（或 coverage_ok=False 或
   soft_issues 非空）→ 副 LLM 判失败，error_type="soft_quality"
4. failure_recovery_hint：本身不算独立 failure，把 hint 信息附到最近同
   iteration 的 tool failure 的 extras["recovery_hint"]
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nanoagent.core.logger import get_logger
from nanoagent.core.trace_schema import (
    ACTION_COVERAGE_CHECK,
    ACTION_COVERAGE_HINT_INJECTED,
    ACTION_EVALUATOR_CALL_END,
    ACTION_FAILURE_RECOVERY_HINT,
    ACTION_TOOL_CALL_END,
    ACTION_TOOL_CALL_REPAIR_REQUIRED,
    ACTION_TOOL_CALL_START,
)
from nanoagent.evolution.runtime_memory.schema import FailureEvent, RuntimeEpisode

# 失败签名 / 分类 / args_hash 三个常量与函数搬到 SSoT 模块。re-export 保持
# 旧 import 路径（`from ...episode_extractor import _is_tool_failure` 等）。
# 物理上 `episode_extractor._FAILURE_SIGNATURES is failure_memory._FAILURE_SIGNATURES`
# 即可证明 drift 不可能再发生（曾因 _SCHEMA_MISMATCH_HINTS 漏 "action" 项与
# failure_memory drift，踩过此类 bug）。详见 runtime/tool_failure.py。
from nanoagent.runtime.tool_failure import (  # noqa: F401
    _FAILURE_SIGNATURES,
    _SCHEMA_MISMATCH_PATTERNS,
    _TRANSIENT_PATTERNS,
    _canonical_args_hash,
    _is_tool_failure,
)
from nanoagent.runtime.tool_failure import (
    _classify_tool_failure as _classify_tool_error,  # 历史命名保留
)

_logger = get_logger("episode_extractor")


def _build_tool_key(tool: str, raw_args: str) -> str:
    """构造 (tool_name, args) → tool_key。

    skill_exec 特殊：含 skill / script 拼到 key 前缀，便于跨 trace 关联同一 script。
    其他 tool 直接用 tool 名。
    """
    if tool != "skill_exec":
        return tool
    try:
        parsed = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        return tool
    if not isinstance(parsed, dict):
        return tool
    skill = parsed.get("skill", "?")
    script = parsed.get("script", "?")
    return f"skill_exec:{skill}/{script}"


# ============================================================
# Extractor
# ============================================================


class EpisodeExtractor:
    """trace JSONL → RuntimeEpisode。无 failure → None。"""

    def extract_from_file(self, trace_path: Path) -> Optional[RuntimeEpisode]:
        if not trace_path.exists():
            raise FileNotFoundError(f"trace 不存在: {trace_path}")
        events = self._read_jsonl(trace_path)
        return self.extract_from_events(events, str(trace_path))

    def extract_from_events(
        self, events: List[Dict[str, Any]], trace_path: str
    ) -> Optional[RuntimeEpisode]:
        if not events:
            _logger.warning(f"[episode] empty events: {trace_path}")
            return None

        header = events[0] if events[0].get("type") == "run_header" else {}
        summary = (
            events[-1] if events and events[-1].get("type") == "run_summary" else {}
        )
        body_start = 1 if header else 0
        body_end = -1 if summary else None
        body = events[body_start:body_end]

        failures: List[FailureEvent] = []
        coverage_hits: List[Dict[str, Any]] = []
        evaluator_verdicts: List[Dict[str, Any]] = []

        # iteration -> 最近一次 tool_call_start，用于配对 tool_call_end 拿 args
        last_call_by_iter: Dict[Tuple[int, str], Dict[str, Any]] = {}
        # (iteration, tool) -> 结构化 repair_example dict（main_loop
        # _apply_repair_gate 已抽好放进 ACTION_TOOL_CALL_REPAIR_REQUIRED 事件）。
        # tool_call_end 的 output 含同源的 example_call JSON 块，但截断/格式
        # 漂移会破坏文本重解析；从专用 trace 事件直接读 dict 最稳。
        repair_example_by_iter: Dict[Tuple[int, str], Dict[str, Any]] = {}

        for ev in body:
            action = ev.get("action")
            if action == ACTION_TOOL_CALL_START:
                last_call_by_iter[(ev.get("iteration", -1), ev.get("tool", ""))] = ev
                continue

            if action == ACTION_TOOL_CALL_REPAIR_REQUIRED:
                example = ev.get("repair_example")
                if isinstance(example, dict):
                    repair_example_by_iter[
                        (ev.get("iteration", -1), ev.get("tool", ""))
                    ] = example
                continue

            if action == ACTION_TOOL_CALL_END:
                output = ev.get("output", "") or ""
                if _is_tool_failure(output):
                    fe = _build_tool_failure(
                        ev, last_call_by_iter, repair_example_by_iter
                    )
                    failures.append(fe)
                continue

            if action == ACTION_COVERAGE_CHECK:
                coverage_hits.append(dict(ev))
                continue

            if action == ACTION_COVERAGE_HINT_INJECTED:
                failures.append(_build_coverage_failure(ev))
                continue

            if action == ACTION_FAILURE_RECOVERY_HINT:
                _attach_recovery_hint(failures, ev)
                continue

            if action == ACTION_EVALUATOR_CALL_END:
                evaluator_verdicts.append(dict(ev))
                if _evaluator_failed(ev):
                    failures.append(_build_evaluator_failure(ev))
                continue

        if not failures:
            return None

        episode_id = _episode_id(trace_path, header.get("started_at", ""))
        return RuntimeEpisode(
            episode_id=episode_id,
            trace_path=trace_path,
            agent_name=header.get("agent", "unknown"),
            user_input=(header.get("user_input", "") or "")[:500],
            started_at=header.get("started_at", ""),
            finished_at=summary.get("ended_at", ""),
            stop_reason=_extract_stop_reason(body),
            total_steps=int(summary.get("total_steps", len(body))),
            failures=failures,
            coverage_hits=coverage_hits,
            evaluator_verdicts=evaluator_verdicts,
            raw_summary=dict(summary),
        )

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError as e:
                    _logger.warning(
                        f"[episode] 跳过损坏行 {path.name}:{lineno}: {e}"
                    )
        return events


# ============================================================
# Builder helpers
# ============================================================


def _build_tool_failure(
    end_ev: Dict[str, Any],
    last_call_by_iter: Dict[Tuple[int, str], Dict[str, Any]],
    repair_example_by_iter: Optional[
        Dict[Tuple[int, str], Dict[str, Any]]
    ] = None,
) -> FailureEvent:
    iteration = end_ev.get("iteration", -1)
    tool = end_ev.get("tool", "unknown")
    raw_args = end_ev.get("input", "") or ""
    # 当前 trace 格式 tool_call_end 自带 input；为防未来精简事件 schema，
    # end.input 缺时回退到 start 事件的 input（last_call_by_iter 缓存）
    if not raw_args:
        start_ev = last_call_by_iter.get((iteration, tool))
        if start_ev:
            raw_args = start_ev.get("input", "") or ""
    output = end_ev.get("output", "") or ""
    extras: Dict[str, Any] = {"raw_input": raw_args[:200]}
    # 结构化 repair_example dict 经 extras 流到 LessonGenerator →
    # LessonEvidence.repair_example + RuntimeLesson.suggested_action
    if repair_example_by_iter is not None:
        example = repair_example_by_iter.get((iteration, tool))
        if isinstance(example, dict):
            extras["repair_example"] = example
    # 通用 semantic_failures pass-through：
    # 任何 skill script 输出 JSON 含顶级 `semantic_failures: [...]` 字段时，
    # 抽进 extras 给 LessonGenerator。framework 不识别 failure_type 的具体值，
    # 也不绑死任何 skill—— 任何 skill 输出此 schema 都受益。
    sf = _extract_semantic_failures(output)
    if sf:
        extras["semantic_failures"] = sf
    return FailureEvent(
        iteration=iteration,
        tool_key=_build_tool_key(tool, raw_args),
        args_hash=_canonical_args_hash(raw_args),
        error_type=_classify_tool_error(output),
        error_message=output[:300],
        timestamp=end_ev.get("timestamp", ""),
        raw_action=ACTION_TOOL_CALL_END,
        extras=extras,
    )


def _extract_semantic_failures(output_text: str) -> List[Dict[str, Any]]:
    """从 tool_output 文本中抽 `semantic_failures` 列表。

    任意 skill script 可在 stdout 输出 JSON 含 `semantic_failures: [...]` 标记
    软失败（质量/语义问题）。前缀可能有 `[error] ...` 行让 _is_tool_failure
    命中；JSON 体从第一个 `{` 开始。无 JSON / 解析失败 / 字段缺失 → 返回空 list。

    framework 仅做 pass-through——不识别 failure_type 的具体值，由 LessonGenerator
    决定如何按 type 模板化（保 framework 不知 skill 知识的边界）。
    """
    if not output_text:
        return []
    brace_idx = output_text.find("{")
    if brace_idx < 0:
        return []
    try:
        parsed = json.loads(output_text[brace_idx:])
    except (json.JSONDecodeError, ValueError):
        return []
    if not isinstance(parsed, dict):
        return []
    sf = parsed.get("semantic_failures")
    if not isinstance(sf, list):
        return []
    # 只保留 dict 元素 + 截前 20 条防 trace 膨胀（lesson 也不需要无限细粒度）
    return [item for item in sf if isinstance(item, dict)][:20]


def _build_coverage_failure(ev: Dict[str, Any]) -> FailureEvent:
    missing = ev.get("missing", [])
    counts = ev.get("counts", {})
    msg = f"coverage missing categories={missing} counts={counts}"
    return FailureEvent(
        iteration=ev.get("iteration", -1),
        tool_key="",
        args_hash="",
        error_type="coverage_gap",
        error_message=msg[:300],
        timestamp=ev.get("timestamp", ""),
        raw_action=ACTION_COVERAGE_HINT_INJECTED,
        extras={"missing": list(missing), "counts": dict(counts)},
    )


def _evaluator_failed(ev: Dict[str, Any]) -> bool:
    """副 LLM 判定不达标：recommended_action != finalize 或 coverage_ok=False 或
    soft_issues 非空（任一即视为本 episode 的 soft_quality failure）。"""
    if ev.get("recommended_action", "finalize") != "finalize":
        return True
    if ev.get("coverage_ok") is False:
        return True
    if ev.get("soft_issues"):
        return True
    return False


def _build_evaluator_failure(ev: Dict[str, Any]) -> FailureEvent:
    reason = ev.get("reason", "") or ""
    msg = (
        f"evaluator recommended={ev.get('recommended_action', '?')} "
        f"missing={ev.get('missing', [])} soft_issues={ev.get('soft_issues', [])} "
        f"reason={reason}"
    )
    return FailureEvent(
        iteration=ev.get("attempt", -1),  # evaluator 用 attempt 字段
        tool_key="",
        args_hash="",
        error_type="soft_quality",
        error_message=msg[:300],
        timestamp=ev.get("timestamp", ""),
        raw_action=ACTION_EVALUATOR_CALL_END,
        extras={
            "recommended_action": ev.get("recommended_action"),
            "missing": list(ev.get("missing", [])),
            "soft_issues": list(ev.get("soft_issues", [])),
            "fail_open": ev.get("fail_open"),
            "coverage_ok": ev.get("coverage_ok"),
            "reason": reason,  # 给 LessonGenerator soft_quality_issue 模板用
        },
    )


def _attach_recovery_hint(
    failures: List[FailureEvent], ev: Dict[str, Any]
) -> None:
    """把 failure_recovery_hint 信息附到最近同 iteration 的 tool failure。

    若没有匹配（首条 hint 在 tool failure 前，或 iteration 不对应）则忽略——
    augment 总是 tool_call_end 之后才发，正常 trace 一定能匹配。
    """
    iteration = ev.get("iteration", -1)
    for fe in reversed(failures):
        if fe.iteration != iteration:
            continue
        if fe.raw_action != ACTION_TOOL_CALL_END:
            continue
        fe.extras["recovery_hint"] = {
            "failure_count": ev.get("failure_count"),
            "last_error_type": ev.get("last_error_type"),
            "suggested_next_action": ev.get("suggested_next_action"),
        }
        return


def _extract_stop_reason(body: List[Dict[str, Any]]) -> Optional[str]:
    for ev in body:
        if ev.get("action") == "stop_condition_met":
            return ev.get("reason")
    return None


def _episode_id(trace_path: str, started_at: str) -> str:
    payload = f"{trace_path}|{started_at}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
