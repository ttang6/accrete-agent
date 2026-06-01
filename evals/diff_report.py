"""Baseline vs Evolved EvalReport 对比生成器。

输出两份：
- Markdown 报告：summary 表（含 Δ 列）+ task 分类（改善 / 回归 / 不变）+ per-task 明细
- CSV：每行一个 task_id，列名带 baseline_/evolved_ 前缀，方便后续 pandas / Excel 切片

设计：
- 纯函数无副作用，直接写文件；不修改 EvalReport 或 TaskScore
- 缺失 task（一边有一边没有）容忍：标记 missing_in_baseline / missing_in_evolved
- 数值 Δ：evolved - baseline；百分比保留 4 位精度，整数显示原样
- 不在内部 print/log——caller 决定怎么提示

简历叙事关键 metric（要在 markdown summary 里显眼）：
- success_rate
- expected_tool_calls_coverage_rate
- obligation_completion_rate_avg
- lesson_help_rate / lesson_hurt_rate / lesson_ineffective_rate
- avg_steps_per_task / tool_failure_rate
"""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from evals.specs import EvalReport


# ============================================================
# Summary diff
# ============================================================


def diff_summary(
    baseline: EvalReport, evolved: EvalReport
) -> Dict[str, Dict[str, Any]]:
    """两份 summary 的逐 key Δ。

    返回 {metric_key: {"baseline": v1, "evolved": v2, "delta": v2-v1 | None}}。
    delta=None 当 metric 不是数值（dist / list / per_task）。

    包括：所有数值 metric + finish_reason_dist 双侧 + coverage_missing 双侧。
    per_task 不进 summary diff（走 per_task_comparison）。
    """
    bs = baseline.summary
    es = evolved.summary
    out: Dict[str, Dict[str, Any]] = {}

    for key in sorted(set(bs.keys()) | set(es.keys())):
        if key == "per_task":
            continue
        b_val = bs.get(key)
        e_val = es.get(key)
        delta: Any = None
        if isinstance(b_val, (int, float)) and isinstance(e_val, (int, float)):
            delta = e_val - b_val
        out[key] = {"baseline": b_val, "evolved": e_val, "delta": delta}
    return out


# ============================================================
# Per-task diff
# ============================================================


def per_task_comparison(
    baseline: EvalReport, evolved: EvalReport
) -> List[Dict[str, Any]]:
    """task_id 对齐 → 每个 task 的 baseline / evolved 行 + 分类标签。

    分类（status 字段）：
    - both_succeeded / both_failed
    - improved (False → True)
    - regressed (True → False)
    - missing_in_baseline / missing_in_evolved
    """
    b_rows = {row["task_id"]: row for row in baseline.summary.get("per_task", [])}
    e_rows = {row["task_id"]: row for row in evolved.summary.get("per_task", [])}
    all_ids = sorted(set(b_rows.keys()) | set(e_rows.keys()))

    rows: List[Dict[str, Any]] = []
    for tid in all_ids:
        b = b_rows.get(tid)
        e = e_rows.get(tid)
        rows.append({
            "task_id": tid,
            "baseline": b,
            "evolved": e,
            "status": _classify_task(b, e),
            "step_delta": _safe_delta(b, e, "total_steps"),
            "duration_delta_ms": _safe_delta(b, e, "duration_ms"),
            "lesson_helped_delta": _safe_delta(b, e, "lesson_helped"),
            "lesson_hurt_delta": _safe_delta(b, e, "lesson_hurt"),
            "lesson_ineffective_delta": _safe_delta(b, e, "lesson_ineffective"),
        })
    return rows


def _classify_task(b: Optional[Dict], e: Optional[Dict]) -> str:
    if b is None and e is None:
        return "unknown"
    if b is None:
        return "missing_in_baseline"
    if e is None:
        return "missing_in_evolved"
    bs = bool(b.get("success"))
    es = bool(e.get("success"))
    if bs and es:
        return "both_succeeded"
    if not bs and not es:
        return "both_failed"
    if not bs and es:
        return "improved"
    return "regressed"


def _safe_delta(
    b: Optional[Dict], e: Optional[Dict], key: str
) -> Optional[float]:
    if b is None or e is None:
        return None
    bv, ev = b.get(key), e.get(key)
    if isinstance(bv, (int, float)) and isinstance(ev, (int, float)):
        return ev - bv
    return None


# ============================================================
# Markdown report
# ============================================================


# 简历叙事核心 metric——markdown summary 表里高亮显示
_HEADLINE_METRICS: Tuple[str, ...] = (
    "success_rate",
    "expected_tool_calls_coverage_rate",
    "obligation_completion_rate_avg",
    "lesson_use_rate",
    "lesson_help_rate",
    "lesson_hurt_rate",
    "lesson_ineffective_rate",
    "tool_failure_rate",
    "avg_steps_per_task",
    "avg_duration_ms_per_task",
    "avg_final_answer_chars",
)


def write_markdown_report(
    baseline: EvalReport, evolved: EvalReport, out_path: Path
) -> None:
    """生成 markdown 报告：headline summary + task 分类 + per-task 明细。"""
    diff = diff_summary(baseline, evolved)
    rows = per_task_comparison(baseline, evolved)
    status_counts = _count_statuses(rows)

    lines: List[str] = []
    lines.append("# Eval Report — baseline vs evolved")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append("")
    lines.append(f"- Baseline: {baseline.config.mode} (lesson_recall={baseline.config.enable_lesson_recall}, promotion_gate={baseline.config.enable_promotion_gate})")
    lines.append(f"- Evolved: {evolved.config.mode} (lesson_recall={evolved.config.enable_lesson_recall}, promotion_gate={evolved.config.enable_promotion_gate})")
    lines.append(f"- Tasks: {baseline.total_tasks} baseline / {evolved.total_tasks} evolved")
    lines.append("")

    # Headline summary
    lines.append("## Headline Metrics")
    lines.append("")
    lines.append("| Metric | Baseline | Evolved | Δ |")
    lines.append("|---|---:|---:|---:|")
    for key in _HEADLINE_METRICS:
        if key not in diff:
            continue
        d = diff[key]
        lines.append(
            f"| {key} | {_fmt(d['baseline'])} | {_fmt(d['evolved'])} | {_fmt_delta(d['delta'])} |"
        )
    lines.append("")

    # Task status 分类
    lines.append("## Task Status Distribution")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---:|")
    for status in (
        "both_succeeded",
        "improved",
        "regressed",
        "both_failed",
        "missing_in_baseline",
        "missing_in_evolved",
    ):
        c = status_counts.get(status, 0)
        if c > 0:
            lines.append(f"| {status} | {c} |")
    lines.append("")

    # Lesson 飞轮
    lines.append("## Lesson Flywheel Signals")
    lines.append("")
    lines.append(
        "| Signal | Baseline | Evolved | Δ |\n|---|---:|---:|---:|"
    )
    for key in (
        "total_lesson_uses",
        "total_lesson_helped",
        "total_lesson_hurt",
        "total_lesson_ineffective",
        "total_obligations_required",
        "total_obligations_violations",
    ):
        if key not in diff:
            continue
        d = diff[key]
        lines.append(
            f"| {key} | {_fmt(d['baseline'])} | {_fmt(d['evolved'])} | {_fmt_delta(d['delta'])} |"
        )
    lines.append("")

    # Per-task 明细
    lines.append("## Per-Task Comparison")
    lines.append("")
    lines.append(
        "| task_id | status | b.success | e.success | b.steps | e.steps | step Δ | b.lesson_helped | e.lesson_helped |"
    )
    lines.append(
        "|---|---|---|---|---:|---:|---:|---:|---:|"
    )
    for row in rows:
        b = row["baseline"] or {}
        e = row["evolved"] or {}
        lines.append(
            f"| {row['task_id']} | {row['status']} | "
            f"{_yn(b.get('success'))} | {_yn(e.get('success'))} | "
            f"{_fmt(b.get('total_steps'))} | {_fmt(e.get('total_steps'))} | "
            f"{_fmt_delta(row['step_delta'])} | "
            f"{_fmt(b.get('lesson_helped'))} | {_fmt(e.get('lesson_helped'))} |"
        )
    lines.append("")

    # 完整 finish_reason 分布
    if "finish_reason_dist" in diff:
        lines.append("## Finish Reason Distribution")
        lines.append("")
        b_dist = diff["finish_reason_dist"]["baseline"] or {}
        e_dist = diff["finish_reason_dist"]["evolved"] or {}
        all_reasons = sorted(set(b_dist.keys()) | set(e_dist.keys()))
        lines.append("| Reason | Baseline | Evolved |")
        lines.append("|---|---:|---:|")
        for r in all_reasons:
            lines.append(f"| {r} | {b_dist.get(r, 0)} | {e_dist.get(r, 0)} |")
        lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _count_statuses(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        s = r["status"]
        counts[s] = counts.get(s, 0) + 1
    return counts


def _fmt(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.4f}" if abs(v) < 100 else f"{v:.1f}"
    if isinstance(v, bool):
        return "✓" if v else "✗"
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return str(v)


def _fmt_delta(v: Any) -> str:
    if v is None:
        return "—"
    if isinstance(v, (int, float)):
        if v == 0:
            return "0"
        sign = "+" if v > 0 else ""
        if isinstance(v, float):
            return f"{sign}{v:.4f}" if abs(v) < 100 else f"{sign}{v:.1f}"
        return f"{sign}{v}"
    return str(v)


def _yn(v: Any) -> str:
    if v is True:
        return "✓"
    if v is False:
        return "✗"
    return "—"


# ============================================================
# CSV report
# ============================================================


_CSV_TASK_FIELDS: Tuple[str, ...] = (
    "success",
    "finish_reason",
    "total_steps",
    "tool_calls",
    "tool_failures",
    "obligations_required",
    "obligations_violations",
    "obligation_completion_rate",
    "expected_tool_calls_required",
    "expected_tool_calls_hit",
    "expected_tool_calls_coverage",
    "lesson_uses",
    "lesson_helped",
    "lesson_hurt",
    "lesson_ineffective",
    "final_answer_chars",
    "duration_ms",
)


def write_csv_report(
    baseline: EvalReport, evolved: EvalReport, out_path: Path
) -> None:
    """每行一 task_id，列名 baseline_<field> / evolved_<field>。"""
    rows = per_task_comparison(baseline, evolved)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames: List[str] = ["task_id", "status"]
    fieldnames += [f"baseline_{f}" for f in _CSV_TASK_FIELDS]
    fieldnames += [f"evolved_{f}" for f in _CSV_TASK_FIELDS]
    fieldnames += ["step_delta", "duration_delta_ms", "lesson_helped_delta"]

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            b = row["baseline"] or {}
            e = row["evolved"] or {}
            csv_row: Dict[str, Any] = {
                "task_id": row["task_id"],
                "status": row["status"],
                "step_delta": row.get("step_delta"),
                "duration_delta_ms": row.get("duration_delta_ms"),
                "lesson_helped_delta": row.get("lesson_helped_delta"),
            }
            for f_name in _CSV_TASK_FIELDS:
                csv_row[f"baseline_{f_name}"] = _csv_val(b.get(f_name))
                csv_row[f"evolved_{f_name}"] = _csv_val(e.get(f_name))
            writer.writerow(csv_row)


def _csv_val(v: Any) -> Any:
    """list/dict → JSON 字符串；其他类型保持原样供 csv 模块处理。"""
    if isinstance(v, (list, dict)):
        return json.dumps(v, ensure_ascii=False)
    return v
