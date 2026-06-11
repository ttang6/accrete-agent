"""TaskScore 列表 → summary dict 聚合。

EvalReport.summary 字段的来源——把多 task 的 TaskScore 折叠成可比较的均值 / 比率。
diff_report.py 后续按这些 key 做 baseline vs evolved 横向对比。

设计：
- 纯函数无副作用，输入 List[TaskScore] 输出 dict[str, Any]
- 所有除法都防 0：空列表 / 0 obligation / 0 lesson use → 返回保守默认（多 0 / 1.0 不 None）
- key 命名按"指标 → 单位"约定：rate（0-1 比率）/ avg_*（均值）/ total_*（求和）
- summary 是 dict（specs.EvalReport.summary 就是 dict[str, Any]）—— 演进 key 比改 dataclass 容易
"""

from __future__ import annotations

from typing import Any, List

from evals.specs import TaskScore


def aggregate(scores: List[TaskScore]) -> dict[str, Any]:
    """聚合多 task TaskScore → summary dict（embedding 进 EvalReport.summary）。"""
    n = len(scores)
    if n == 0:
        return _empty_summary()

    # 任务级聚合
    success_count = sum(1 for s in scores if s.success)
    finish_reason_dist: dict[str, int] = {}
    for s in scores:
        finish_reason_dist[s.finish_reason] = finish_reason_dist.get(s.finish_reason, 0) + 1

    # 步数 / token / 时长 / 输出
    total_steps = sum(s.total_steps for s in scores)
    total_tool_calls = sum(s.tool_calls for s in scores)
    total_tool_failures = sum(s.tool_failures for s in scores)
    total_llm_calls = sum(s.llm_calls for s in scores)
    total_duration_ms = sum(s.duration_ms for s in scores)
    total_final_chars = sum(s.final_answer_chars for s in scores)

    # Obligation 聚合
    total_obligations_required = sum(s.obligations_required for s in scores)
    total_obligations_repair_injected = sum(s.obligations_repair_injected for s in scores)
    total_obligations_violations = sum(s.obligations_violations for s in scores)
    obligation_completion_avg = sum(s.obligation_completion_rate for s in scores) / n

    # Expected tool_calls 聚合
    total_expected_tool_calls_required = sum(s.expected_tool_calls_required for s in scores)
    total_expected_tool_calls_hit = sum(s.expected_tool_calls_hit for s in scores)
    expected_tool_calls_coverage_avg = (
        sum(s.expected_tool_calls_coverage for s in scores) / n
    )

    # Lesson 飞轮信号（核心 metric——决定 baseline vs evolved 的真改善量化）
    total_lesson_uses = sum(s.lesson_uses for s in scores)
    total_lesson_helped = sum(s.lesson_helped for s in scores)
    total_lesson_hurt = sum(s.lesson_hurt for s in scores)
    total_lesson_ineffective = sum(s.lesson_ineffective for s in scores)
    lesson_outcome_total = total_lesson_helped + total_lesson_hurt + total_lesson_ineffective

    # tasks_with_lesson_use：用了 lesson 的 task 数（即 lesson_uses>0）
    tasks_with_lesson_use = sum(1 for s in scores if s.lesson_uses > 0)

    # Coverage（最终 missing 全 task 并集）
    coverage_missing_union = sorted({m for s in scores for m in s.coverage_missing})

    # pass^k（按 task_id 分组——同 fixture 跑 k 次时算"全 trial 成功"占比）
    by_task: dict[str, list[TaskScore]] = {}
    for s in scores:
        by_task.setdefault(s.task_id, []).append(s)
    total_unique_tasks = len(by_task)
    pass_k_full = sum(1 for trials in by_task.values() if all(t.success for t in trials))
    pass_k_rate = pass_k_full / total_unique_tasks if total_unique_tasks > 0 else 0.0
    # per-fixture trial 成功率：用于 debug 哪些 fixture 不稳定
    per_fixture_trial_success: dict[str, str] = {
        tid: f"{sum(1 for t in trials if t.success)}/{len(trials)}"
        for tid, trials in by_task.items()
    }

    # 按 recovery_type 分桶（盲点研究 PaladinEval §3：各类故障 agent 行为不同，
    # 混在一起会掩盖最弱的那类）。空 recovery_type 归到 "unlabeled" 桶。
    by_recovery_type = _bucket_by_recovery_type(scores)

    return {
        # 任务级（n = trials 总数；total_unique_tasks = 去重 fixture 数）
        "total_tasks": n,
        "total_unique_tasks": total_unique_tasks,
        "success_count": success_count,
        "success_rate": success_count / n,
        "pass_k_full": pass_k_full,
        "pass_k_rate": pass_k_rate,
        "per_fixture_trial_success": per_fixture_trial_success,
        "finish_reason_dist": finish_reason_dist,
        # 步数 / 时长
        "total_steps": total_steps,
        "avg_steps_per_task": total_steps / n,
        "avg_llm_calls_per_task": total_llm_calls / n,
        "avg_tool_calls_per_task": total_tool_calls / n,
        "avg_duration_ms_per_task": total_duration_ms / n,
        # Tool failure
        "total_tool_calls": total_tool_calls,
        "total_tool_failures": total_tool_failures,
        "tool_failure_rate": (
            total_tool_failures / total_tool_calls if total_tool_calls > 0 else 0.0
        ),
        # Obligation
        "total_obligations_required": total_obligations_required,
        "total_obligations_violations": total_obligations_violations,
        "total_obligations_repair_injected": total_obligations_repair_injected,
        "obligation_completion_rate_avg": obligation_completion_avg,
        # Expected tool calls
        "total_expected_tool_calls_required": total_expected_tool_calls_required,
        "total_expected_tool_calls_hit": total_expected_tool_calls_hit,
        "expected_tool_calls_coverage_rate": (
            total_expected_tool_calls_hit / total_expected_tool_calls_required
            if total_expected_tool_calls_required > 0
            else 1.0
        ),
        "expected_tool_calls_coverage_avg": expected_tool_calls_coverage_avg,
        # Lesson 飞轮
        "total_lesson_uses": total_lesson_uses,
        "total_lesson_helped": total_lesson_helped,
        "total_lesson_hurt": total_lesson_hurt,
        "total_lesson_ineffective": total_lesson_ineffective,
        "lesson_use_rate": tasks_with_lesson_use / n,  # 多大比例 task 用了 lesson
        "lesson_help_rate": (
            total_lesson_helped / lesson_outcome_total
            if lesson_outcome_total > 0
            else 0.0
        ),
        "lesson_hurt_rate": (
            total_lesson_hurt / lesson_outcome_total
            if lesson_outcome_total > 0
            else 0.0
        ),
        "lesson_ineffective_rate": (
            total_lesson_ineffective / lesson_outcome_total
            if lesson_outcome_total > 0
            else 0.0
        ),
        # 输出
        "avg_final_answer_chars": total_final_chars / n,
        # Coverage
        "coverage_missing_union": coverage_missing_union,
        # 按 recovery_type 分桶（R1-R4 各自 success/steps/max_iter/飞轮）
        "by_recovery_type": by_recovery_type,
        # Per-task 导出（供 diff_report 做 task 维度对比）
        "per_task": [_score_to_row(s) for s in scores],
    }


def _bucket_by_recovery_type(scores: List[TaskScore]) -> dict[str, Any]:
    """按 recovery_type 把 scores 分桶，每桶报 success/步数/max_iter/飞轮信号。

    让 OLD-vs-NEW 的 Δ 能精确到"哪类故障提升最大"，而非一个混合数字。
    """
    buckets: dict[str, List[TaskScore]] = {}
    for s in scores:
        buckets.setdefault(s.recovery_type or "unlabeled", []).append(s)
    out: dict[str, Any] = {}
    for rtype, group in buckets.items():
        m = len(group) or 1
        max_iter = sum(1 for s in group if s.finish_reason == "max_iter")
        out[rtype] = {
            "n": len(group),
            "success_rate": sum(1 for s in group if s.success) / m,
            "avg_steps": sum(s.total_steps for s in group) / m,
            "max_iter_share": max_iter / m,
            "avg_tool_failures": sum(s.tool_failures for s in group) / m,
            "lesson_use_rate": sum(1 for s in group if s.lesson_uses > 0) / m,
            "lesson_helped": sum(s.lesson_helped for s in group),
            "lesson_hurt": sum(s.lesson_hurt for s in group),
        }
    return out


def _empty_summary() -> dict[str, Any]:
    """空列表的默认 summary——保持 dict shape 一致避免 diff_report 拼 key 报错。"""
    return {
        "total_tasks": 0,
        "total_unique_tasks": 0,
        "success_count": 0,
        "success_rate": 0.0,
        "pass_k_full": 0,
        "pass_k_rate": 0.0,
        "per_fixture_trial_success": {},
        "finish_reason_dist": {},
        "total_steps": 0,
        "avg_steps_per_task": 0.0,
        "avg_llm_calls_per_task": 0.0,
        "avg_tool_calls_per_task": 0.0,
        "avg_duration_ms_per_task": 0.0,
        "total_tool_calls": 0,
        "total_tool_failures": 0,
        "tool_failure_rate": 0.0,
        "total_obligations_required": 0,
        "total_obligations_violations": 0,
        "total_obligations_repair_injected": 0,
        "obligation_completion_rate_avg": 1.0,  # 无 obligation = 满分
        "total_expected_tool_calls_required": 0,
        "total_expected_tool_calls_hit": 0,
        "expected_tool_calls_coverage_rate": 1.0,
        "expected_tool_calls_coverage_avg": 1.0,
        "total_lesson_uses": 0,
        "total_lesson_helped": 0,
        "total_lesson_hurt": 0,
        "total_lesson_ineffective": 0,
        "lesson_use_rate": 0.0,
        "lesson_help_rate": 0.0,
        "lesson_hurt_rate": 0.0,
        "lesson_ineffective_rate": 0.0,
        "avg_final_answer_chars": 0.0,
        "coverage_missing_union": [],
        "by_recovery_type": {},
        "per_task": [],
    }


def _score_to_row(s: TaskScore) -> dict[str, Any]:
    """单 TaskScore → flat dict（供 diff_report CSV / markdown 行输出）。"""
    return {
        "task_id": s.task_id,
        "success": s.success,
        "finish_reason": s.finish_reason,
        "total_steps": s.total_steps,
        "tool_calls": s.tool_calls,
        "tool_failures": s.tool_failures,
        "obligations_required": s.obligations_required,
        "obligations_violations": s.obligations_violations,
        "obligation_completion_rate": s.obligation_completion_rate,
        "expected_tool_calls_required": s.expected_tool_calls_required,
        "expected_tool_calls_hit": s.expected_tool_calls_hit,
        "expected_tool_calls_coverage": s.expected_tool_calls_coverage,
        "lesson_uses": s.lesson_uses,
        "lesson_helped": s.lesson_helped,
        "lesson_hurt": s.lesson_hurt,
        "lesson_ineffective": s.lesson_ineffective,
        "coverage_missing": list(s.coverage_missing),
        "final_answer_chars": s.final_answer_chars,
        "duration_ms": s.duration_ms,
    }
