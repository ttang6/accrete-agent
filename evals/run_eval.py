"""Eval harness CLI driver — 跑全部 task fixtures 并产出 EvalReport + diff。

三种模式（顶部常量切换，遵循项目"不走 argparse"约定）：

- `mock`: 不调 LLM，按 spec.expected_tool_calls 合成"全绿" trace，验证 pipeline 通路。
  适合 CI 不冒 API key 风险，单测覆盖。
- `baseline`: 真调 LLM，`NANOAGENT_ENABLE_LESSON_RECALL=0` / `_PROMOTION_GATE=0`。
  飞轮关闭，作为对照参照。LessonIngestor 仍在跑（accumulate lessons 到 backend），
  让后续 evolved 跑有 lesson 可召回。
- `evolved`: 真调 LLM，`NANOAGENT_ENABLE_LESSON_RECALL=1` / `_PROMOTION_GATE=1`。
  飞轮全开，复用 baseline 跑完留下的 sqlite 起点。

`MODES` 含 2 项时自动产 diff_report.markdown / .csv 把 baseline 和 evolved 的 metric 横向对照。

输出（每次 run）：
- `evals/reports/<mode>_<HHMMSS>/report.json`        EvalReport 序列化
- `evals/reports/<mode>_<HHMMSS>/per_task_<id>.json` 每 task 的 score 单独一份（便于 grep）
- `evals/reports/diff_<HHMMSS>/report.md`            两 mode 的 markdown diff
- `evals/reports/diff_<HHMMSS>/report.csv`           CSV per-task 对比

依赖（仅 `baseline` / `evolved` 模式）：
- `OPENAI_API_KEY`（主 LLM）
- DASHSCOPE_API_KEY 不必（eval 默认 evaluator=None 跳过副 LLM 评审）
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional

from nanoagent.core import trace_schema as ts

from evals.aggregator import aggregate
from evals.diff_report import write_csv_report, write_markdown_report
from evals.grader import grade_trace
from evals.specs import (
    EvalConfig,
    EvalReport,
    TaskScore,
    TaskSpec,
    load_task_specs,
)


# ============================================================
# 运行参数（改这里就改行为，不需要命令行）
# ============================================================

TASKS_DIR: Path = Path("evals/tasks")
OUTPUT_BASE_DIR: Path = Path("evals/reports")
# MODES: tuple[str, ...] = ("mock",)  # 单 mode 跑该 mode；("baseline","evolved") 两 mode 跑 + 出 diff
MODES = ("baseline", "evolved")
RUN_TIMEOUT_PER_TASK_S: int = 300  # 单 task 上限（仅 baseline/evolved 生效）


# ============================================================
# 公共：序列化 EvalReport → JSON-friendly dict
# ============================================================


def serialize_eval_report(report: EvalReport) -> dict:
    """frozen dataclass + Path → 可直接 json.dump 的 dict。"""
    return {
        "config": _serialize_config(report.config),
        "started_at": report.started_at,
        "finished_at": report.finished_at,
        "task_scores": [_serialize_score(s) for s in report.task_scores],
        "summary": report.summary,
    }


def _serialize_config(cfg: EvalConfig) -> dict:
    d = asdict(cfg)
    d["sqlite_snapshot_path"] = str(cfg.sqlite_snapshot_path) if cfg.sqlite_snapshot_path else None
    return d


def _serialize_score(score: TaskScore) -> dict:
    return asdict(score)


# ============================================================
# Mock 模式：合成"全绿" trace
# ============================================================


def synthesize_mock_trace(spec: TaskSpec, trace_path: Path) -> None:
    """按 spec 的 expected_tool_calls / expected_obligations / expected_coverage
    合成一份"理想跑"的 trace JSONL。grader 抽完应得满分。

    用途：CI smoke + 单测验证 pipeline，不调 LLM。
    """
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().isoformat()

    events: list[dict] = []
    events.append({
        "type": "run_header",
        "agent": "mock_eval",
        "user_input": spec.query,
        "total_steps": "...",
        "total_duration_ms": "...",
        "started_at": now,
    })

    step = 0
    # 1) LLM call start + tool call pairs for each expected_tool_call
    for i, pattern in enumerate(spec.expected_tool_calls, start=1):
        tool = pattern.get("tool", "fetch")
        if tool == "skill_exec":
            input_obj = {
                "skill": pattern.get("skill", ""),
                "script": pattern.get("script", ""),
                "args": {},
            }
            input_str = json.dumps(input_obj, ensure_ascii=False)
        else:
            input_str = "{}"

        step += 1
        events.append({
            "step": step,
            "action": ts.ACTION_LLM_CALL_END,
            "iteration": i,
            "duration_ms": 100,
        })
        step += 1
        events.append({
            "step": step,
            "action": ts.ACTION_TOOL_CALL_START,
            "iteration": i,
            "tool": tool,
            "input": input_str,
            "duration_ms": 1,
        })
        step += 1
        events.append({
            "step": step,
            "action": ts.ACTION_TOOL_CALL_END,
            "iteration": i,
            "tool": tool,
            "input": input_str,
            "output": "ok",
            "duration_ms": 50,
        })

    # 2) Coverage check 满足 expected_coverage（无 missing）
    if spec.expected_coverage:
        step += 1
        events.append({
            "step": step,
            "action": ts.ACTION_COVERAGE_CHECK,
            "iteration": max(1, len(spec.expected_tool_calls)),
            "missing": [],
            "counts": {c: 5 for c in spec.expected_coverage},
        })

    # 3) Final answer (含 success_keywords 让 LLM-judge eval 也能通过；
    # 当前 grader 不读关键词但留个 hook)
    final_answer = (
        " ".join(spec.success_keywords) if spec.success_keywords else "mock answer"
    )
    step += 1
    events.append({
        "step": step,
        "action": ts.ACTION_FINISH,
        "iterations": max(1, len(spec.expected_tool_calls)),
        "output": final_answer,
    })

    events.append({
        "type": "run_summary",
        "total_steps": step,
        "total_duration_ms": 100 * max(1, len(spec.expected_tool_calls)),
        "ended_at": now,
    })

    with open(trace_path, "w", encoding="utf-8") as f:
        for e in events:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ============================================================
# Real 模式：调 main.py 的装配，per-task 跑 Harness
# ============================================================


def _build_harness_for_eval(mode: str):
    """复用 main.py 的装配链。env vars 已在 caller 设置好（见 run_one_mode）。"""
    # 仅 baseline/evolved 模式需要——延迟 import 让 mock 模式不依赖 LLM 配置
    import main as cli_main
    from nanoagent.evolution.reflexion import ReflexionStore
    from nanoagent.evolution.runtime_memory.promotion_audit import JsonlAuditWriter
    from nanoagent.memory.user_facts import UserFacts
    from nanoagent.runtime.harness import Harness
    from nanoagent.runtime.session import SessionStore
    from nanoagent.skills.loader import SkillLoader

    # main.py 的 ENABLE_* 是 import 时绑定的模块级常量。run_eval 在同一进程
    # 先跑 baseline 再跑 evolved，不能只改 env；这里显式 patch 装配开关。
    #
    # baseline 需要"召回关闭，但 LessonIngestor 仍入库"：否则 baseline 产生的
    # failure episode 不会变成 candidate lesson，evolved 没东西可召回。
    # 因此 runtime memory 组件始终装配；只在传给 MainLoop 时按 mode 控制 retriever。
    cli_main.ENABLE_LESSON_RECALL = True
    cli_main.ENABLE_PROMOTION_GATE = mode == "evolved"

    reflexions = ReflexionStore(cli_main.REFLEXIONS_DIR)
    loader = SkillLoader(cli_main.SKILLS_DIR, reflexions_store=reflexions)
    lesson_retriever, outcome_tracker, lesson_ingestor, promotion_gate = (
        cli_main.build_runtime_memory()
    )
    loop = cli_main.build_loop(
        loader,
        lesson_retriever=lesson_retriever if mode == "evolved" else None,
    )

    store = SessionStore(persist_dir=cli_main.SESSION_DIR)
    user_facts = UserFacts(cli_main.USER_FACTS_PATH)
    audit_cb = (
        JsonlAuditWriter(cli_main.PROMOTION_AUDIT_LOG_PATH)
        if promotion_gate is not None and cli_main.PROMOTION_AUDIT_LOG_PATH
        else None
    )

    return Harness(
        loop=loop,
        store=store,
        loader=loader,
        user_facts=user_facts,
        session_key="eval:default",
        base_identity=cli_main.BASE_IDENTITY,
        evaluator=None,  # eval 不跑副 LLM 评审，避免成本 + 旁路依赖
        outcome_tracker=outcome_tracker,
        lesson_ingestor=lesson_ingestor,
        promotion_gate=promotion_gate,
        promotion_audit_callback=audit_cb,
    )


def _run_one_task_real(harness, spec: TaskSpec) -> TaskScore:
    """real 模式跑单 task。返回 TaskScore（grader 失败时也返一个 error score）。

    多轮（spec.query 是 List[str]）：按顺序喂给同一 session 的 harness，
    收集每轮 trace_path → 累加打分。grader 现在跨 trace 累加飞轮命中、
    obligation、tool_calls 等信号；终态指标（success / coverage / final_answer）
    取最后一份 trace。
    """
    # 每 task 独立 session（避免上 task 历史污染）
    harness.handle("/new")
    if spec.skill:
        harness.handle(f"/skill {spec.skill}")

    queries = [spec.query] if isinstance(spec.query, str) else list(spec.query)
    trace_paths: List[Path] = []
    for q in queries:
        try:
            harness.handle(q)
        except Exception as e:  # noqa: BLE001
            last = str(trace_paths[-1]) if trace_paths else ""
            return _error_score(spec, last, f"harness exception: {e}")
        tracer = getattr(harness._loop, "_tracer", None)
        tp = getattr(tracer, "_trace_path", None) if tracer else None
        if tp:
            path = Path(tp)
            # 同一 session 内连续多轮可能复用同一 trace 文件，去重避免双计
            if not trace_paths or trace_paths[-1] != path:
                trace_paths.append(path)

    if not trace_paths:
        return _error_score(spec, "", "no trace_path")
    return grade_trace(spec, trace_paths)


def _error_score(spec: TaskSpec, trace_path: str, reason: str) -> TaskScore:
    return TaskScore(
        task_id=spec.id,
        success=False,
        finish_reason="error",
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
        trace_path=trace_path or f"<error:{reason}>",
    )


# ============================================================
# Mode 级 driver
# ============================================================


def run_one_mode(
    mode: str,
    specs: List[TaskSpec],
    output_dir: Path,
    *,
    sqlite_snapshot_path: Optional[Path] = None,
    pass_k_per_spec: Optional[Callable[[TaskSpec], int]] = None,
) -> EvalReport:
    """跑一个 mode 全部 specs，写 report.json + per_task_*.json，返回 EvalReport。

    pass_k_per_spec: 给每个 spec 决定跑多少次（pass^k 测 stochasticity）。
        默认 None = 每 spec 跑 1 次。callable 返回 k>=1。
        典型用法：lambda s: 2 if isinstance(s.query, list) else 1
        （多轮跑 2 次、单轮跑 1 次）。

    pass^k 含义：每个 spec 跑 k 次，"k 次都成功"才算该 spec pass^k。
    aggregator 计算 pass_k_rate = pass^k spec 数 / 总 spec 数。
    """
    if mode not in ("mock", "baseline", "evolved"):
        raise ValueError(f"unknown mode: {mode!r}")

    config = EvalConfig(
        mode=mode,
        enable_lesson_recall=(mode == "evolved"),
        enable_promotion_gate=(mode == "evolved"),
        sqlite_snapshot_path=sqlite_snapshot_path,
    )

    # 设置 env override（仅影响 import main 时的常量绑定；mock 模式不 import main）
    if mode != "mock":
        os.environ["NANOAGENT_ENABLE_LESSON_RECALL"] = "1" if mode == "evolved" else "0"
        os.environ["NANOAGENT_ENABLE_PROMOTION_GATE"] = "1" if mode == "evolved" else "0"

    started_at = datetime.now().isoformat(timespec="seconds")
    output_dir.mkdir(parents=True, exist_ok=True)

    scores: List[TaskScore] = []

    def _k_for(spec: TaskSpec) -> int:
        if pass_k_per_spec is None:
            return 1
        k = pass_k_per_spec(spec)
        return max(1, int(k))

    if mode == "mock":
        # mock 不跑 pass^k（合成 trace 是 deterministic）
        for spec in specs:
            trace_path = output_dir / "mock_traces" / f"{spec.id}.jsonl"
            synthesize_mock_trace(spec, trace_path)
            score = grade_trace(spec, trace_path)
            scores.append(score)
            _write_score_json(output_dir, spec, score)
    else:
        # 装配 harness（一次，跨 task 复用 session store / loader / backend）
        harness = _build_harness_for_eval(mode)
        for spec in specs:
            k = _k_for(spec)
            for trial in range(k):
                t0 = time.time()
                score = _run_one_task_real(harness, spec)
                elapsed_s = time.time() - t0
                trial_tag = f" trial={trial+1}/{k}" if k > 1 else ""
                print(f"[eval/{mode}] {spec.id}: success={score.success} steps={score.total_steps} ({elapsed_s:.1f}s){trial_tag}")
                scores.append(score)
                _write_score_json(output_dir, spec, score, trial=trial if k > 1 else None)

    finished_at = datetime.now().isoformat(timespec="seconds")
    summary = aggregate(scores)
    report = EvalReport(
        config=config,
        started_at=started_at,
        finished_at=finished_at,
        task_scores=scores,
        summary=summary,
    )

    # 写整体 report.json
    with open(output_dir / "report.json", "w", encoding="utf-8") as f:
        json.dump(serialize_eval_report(report), f, ensure_ascii=False, indent=2)

    return report


def _write_score_json(
    output_dir: Path, spec: TaskSpec, score: TaskScore, trial: Optional[int] = None
) -> None:
    """单 task 评分单独落一份（便于 grep / debug）。

    trial=None 走原文件名（兼容 pass_k=1 默认场景）；
    trial=int 时文件名带 _trial_<i> 后缀，避免 pass^k 多 trial 互相覆盖。
    """
    per_task_dir = output_dir / "per_task"
    per_task_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_trial_{trial}" if trial is not None else ""
    out = per_task_dir / f"{spec.id}{suffix}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(_serialize_score(score), f, ensure_ascii=False, indent=2)


# ============================================================
# 入口
# ============================================================


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    # 加载 .env（baseline / evolved 模式需要 OPENAI_API_KEY 等）
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # dotenv 不强制；用户也可手动 export 环境变量

    if not TASKS_DIR.exists():
        print(f"[eval] tasks dir 不存在: {TASKS_DIR}")
        return 1

    specs = load_task_specs(TASKS_DIR)
    if not specs:
        print(f"[eval] {TASKS_DIR} 里没有 *.yaml fixture")
        return 1

    print(f"[eval] 发现 {len(specs)} 个 task fixtures，跑 modes={MODES}")
    timestamp = datetime.now().strftime("%H%M%S")
    reports: dict[str, EvalReport] = {}

    for mode in MODES:
        out_dir = OUTPUT_BASE_DIR / f"{mode}_{timestamp}"
        print(f"[eval] === MODE: {mode} -> {out_dir} ===")
        report = run_one_mode(mode, specs, out_dir)
        reports[mode] = report
        print(f"[eval/{mode}] success_rate={report.summary['success_rate']:.2f} "
              f"lesson_help_rate={report.summary['lesson_help_rate']:.2f}")

    # 两 mode 时自动产 diff
    if len(MODES) == 2:
        baseline_mode = "baseline" if "baseline" in reports else MODES[0]
        evolved_mode = "evolved" if "evolved" in reports else MODES[1]
        diff_dir = OUTPUT_BASE_DIR / f"diff_{timestamp}"
        diff_dir.mkdir(parents=True, exist_ok=True)
        write_markdown_report(
            reports[baseline_mode], reports[evolved_mode], diff_dir / "report.md"
        )
        write_csv_report(
            reports[baseline_mode], reports[evolved_mode], diff_dir / "report.csv"
        )
        print(f"[eval] diff 写入 {diff_dir}/report.md + report.csv")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
