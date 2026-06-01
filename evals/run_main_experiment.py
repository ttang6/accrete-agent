"""4o-mini × 50 fixture × pass^k 主实验 driver（chapter 78 主线证据）。

为什么 4o-mini？spike 实证：4o-mini 上飞轮 9× tool_failure / 16× lesson_use_rate
（vs gpt-5.4-mini），是飞轮"上岗"的能力档。capability-axis 主战场。

跑：
  .\\.venv\\Scripts\\python.exe -m evals.run_main_experiment

设计：
- single-turn 28 fixture × 1 trial（基础统计）
- multi-turn 22 fixture × 2 trial（pass^k stochasticity）
  → 28×1 + 22×2 = 72 trial/mode × 2 mode (baseline+evolved) = 144 trial
- baseline + evolved 双 mode，共享 sqlite snapshot（让 baseline 跑添加的 candidate
  lesson 能被 evolved 召回——run_eval 原设计意图）
- backend 用 snapshot 隔离不污染主 backend
- diff_report 自动产出
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path


# ============================================================
# 主实验配置
# ============================================================
MODEL: str = "gpt-4o-mini"
PROVIDER: str = ""  # 空 = openai
PASS_K_MULTI_TURN: int = 2
MODES: tuple[str, ...] = ("baseline", "evolved")

TASKS_DIR: Path = Path("evals/tasks")
REPORTS_DIR: Path = Path("evals/reports")
MAIN_DB: Path = Path("data/runtime/lessons/runtime_memory.sqlite")


def main() -> int:
    # 必须在 import main 之前设 env，让 main.py 顶部 MODEL 常量绑定到 4o-mini
    os.environ["OPENAI_MODEL_ID"] = MODEL

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    try:
        from dotenv import load_dotenv
        load_dotenv()  # default override=False，保留上面 set 的 OPENAI_MODEL_ID
    except ImportError:
        pass

    if not MAIN_DB.exists():
        print(f"[main-exp] 主 backend 不存在: {MAIN_DB}")
        return 1

    # 拷贝主 backend 到 snapshot，main.py LESSONS_DB_PATH 通过 monkey-patch 指过去
    timestamp = datetime.now().strftime("%H%M%S")
    snapshot_db = Path(f"data/runtime/lessons/main_exp_{timestamp}.sqlite")
    shutil.copy2(MAIN_DB, snapshot_db)

    import main as cli_main
    cli_main.LESSONS_DB_PATH = snapshot_db

    from evals.run_eval import run_one_mode
    from evals.specs import load_task_specs, TaskSpec
    from evals.diff_report import write_csv_report, write_markdown_report

    specs = load_task_specs(TASKS_DIR)
    multi_specs = [s for s in specs if isinstance(s.query, list)]
    single_specs = [s for s in specs if isinstance(s.query, str)]

    def pass_k_for(s: TaskSpec) -> int:
        return PASS_K_MULTI_TURN if isinstance(s.query, list) else 1

    total_trials_per_mode = len(single_specs) + len(multi_specs) * PASS_K_MULTI_TURN
    print(f"[main-exp] model: {MODEL}")
    print(f"[main-exp] backend snapshot: {snapshot_db}")
    print(
        f"[main-exp] fixtures: {len(specs)} "
        f"({len(single_specs)} single + {len(multi_specs)} multi × pass^{PASS_K_MULTI_TURN})"
    )
    print(f"[main-exp] total trials per mode: {total_trials_per_mode}")
    print(f"[main-exp] modes: {list(MODES)} (sequential, share snapshot)")

    reports: dict = {}
    try:
        for mode in MODES:
            out_dir = REPORTS_DIR / f"main_{mode}_{timestamp}"
            print(f"\n[main-exp] === MODE: {mode} -> {out_dir} ===")
            t0 = time.time()
            report = run_one_mode(
                mode,
                specs,
                out_dir,
                sqlite_snapshot_path=snapshot_db,
                pass_k_per_spec=pass_k_for,
            )
            elapsed = time.time() - t0
            s = report.summary
            reports[mode] = report
            print(
                f"[main-exp/{mode}] "
                f"success_rate={s['success_rate']:.3f} "
                f"pass_k_rate={s['pass_k_rate']:.3f} "
                f"use_rate={s['lesson_use_rate']:.4f} "
                f"helped={s['total_lesson_helped']} "
                f"tool_fail={s['tool_failure_rate']:.4f} "
                f"elapsed={elapsed:.1f}s"
            )

        # diff_report
        if len(MODES) == 2 and "baseline" in reports and "evolved" in reports:
            diff_dir = REPORTS_DIR / f"main_diff_{timestamp}"
            diff_dir.mkdir(parents=True, exist_ok=True)
            write_markdown_report(
                reports["baseline"], reports["evolved"], diff_dir / "report.md"
            )
            write_csv_report(
                reports["baseline"], reports["evolved"], diff_dir / "report.csv"
            )
            print(f"\n[main-exp] diff written: {diff_dir}/report.md + report.csv")
    finally:
        try:
            snapshot_db.unlink()
            print(f"[main-exp] snapshot cleaned up: {snapshot_db}")
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
