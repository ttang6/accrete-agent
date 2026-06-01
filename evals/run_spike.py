"""evals/run_spike.py — 模型对照 spike：验证"飞轮天花板悖论"。

假设：agent 越成熟（model 越强），失败越少 → 飞轮没米下锅 → lesson_use_rate ≈ 0。
反过来：弱 model 应该让飞轮真正"上岗"——lesson_use_rate 应飙升。

跑：2 fixture × 2 model × 2 mode = 8 cells。
- fixture: ai_digest_basic（标准种子）+ mut2_mark_screenshot_style（已知唯一飞轮命中）
- model: gpt-4o-mini, gpt-5-mini（5.4-mini 不重跑，复用 evals/reports/diff_200344）
- mode: baseline (lesson 召回关) / evolved (lesson 召回开)

每 model 用独立 sqlite spike 副本（从主 backend 拷起点），跑完父进程删副本。
主 backend 完全不动——spike 不污染生产 lesson 库。

执行（PowerShell）：
  .\\.venv\\Scripts\\python.exe -m evals.run_spike
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# spike 配置
# ============================================================

# 每行 = (model, provider)。provider=None → openai 默认。
# 5.4-mini (openai) 复用 evals/reports/diff_200344；4o-mini 已在 spike_summary_221616.json。
# 本轮加 qwen3.5-plus (dashscope) 验证跨家弱模型路径 + 补三档梯度中段。
SPIKE_RUNS: tuple[tuple[str, str | None], ...] = (
    ("qwen3.5-plus", "dashscope"),
)
SPIKE_FIXTURE_IDS: tuple[str, ...] = ("ai_digest_basic", "mut2_mark_screenshot_style")

MAIN_DB: Path = Path("data/runtime/lessons/runtime_memory.sqlite")
TASKS_DIR: Path = Path("evals/tasks")
REPORTS_DIR: Path = Path("evals/reports")

# 5.4-mini 基线参考（spike 完后人工对比，不重跑）
BASELINE_5_4_MINI_REF: Path = Path("evals/reports/diff_200344/report.md")


# ============================================================


def _safe_model_tag(model: str) -> str:
    return model.replace("-", "_").replace(".", "_")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if not MAIN_DB.exists():
        print(f"[spike] 主 backend 不存在: {MAIN_DB}")
        return 1

    from evals.run_eval import run_one_mode
    from evals.specs import load_task_specs

    all_specs = load_task_specs(TASKS_DIR)
    specs = [s for s in all_specs if s.id in SPIKE_FIXTURE_IDS]
    missing = set(SPIKE_FIXTURE_IDS) - {s.id for s in specs}
    if missing:
        print(f"[spike] 缺 fixture: {missing}")
        return 1
    print(f"[spike] fixtures: {[s.id for s in specs]}")
    print(f"[spike] runs:     {[(m, p) for m, p in SPIKE_RUNS]}")

    # spike 不污染 main.py 的生产签名——直接 monkey-patch 模块级常量。
    # build_loop / build_runtime_memory 的函数体在调用时现读这三个常量，
    # 所以重赋值后下一次装配就走 spike 注入的 model / provider / db_path。
    import main as cli_main

    timestamp = datetime.now().strftime("%H%M%S")
    summary_rows: list[dict] = []

    for model, provider in SPIKE_RUNS:
        tag = _safe_model_tag(model)
        spike_db = Path(f"data/runtime/lessons/spike_{tag}.sqlite")
        shutil.copy2(MAIN_DB, spike_db)
        print(f"\n[spike] === MODEL: {model} (provider={provider or 'openai'}, tag={tag}) ===")
        print(f"[spike] backend snapshot: {spike_db}")

        cli_main.MODEL = model
        cli_main.PROVIDER = provider or ""
        cli_main.LESSONS_DB_PATH = spike_db

        try:
            for mode in ("baseline", "evolved"):
                out_dir = REPORTS_DIR / f"spike_{tag}_{mode}_{timestamp}"
                t0 = time.time()
                report = run_one_mode(
                    mode,
                    specs,
                    out_dir,
                    sqlite_snapshot_path=spike_db,
                )
                elapsed = time.time() - t0
                s = report.summary
                row = {
                    "model": model,
                    "provider": provider or "openai",
                    "mode": mode,
                    "n": s["total_tasks"],
                    "success_rate": s["success_rate"],
                    "lesson_use_rate": s["lesson_use_rate"],
                    "lesson_help_rate": s["lesson_help_rate"],
                    "lesson_uses_total": s["total_lesson_uses"],
                    "lesson_helped_total": s["total_lesson_helped"],
                    "tool_failure_rate": s["tool_failure_rate"],
                    "avg_steps": s["avg_steps_per_task"],
                    "avg_duration_ms": s["avg_duration_ms_per_task"],
                    "elapsed_s": elapsed,
                    "report_dir": str(out_dir),
                }
                summary_rows.append(row)
                print(
                    f"[spike/{tag}/{mode}] "
                    f"success={s['success_rate']:.2f} "
                    f"use_rate={s['lesson_use_rate']:.4f} "
                    f"uses={s['total_lesson_uses']} helped={s['total_lesson_helped']} "
                    f"tool_fail={s['tool_failure_rate']:.4f} "
                    f"steps={s['avg_steps_per_task']:.1f} "
                    f"dur={s['avg_duration_ms_per_task']:.0f}ms "
                    f"elapsed={elapsed:.1f}s"
                )
        finally:
            try:
                spike_db.unlink()
            except OSError:
                pass

    # 写顶层 spike summary
    summary_path = REPORTS_DIR / f"spike_summary_{timestamp}.json"
    summary_payload = {
        "timestamp": timestamp,
        "runs": [(m, p) for m, p in SPIKE_RUNS],
        "fixtures": list(SPIKE_FIXTURE_IDS),
        "rows": summary_rows,
        "baseline_5_4_mini_ref": str(BASELINE_5_4_MINI_REF),
    }
    summary_path.write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[spike] done. summary: {summary_path}")
    print(f"[spike] 5.4-mini 基线参考: {BASELINE_5_4_MINI_REF}")
    print()
    print("| model | provider | mode | n | success | use_rate | uses | helped | tool_fail | steps |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary_rows:
        print(
            f"| {r['model']} | {r['provider']} | {r['mode']} | {r['n']} | "
            f"{r['success_rate']:.2f} | "
            f"{r['lesson_use_rate']:.4f} | "
            f"{r['lesson_uses_total']} | "
            f"{r['lesson_helped_total']} | "
            f"{r['tool_failure_rate']:.4f} | "
            f"{r['avg_steps']:.1f} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
