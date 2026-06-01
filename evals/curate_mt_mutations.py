"""Curate raw multi-turn mutation fixtures into top-level eval tasks.

跟 curate_mutations.py 同一套 review-then-materialize 流程，仅作用于 mt_*：
- 读 evals/tasks/mt_mutations/raw_*.yaml 最新一份
- 按 KEEP_IDS 选保留条目
- strip review-only 字段（expected_failure_type / expected_pattern）
- 写到 evals/tasks/<id>.yaml

人工 review 后改 KEEP_IDS 再跑。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# 运行参数
# ============================================================
RAW_DIR: Path = Path("evals/tasks/mt_mutations")
OUTPUT_DIR: Path = Path("evals/tasks")
RAW_FILE: Path | None = None
OVERWRITE: bool = True


# ============================================================
# 人工筛选结果（review raw_*.yaml 后填入）
# ============================================================
KEEP_IDS: tuple[str, ...] = (
    # 短多轮（2-4 turn）—— Stage 2 第一批
    "mt_cross_query_mark_github_subset",
    "mt_cross_query_mark_news_only",
    "mt_drill_down_rag_then_mark",
    "mt_correct_redo_time_window",
    "mt_correct_redo_category_switch",
    "mt_consistency_check_before_mark",
    "mt_consistency_no_dup_digest",
    "mt_soft_quality_too_short",
    "mt_conflict_source_then_expand",
    "mt_conflict_today_vs_yesterday",
    # 长多轮（5-7 turn）—— Stage 2 第二批
    "mt_long_drill_down_moe_then_mark",
    "mt_long_drill_down_agent_eval_then_mark",
    "mt_long_cross_topic_paper_oss_news_mark",
    "mt_long_correct_redo_stacking",
    "mt_long_failchain_schema_mismatch",
    "mt_long_dup_check_batch_then_mark",
)


TASK_FIELD_ORDER: tuple[str, ...] = (
    "id",
    "description",
    "query",
    "expected_obligations",
    "expected_tool_calls",
    "expected_coverage",
    "max_iterations",
    "success_keywords",
)


def _all_raw_files(raw_dir: Path) -> list[Path]:
    """读全部 raw_*.yaml（包含 raw_long_*.yaml）按 mtime 排序。

    跨 raw 文件 merge：让短 fixture 和长 fixture 在同一次 curate 里被 KEEP_IDS 选。
    """
    candidates = sorted(
        raw_dir.glob("raw*.yaml"),
        key=lambda path: path.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(f"未找到 raw mutation YAML: {raw_dir}")
    return candidates


def _repair_raw_yaml(raw_text: str) -> str:
    """Repair common LLM YAML quoting issue in description lines.

    Example:
      description: "存一份"是隐式 mark 触发
    becomes:
      description: 存一份是隐式 mark 触发
    """
    repaired_lines: list[str] = []
    pattern = re.compile(r'^(description:\s*)"([^"]+)"(.+)$')
    for line in raw_text.splitlines():
        match = pattern.match(line)
        if match:
            line = f"{match.group(1)}{match.group(2)}{match.group(3)}"
        repaired_lines.append(line)
    return "\n".join(repaired_lines) + "\n"


def _load_raw_docs(raw_path: Path) -> list[dict[str, Any]]:
    raw_text = raw_path.read_text(encoding="utf-8")
    repaired = _repair_raw_yaml(raw_text)
    docs = [doc for doc in yaml.safe_load_all(repaired) if doc is not None]
    tasks: list[dict[str, Any]] = []
    for doc in docs:
        if not isinstance(doc, dict):
            raise ValueError(f"{raw_path}: YAML doc root 必须是 dict")
        tasks.append(doc)
    return tasks


def _task_for_output(raw_task: dict[str, Any]) -> dict[str, Any]:
    return {
        key: raw_task[key]
        for key in TASK_FIELD_ORDER
        if key in raw_task
    }


def _write_task(task: dict[str, Any], output_dir: Path) -> Path:
    task_id = str(task["id"])
    output_path = output_dir / f"{task_id}.yaml"
    if output_path.exists() and not OVERWRITE:
        raise FileExistsError(f"fixture 已存在: {output_path}")
    text = yaml.safe_dump(
        task,
        allow_unicode=True,
        sort_keys=False,
        default_flow_style=False,
    )
    output_path.write_text(text, encoding="utf-8")
    return output_path


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

    if not KEEP_IDS:
        print("[curate-mt] KEEP_IDS 为空——请 review raw 后填入再跑")
        return 1

    if RAW_FILE is not None:
        raw_paths = [RAW_FILE]
    else:
        raw_paths = _all_raw_files(RAW_DIR)

    # merge by id（跨多份 raw 文件；后写的 raw 同 id 覆盖前面，按 mtime 升序保证）
    by_id: dict[str, dict[str, Any]] = {}
    src_by_id: dict[str, Path] = {}
    for raw_path in raw_paths:
        for task in _load_raw_docs(raw_path):
            tid = str(task.get("id"))
            by_id[tid] = task
            src_by_id[tid] = raw_path

    missing = [task_id for task_id in KEEP_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"raw 文件缺少 keep id: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task_id in KEEP_IDS:
        task = _task_for_output(by_id[task_id])
        written.append(_write_task(task, OUTPUT_DIR))

    print(f"[curate-mt] raw sources ({len(raw_paths)}):")
    for p in raw_paths:
        print(f"[curate-mt]   - {p}")
    print(f"[curate-mt] wrote: {len(written)} fixtures")
    for task_id, path in zip(KEEP_IDS, written):
        print(f"[curate-mt] - {path} (from {src_by_id[task_id].name})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
