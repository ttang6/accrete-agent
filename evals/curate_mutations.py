"""Curate raw mutation fixtures into top-level eval tasks.

This is the human-review materialization step for docs/_fixture_mutation_plan.md:
- read the latest raw LLM output under evals/tasks/mutations/
- apply a fixed keep list chosen after review
- strip review-only fields
- write one top-level evals/tasks/mut_*.yaml file per kept fixture
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml


# ============================================================
# 运行参数（改这里就改行为，不需要命令行）
# ============================================================
RAW_DIR: Path = Path("evals/tasks/mutations")
OUTPUT_DIR: Path = Path("evals/tasks")
RAW_FILE: Path | None = None
OVERWRITE: bool = True


# ============================================================
# 人工筛选结果
# ============================================================
KEEP_IDS: tuple[str, ...] = (
    "mut_schema_date_range",
    "mut_schema_topk_unusual",
    "mut_schema_filter_keywords",
    "mut_coverage_only_news",
    "mut_coverage_minimal_query",
    "mut_coverage_topic_narrow",
    "mut_coverage_yesterday",
    "mut_soft_quality_too_brief",
    "mut_soft_quality_chinese_only",
    "mut_soft_quality_compare",
    "mut_soft_quality_no_link",
    "mut_obligation_only_papers_mark",
    "mut_conflict_brief_but_full",
    "mut_conflict_no_dup_no_mark",
    "mut_conflict_dedup_but_show_old",
    "mut_runtime_github_rate",
    "mut_runtime_mixed_sources",
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


def _latest_raw_file(raw_dir: Path) -> Path:
    candidates = sorted(
        raw_dir.glob("raw_*.yaml"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"未找到 raw mutation YAML: {raw_dir}")
    return candidates[0]


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

    raw_path = RAW_FILE or _latest_raw_file(RAW_DIR)
    raw_tasks = _load_raw_docs(raw_path)
    by_id = {str(task.get("id")): task for task in raw_tasks}

    missing = [task_id for task_id in KEEP_IDS if task_id not in by_id]
    if missing:
        raise ValueError(f"raw 文件缺少 keep id: {missing}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for task_id in KEEP_IDS:
        task = _task_for_output(by_id[task_id])
        written.append(_write_task(task, OUTPUT_DIR))

    print(f"[curate] raw: {raw_path}")
    print(f"[curate] wrote: {len(written)} fixtures")
    for path in written:
        print(f"[curate] - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
