"""lesson 只读运维 CLI（list / show）。

lesson 生命周期已改为 lesson_score 从账本（helped/hurt）派生的单分数注入闸
（score ≥ T 即 active），无存储 status、无手动 promote/retire/reset——晋降退休不再是
人可手动改的状态,而是账本决定的派生量。CLI 只负责只读查询 + 稳定 JSON 输出。

Exit code:
    0 = 成功
    1 = lesson_id not found
    3 = 参数错误（argparse 默认）
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from nanoagent.evolution.runtime_memory.backend import MemoryBackend
from nanoagent.evolution.runtime_memory.lesson_score import (
    compute_score,
    is_active,
    status_label,
)
from nanoagent.evolution.runtime_memory.schema import RuntimeLesson
from nanoagent.evolution.runtime_memory.sqlite_backend import (
    DEFAULT_DB_PATH,
    SqliteMemoryBackend,
)


EXIT_OK = 0
EXIT_NOT_FOUND = 1


def _lesson_to_summary_dict(lesson: RuntimeLesson) -> dict:
    """list / show 共用的稳定 JSON schema。下游脚本会依赖这套字段。
    status / score 均为账本派生（非存储字段）。"""
    return {
        "lesson_id": lesson.lesson_id,
        "status": status_label(lesson),   # 派生：active / dormant
        "score": compute_score(lesson),   # 派生：注入分
        "failure_class": lesson.trigger.failure_class,
        "op": lesson.trigger.op,
        "stats": lesson.stats.to_dict(),
        "evidence_episode_count": len(lesson.evidence.source_episode_ids),
        "updated_at": lesson.updated_at,
        "source_type": lesson.source_type,
        "advice": lesson.advice,
    }


def _print_human_row(lesson: RuntimeLesson) -> None:
    rec = lesson.advice or ""
    if len(rec) > 80:
        rec = rec[:77] + "..."
    tool = lesson.trigger.op or "(none)"
    print(
        f"[{lesson.lesson_id}] {status_label(lesson):8s} score={compute_score(lesson):<3d} "
        f"{lesson.trigger.failure_class:30s} {tool}\n"
        f"    {rec}"
    )


# ============================================================
# 子命令处理
# ============================================================


def _cmd_list(backend: MemoryBackend, args: argparse.Namespace) -> int:
    lessons = backend.search_lessons(limit=1000)
    if args.active:
        lessons = [l for l in lessons if is_active(l)]
    if args.json:
        print(json.dumps(
            [_lesson_to_summary_dict(l) for l in lessons],
            ensure_ascii=False,
            indent=2,
        ))
        return EXIT_OK
    if not lessons:
        print("(no lessons)")
        return EXIT_OK
    for lesson in lessons:
        _print_human_row(lesson)
    return EXIT_OK


def _cmd_show(backend: MemoryBackend, args: argparse.Namespace) -> int:
    lesson = backend.get_lesson(args.lesson_id)
    if lesson is None:
        print(f"[error] lesson_id={args.lesson_id!r} not found", file=sys.stderr)
        return EXIT_NOT_FOUND
    if args.json:
        print(json.dumps(_lesson_to_summary_dict(lesson), ensure_ascii=False, indent=2))
        return EXIT_OK
    for k, v in _lesson_to_summary_dict(lesson).items():
        print(f"  {k}: {v}")
    return EXIT_OK


# ============================================================
# argparse 装配
# ============================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoagent.lesson",
        description="Lesson 只读运维 CLI（list / show；生命周期由账本派生分决定,无手动状态操作）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出 lesson")
    p_list.add_argument("--active", action="store_true", help="只列 active（score≥T）")
    p_list.add_argument("--json", action="store_true", help="输出 JSON 格式")

    p_show = sub.add_parser("show", help="查看一条 lesson 详情")
    p_show.add_argument("lesson_id")
    p_show.add_argument("--json", action="store_true")

    return parser


_DISPATCH = {
    "list": _cmd_list,
    "show": _cmd_show,
}


def main(
    argv: Optional[list[str]] = None,
    backend: Optional[MemoryBackend] = None,
) -> int:
    """CLI 入口。backend 可注入（测试用 InMemory）；None → SqliteMemoryBackend。

    注意：本函数**不**调 sys.stdout.reconfigure——会污染 pytest capsys 的 tee。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    owned_backend = backend is None
    if backend is None:
        backend = SqliteMemoryBackend(db_path=DEFAULT_DB_PATH)
    try:
        return _DISPATCH[args.cmd](backend, args)
    finally:
        if owned_backend:
            backend.close()
