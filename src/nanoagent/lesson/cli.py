"""lesson 管理 CLI（手动状态机操作）。

子命令：list / show / promote / retire / reset / expire
状态机校验在 backend ABC 模板方法层（IllegalStatusTransition）；CLI 只负责 IO + exit code。

Exit code:
    0 = 成功
    1 = lesson_id not found
    2 = illegal status transition
    3 = 参数错误（argparse 默认）

不在本 CLI 范围：auto-promote / auto-retire / cooldown / Bayesian 阈值 / cron sweep / PROBATION 启用。
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional

from nanoagent.evolution.runtime_memory.backend import (
    IllegalStatusTransition,
    LessonNotFound,
    MemoryBackend,
)
from nanoagent.evolution.runtime_memory.schema import LessonStatus, RuntimeLesson
from nanoagent.evolution.runtime_memory.sqlite_backend import (
    DEFAULT_DB_PATH,
    SqliteMemoryBackend,
)


EXIT_OK = 0
EXIT_NOT_FOUND = 1
EXIT_ILLEGAL_TRANSITION = 2

_VALID_STATUSES = [s.value for s in LessonStatus]


def _lesson_to_summary_dict(lesson: RuntimeLesson) -> dict:
    """list / show 共用的稳定 JSON schema。下游脚本会依赖这套字段。"""
    return {
        "lesson_id": lesson.lesson_id,
        "status": lesson.status.value,
        "error_class": lesson.trigger.error_class,
        "tool_name": lesson.trigger.tool_name,
        "confidence": lesson.confidence,
        "stats": lesson.stats.to_dict(),
        "evidence_episode_count": len(lesson.evidence.source_episode_ids),
        "expires_on": lesson.expires_on,
        "updated_at": lesson.updated_at,
        "source_type": lesson.source_type,
        "advice": lesson.advice,
    }


def _print_human_row(lesson: RuntimeLesson) -> None:
    rec = lesson.advice or ""
    if len(rec) > 80:
        rec = rec[:77] + "..."
    tool = lesson.trigger.tool_name or "(none)"
    print(
        f"[{lesson.lesson_id}] {lesson.status.value:10s} "
        f"{lesson.trigger.error_class:30s} {tool}\n"
        f"    {rec}"
    )


# ============================================================
# 子命令处理
# ============================================================


def _cmd_list(backend: MemoryBackend, args: argparse.Namespace) -> int:
    filters = None
    if args.status:
        filters = {"status": args.status}
    lessons = backend.search_lessons(filters=filters, limit=1000)
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
    summary = _lesson_to_summary_dict(lesson)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    return EXIT_OK


def _change_status(
    backend: MemoryBackend, lesson_id: str, target: LessonStatus
) -> int:
    try:
        before = backend.get_lesson(lesson_id)
        if before is None:
            print(f"[error] lesson_id={lesson_id!r} not found", file=sys.stderr)
            return EXIT_NOT_FOUND
        backend.update_lesson_metadata(lesson_id, status=target)
        print(f"[ok] {lesson_id}: {before.status.value} → {target.value}")
        return EXIT_OK
    except LessonNotFound:
        print(f"[error] lesson_id={lesson_id!r} not found", file=sys.stderr)
        return EXIT_NOT_FOUND
    except IllegalStatusTransition as e:
        print(f"[error] {e}", file=sys.stderr)
        return EXIT_ILLEGAL_TRANSITION


def _cmd_promote(backend: MemoryBackend, args: argparse.Namespace) -> int:
    return _change_status(backend, args.lesson_id, LessonStatus.PROMOTED)


def _cmd_retire(backend: MemoryBackend, args: argparse.Namespace) -> int:
    return _change_status(backend, args.lesson_id, LessonStatus.RETIRED)


def _cmd_reset(backend: MemoryBackend, args: argparse.Namespace) -> int:
    return _change_status(backend, args.lesson_id, LessonStatus.CANDIDATE)


def _cmd_expire(backend: MemoryBackend, args: argparse.Namespace) -> int:
    return _change_status(backend, args.lesson_id, LessonStatus.EXPIRED)


# ============================================================
# argparse 装配
# ============================================================


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nanoagent.lesson",
        description="Lesson 管理 CLI（手动 promote / retire / reset / expire）",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出 lesson")
    p_list.add_argument("--status", choices=_VALID_STATUSES, default=None,
                        help="过滤 status")
    p_list.add_argument("--json", action="store_true", help="输出 JSON 格式")

    p_show = sub.add_parser("show", help="查看一条 lesson 详情")
    p_show.add_argument("lesson_id")
    p_show.add_argument("--json", action="store_true")

    for cmd, helptext in [
        ("promote", "candidate/probation/retired → PROMOTED"),
        ("retire", "candidate/promoted/probation → RETIRED"),
        ("reset", "promoted/retired/probation → CANDIDATE"),
        ("expire", "任意非 EXPIRED → EXPIRED"),
    ]:
        sp = sub.add_parser(cmd, help=helptext)
        sp.add_argument("lesson_id")

    return parser


# 子命令到 handler 的映射
_DISPATCH = {
    "list": _cmd_list,
    "show": _cmd_show,
    "promote": _cmd_promote,
    "retire": _cmd_retire,
    "reset": _cmd_reset,
    "expire": _cmd_expire,
}


def main(
    argv: Optional[list[str]] = None,
    backend: Optional[MemoryBackend] = None,
) -> int:
    """CLI 入口。

    Args:
        argv: 命令行参数；None → sys.argv[1:]
        backend: 测试可注入 InMemory 等替代；None → SqliteMemoryBackend(DEFAULT_DB_PATH)

    Returns:
        exit code (0/1/2)

    注意：本函数**不**调 sys.stdout.reconfigure——会污染 pytest capsys 的 tee。
    控制台 utf-8 配置在 __main__.py 入口处理。
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    owned_backend = backend is None
    if backend is None:
        backend = SqliteMemoryBackend(db_path=DEFAULT_DB_PATH)
    try:
        handler = _DISPATCH[args.cmd]
        return handler(backend, args)
    finally:
        if owned_backend:
            backend.close()  # 仅 close 我们自己构造的 backend；caller 注入的归 caller 管
