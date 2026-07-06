"""InMemoryMemoryBackend — 内存实现。

不持久化、不语义检索；目的是跑通 policy 闭环，并作为 fast path 单测对照。
filter 引擎覆盖 LessonRetriever 预期的 AND/OR/NOT + eq/in/ne 嵌套语法。
"""

from __future__ import annotations

import datetime
import enum
from typing import Any, Dict, List, Optional

from nanoagent.evolution.runtime_memory.backend import (
    LessonAlreadyExists,
    LessonNotFound,
    MemoryBackend,
)
from nanoagent.evolution.runtime_memory.schema import (
    LessonStats,
    RuntimeEpisode,
    RuntimeLesson,
)


# ============================================================
# Filter 引擎（递归）
# ============================================================


_LOGICAL_OPS = ("AND", "OR", "NOT")


def _resolve_path(obj: Any, dotted: str) -> Any:
    """按点号路径取值。`getattr` 链式；遇 enum 取 .value；任何缺失返回 None。"""
    cur = obj
    for part in dotted.split("."):
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(part)
            continue
        cur = getattr(cur, part, None)
    if isinstance(cur, enum.Enum):
        return cur.value
    return cur


def _match(lesson: RuntimeLesson, node: Optional[Dict[str, Any]]) -> bool:
    """递归 match。node 为 None / 空 dict 视为 match-all。"""
    if not node:
        return True
    if "AND" in node:
        return all(_match(lesson, c) for c in node["AND"])
    if "OR" in node:
        return any(_match(lesson, c) for c in node["OR"])
    if "NOT" in node:
        return not _match(lesson, node["NOT"])
    if len(node) != 1:
        raise ValueError(
            f"leaf filter must have exactly one key (got {len(node)}): {node!r}"
        )
    field, spec = next(iter(node.items()))
    if field in _LOGICAL_OPS:
        # 防御：上面分支已处理；走到这说明组合不规范
        raise ValueError(f"logical op '{field}' must take a list/dict child")
    actual = _resolve_path(lesson, field)
    if isinstance(spec, dict):
        if "in" in spec:
            return actual in spec["in"]
        if "ne" in spec:
            return actual != spec["ne"]
        raise ValueError(f"unknown leaf op spec: {spec!r}")
    return actual == spec


# ============================================================
# Backend
# ============================================================


class InMemoryMemoryBackend(MemoryBackend):
    """dict 实现。重启即丢——单元测试 / smoke 专用。"""

    def __init__(self):
        self._episodes: Dict[str, RuntimeEpisode] = {}
        self._lessons: Dict[str, RuntimeLesson] = {}

    # ---- Episode ----

    def add_episode(self, episode: RuntimeEpisode) -> str:
        # 序列化往返做对象隔离——caller 之后 mutate 传入对象不应影响存储
        self._episodes[episode.episode_id] = RuntimeEpisode.from_dict(
            episode.to_dict()
        )
        return episode.episode_id

    def get_episode(self, episode_id: str) -> Optional[RuntimeEpisode]:
        stored = self._episodes.get(episode_id)
        if stored is None:
            return None
        return RuntimeEpisode.from_dict(stored.to_dict())

    # ---- Lesson CRUD ----

    def add_lesson(self, lesson: RuntimeLesson) -> str:
        if lesson.lesson_id in self._lessons:
            raise LessonAlreadyExists(
                f"lesson_id={lesson.lesson_id!r} already exists"
            )
        # 序列化往返做对象隔离——caller mutate 传入对象不影响内部
        self._lessons[lesson.lesson_id] = RuntimeLesson.from_dict(lesson.to_dict())
        return lesson.lesson_id

    def get_lesson(self, lesson_id: str) -> Optional[RuntimeLesson]:
        stored = self._lessons.get(lesson_id)
        if stored is None:
            return None
        # 返回拷贝——caller mutate 不能回写
        return RuntimeLesson.from_dict(stored.to_dict())

    def search_lessons(
        self,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> List[RuntimeLesson]:
        out: List[RuntimeLesson] = []
        q = query.lower() if query else None
        for lesson in self._lessons.values():
            if filters and not _match(lesson, filters):
                continue
            # memory_text 已删；query 子串 fallback 改派生在 canonical 的 advice 上
            if q and q not in (lesson.advice or "").lower():
                continue
            # 返回拷贝保持隔离
            out.append(RuntimeLesson.from_dict(lesson.to_dict()))
            if len(out) >= limit:
                break
        return out

    def _apply_lesson_metadata(
        self,
        lesson_id: str,
        *,
        stats: Optional[LessonStats] = None,
    ) -> RuntimeLesson:
        lesson = self._lessons.get(lesson_id)
        if lesson is None:
            raise LessonNotFound(f"lesson_id={lesson_id!r} not found")
        if stats is not None:
            lesson.stats = stats
        lesson.updated_at = _now_iso()
        # 返回拷贝
        return RuntimeLesson.from_dict(lesson.to_dict())

    def extend_lesson_evidence(
        self,
        lesson_id: str,
        *,
        episode_id: str,
        sample_trace_path: Optional[str] = None,
        sample_failure_iteration: Optional[int] = None,
        sample_error_message: Optional[str] = None,
        example: Optional[Dict[str, Any]] = None,
    ) -> RuntimeLesson:
        lesson = self._lessons.get(lesson_id)
        if lesson is None:
            raise LessonNotFound(f"lesson_id={lesson_id!r} not found")
        ev = lesson.evidence
        if episode_id not in ev.source_episode_ids:
            ev.source_episode_ids.append(episode_id)
        # 仅当 sample_* 当前缺失时填入，避免覆盖第一份原始证据
        if sample_trace_path and not ev.sample_trace_path:
            ev.sample_trace_path = sample_trace_path
        if sample_failure_iteration is not None and ev.sample_failure_iteration in (None, 0, -1):
            ev.sample_failure_iteration = sample_failure_iteration
        if sample_error_message and not ev.sample_error_message:
            ev.sample_error_message = sample_error_message[:300]
        # 首次写入 content.example（canonical 结构化示范）；不覆盖。
        if example is not None and lesson.example is None:
            lesson.example = example
        lesson.updated_at = _now_iso()
        return RuntimeLesson.from_dict(lesson.to_dict())

    def delete_lesson(self, lesson_id: str) -> bool:
        return self._lessons.pop(lesson_id, None) is not None


# ============================================================
# helpers
# ============================================================


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
