"""SqliteMemoryBackend — SQLite 持久化实现。

InMemoryMemoryBackend 重启即丢；SqliteMemoryBackend 用 SQLite 文件做真持久化，
满足同一 `MemoryBackend` ABC。

设计要点：
- 2 表：`runtime_episodes` + `runtime_lessons`
- 完整 dataclass 序列化进 `payload_json` 列（schema 演进零 ALTER TABLE 成本）
- 高频 filter 字段（status / error_class / tool_name / expires_on / confidence / updated_at）
  提前出来做 SQL pre-cut + indexed
- 复杂 filter 表达式（嵌套 OR/NOT、点号路径）→ 客户端 `_match` 复用 InMemory 的实现
  保证两个 backend 行为零差异
- WAL mode 单 connection；显式 `close()` commit + 关连接

暂不做：
- FTS5 / sqlite-vec
- schema migration（第一版手动 DROP TABLE 重建即可）
"""

from __future__ import annotations

import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from nanoagent.core.logger import get_logger
from nanoagent.evolution.runtime_memory.backend import (
    LessonAlreadyExists,
    LessonNotFound,
    MemoryBackend,
)
from nanoagent.evolution.runtime_memory.inmemory_backend import _match
from nanoagent.evolution.runtime_memory.schema import (
    LessonStats,
    LessonStatus,
    RuntimeEpisode,
    RuntimeLesson,
)

_logger = get_logger("sqlite_backend")

DEFAULT_DB_PATH: Path = Path("data/runtime/lessons/runtime_memory.sqlite")

# Indexed 列字段路径 → SQL 列名映射。SQL pre-cut 走这些路径；
# 路径与 RuntimeLesson 的 dataclass attr 一致（`trigger.X` 走嵌套）——
# 单一来源避免与客户端 `_match` 路径解析双轨。
_INDEXED_FIELDS_TO_COLUMNS: Dict[str, str] = {
    "status": "status",
    "expires_on": "expires_on",
    "confidence": "confidence",
    "updated_at": "updated_at",
    "trigger.error_class": "error_class",
    "trigger.tool_name": "tool_name",
}

_SCHEMA_DDL: str = """
CREATE TABLE IF NOT EXISTS runtime_episodes (
    episode_id   TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    agent_name   TEXT,
    started_at   TEXT
);

CREATE TABLE IF NOT EXISTS runtime_lessons (
    lesson_id    TEXT PRIMARY KEY,
    payload_json TEXT NOT NULL,
    status       TEXT NOT NULL,
    error_class  TEXT NOT NULL,
    tool_name    TEXT,
    expires_on   TEXT,
    confidence   REAL DEFAULT 0.0,
    updated_at   TEXT
);

CREATE INDEX IF NOT EXISTS idx_lessons_status     ON runtime_lessons(status);
CREATE INDEX IF NOT EXISTS idx_lessons_expires    ON runtime_lessons(expires_on);
CREATE INDEX IF NOT EXISTS idx_lessons_tool       ON runtime_lessons(tool_name);
CREATE INDEX IF NOT EXISTS idx_lessons_err_class  ON runtime_lessons(error_class);
CREATE INDEX IF NOT EXISTS idx_lessons_confidence ON runtime_lessons(confidence);
CREATE INDEX IF NOT EXISTS idx_lessons_updated    ON runtime_lessons(updated_at);
"""


class SqliteMemoryBackend(MemoryBackend):
    """SQLite-backed RuntimeMemory 实现。

    构造方式：
        backend = SqliteMemoryBackend()  # 默认路径 data/runtime/lessons/runtime_memory.sqlite
        backend = SqliteMemoryBackend(db_path=Path("custom.sqlite"))

    ContextManager 也可以：
        with SqliteMemoryBackend(db_path=...) as backend:
            backend.add_lesson(...)
    """

    def __init__(
        self,
        db_path: Union[Path, str] = DEFAULT_DB_PATH,
        *,
        check_same_thread: bool = True,
    ):
        """初始化 SQLite backend。

        check_same_thread:
            默认 True（sqlite3 默认行为，CLI 单线程）。
            channel 场景（如 Telegram bot）main thread 装配 + worker thread
            访问 → 需显式传 False。run_bot.py 用单 worker ThreadPoolExecutor
            把所有 handle 串在同一 worker thread，配合 False 让 connection
            可跨"装配 thread → 单 worker thread"使用。多 worker 并发写仍
            不安全（v2 当前单 worker，未来需要并发再加锁）。
        """
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None → autocommit；显式开 WAL；text_factory 强制 utf-8
        self._conn: Optional[sqlite3.Connection] = sqlite3.connect(
            str(self._db_path),
            isolation_level=None,
            check_same_thread=check_same_thread,
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(_SCHEMA_DDL)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.commit()
                self._conn.close()
            except sqlite3.Error as e:
                _logger.warning(f"[sqlite_backend] close 期间异常（已忽略）: {e}")
            self._conn = None

    def __enter__(self) -> "SqliteMemoryBackend":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __del__(self) -> None:
        # 防御性兜底；正常使用应显式 close()
        try:
            self.close()
        except Exception:
            pass

    # ============================================================
    # Episode CRUD
    # ============================================================

    def add_episode(self, episode: RuntimeEpisode) -> str:
        assert self._conn is not None
        payload = json.dumps(episode.to_dict(), ensure_ascii=False)
        self._conn.execute(
            "INSERT OR REPLACE INTO runtime_episodes "
            "(episode_id, payload_json, agent_name, started_at) VALUES (?, ?, ?, ?)",
            (episode.episode_id, payload, episode.agent_name, episode.started_at),
        )
        return episode.episode_id

    def get_episode(self, episode_id: str) -> Optional[RuntimeEpisode]:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT payload_json FROM runtime_episodes WHERE episode_id = ?",
            (episode_id,),
        ).fetchone()
        if row is None:
            return None
        return RuntimeEpisode.from_dict(json.loads(row[0]))

    # ============================================================
    # Lesson CRUD
    # ============================================================

    def add_lesson(self, lesson: RuntimeLesson) -> str:
        assert self._conn is not None
        existing = self._conn.execute(
            "SELECT 1 FROM runtime_lessons WHERE lesson_id = ?", (lesson.lesson_id,)
        ).fetchone()
        if existing is not None:
            raise LessonAlreadyExists(
                f"lesson_id={lesson.lesson_id!r} already exists"
            )
        payload = json.dumps(lesson.to_dict(), ensure_ascii=False)
        self._conn.execute(
            "INSERT INTO runtime_lessons "
            "(lesson_id, payload_json, status, error_class, tool_name, "
            " expires_on, confidence, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                lesson.lesson_id,
                payload,
                lesson.status.value,
                lesson.trigger.error_class,
                lesson.trigger.tool_name,
                lesson.expires_on,
                lesson.confidence,
                lesson.updated_at,
            ),
        )
        return lesson.lesson_id

    def get_lesson(self, lesson_id: str) -> Optional[RuntimeLesson]:
        assert self._conn is not None
        row = self._conn.execute(
            "SELECT payload_json FROM runtime_lessons WHERE lesson_id = ?",
            (lesson_id,),
        ).fetchone()
        if row is None:
            return None
        return RuntimeLesson.from_dict(json.loads(row[0]))

    def search_lessons(
        self,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> List[RuntimeLesson]:
        assert self._conn is not None
        where_clauses, params = _build_sql_pre_cut(filters)
        sql = "SELECT payload_json FROM runtime_lessons"
        if where_clauses:
            sql += " WHERE " + " AND ".join(where_clauses)
        rows = self._conn.execute(sql, params).fetchall()

        lessons = [RuntimeLesson.from_dict(json.loads(r[0])) for r in rows]
        # 客户端 _match 复审完整 filter 表达式（嵌套 OR/NOT 等）
        if filters:
            lessons = [l for l in lessons if _match(l, filters)]
        if query:
            q = query.lower()
            lessons = [l for l in lessons if q in l.memory_text.lower()]
        return lessons[:limit]

    def _apply_lesson_metadata(
        self,
        lesson_id: str,
        *,
        status: Optional[LessonStatus] = None,
        stats: Optional[LessonStats] = None,
        expires_on: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> RuntimeLesson:
        # 状态机校验在 ABC 模板方法 update_lesson_metadata 已处理
        assert self._conn is not None
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            raise LessonNotFound(f"lesson_id={lesson_id!r} not found")
        if status is not None:
            lesson.status = status
        if stats is not None:
            lesson.stats = stats
        if expires_on is not None:
            lesson.expires_on = expires_on
        if confidence is not None:
            lesson.confidence = confidence
        lesson.updated_at = _now_iso()
        payload = json.dumps(lesson.to_dict(), ensure_ascii=False)
        self._conn.execute(
            "UPDATE runtime_lessons SET "
            "payload_json = ?, status = ?, expires_on = ?, "
            "confidence = ?, updated_at = ? "
            "WHERE lesson_id = ?",
            (
                payload,
                lesson.status.value,
                lesson.expires_on,
                lesson.confidence,
                lesson.updated_at,
                lesson_id,
            ),
        )
        return lesson

    def extend_lesson_evidence(
        self,
        lesson_id: str,
        *,
        episode_id: str,
        sample_trace_path: Optional[str] = None,
        sample_failure_iteration: Optional[int] = None,
        sample_error_message: Optional[str] = None,
        repair_example: Optional[Dict[str, Any]] = None,
    ) -> RuntimeLesson:
        assert self._conn is not None
        lesson = self.get_lesson(lesson_id)
        if lesson is None:
            raise LessonNotFound(f"lesson_id={lesson_id!r} not found")
        ev = lesson.evidence
        if episode_id not in ev.source_episode_ids:
            ev.source_episode_ids.append(episode_id)
        if sample_trace_path and not ev.sample_trace_path:
            ev.sample_trace_path = sample_trace_path
        if sample_failure_iteration is not None and ev.sample_failure_iteration in (
            None,
            0,
            -1,
        ):
            ev.sample_failure_iteration = sample_failure_iteration
        if sample_error_message and not ev.sample_error_message:
            ev.sample_error_message = sample_error_message[:300]
        # 首次写入 repair_example / suggested_action；不覆盖。
        if repair_example is not None:
            if ev.repair_example is None:
                ev.repair_example = repair_example
            if lesson.suggested_action is None:
                lesson.suggested_action = repair_example
        lesson.updated_at = _now_iso()
        payload = json.dumps(lesson.to_dict(), ensure_ascii=False)
        self._conn.execute(
            "UPDATE runtime_lessons SET payload_json = ?, updated_at = ? "
            "WHERE lesson_id = ?",
            (payload, lesson.updated_at, lesson_id),
        )
        return lesson

    def list_expired(self, today_iso: str) -> List[RuntimeLesson]:
        assert self._conn is not None
        rows = self._conn.execute(
            "SELECT payload_json FROM runtime_lessons "
            "WHERE expires_on < ? AND status != ?",
            (today_iso, LessonStatus.EXPIRED.value),
        ).fetchall()
        return [RuntimeLesson.from_dict(json.loads(r[0])) for r in rows]

    def delete_lesson(self, lesson_id: str) -> bool:
        assert self._conn is not None
        cur = self._conn.execute(
            "DELETE FROM runtime_lessons WHERE lesson_id = ?", (lesson_id,)
        )
        return cur.rowcount > 0


# ============================================================
# SQL pre-cut helpers
# ============================================================


def _build_sql_pre_cut(
    filters: Optional[Dict[str, Any]],
) -> Tuple[List[str], List[Any]]:
    """从 filter 树中提取顶层 AND 子句下的"扁平 eq leaf"作为 SQL WHERE 条件。

    保守策略：只识别两种形态——
      1. 顶层是单个 leaf：`{"status": "promoted"}`
      2. 顶层是 AND：`{"AND": [leaf1, leaf2, ...]}`，所有 leaf 是简单 eq
    其他形态（OR / NOT / 嵌套 AND / `{spec: dict}` 复杂运算）直接返回空，
    让客户端 `_match` 全负责——保证零行为差异。
    """
    if not filters:
        return ([], [])

    # 顶层是单 leaf（无 AND/OR/NOT key）
    if not any(k in filters for k in ("AND", "OR", "NOT")):
        return _extract_indexed_leaf(filters)

    # 顶层是单一 AND
    if "AND" in filters and len(filters) == 1:
        clauses: List[str] = []
        params: List[Any] = []
        for child in filters["AND"]:
            ch_clauses, ch_params = _extract_indexed_leaf(child)
            clauses.extend(ch_clauses)
            params.extend(ch_params)
        return (clauses, params)

    return ([], [])


def _extract_indexed_leaf(
    leaf: Dict[str, Any],
) -> Tuple[List[str], List[Any]]:
    """识别一个 leaf；若是 indexed 列的简单 eq 则翻译为 SQL；其他放弃。"""
    if not isinstance(leaf, dict) or len(leaf) != 1:
        return ([], [])
    field, spec = next(iter(leaf.items()))
    if field not in _INDEXED_FIELDS_TO_COLUMNS:
        return ([], [])
    if isinstance(spec, dict):
        # in / ne / contains 等不识别，全交给客户端
        return ([], [])
    column = _INDEXED_FIELDS_TO_COLUMNS[field]
    return ([f"{column} = ?"], [spec])


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")
