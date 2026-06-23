"""MemoryBackend 抽象接口。

InMemoryMemoryBackend 跑通 policy；SqliteMemoryBackend 作为持久化默认实现。
两者实现同一个 ABC，pytest 共享契约 fixture（test/runtime_memory/conftest.py）
跑同一套测试保证行为一致。

历史：曾实现 Mem0MemoryBackend 评估 mem0 OSS 路线，最终因 v2.0.1 filter
行为与本项目 workload 不匹配而回退。

异常约定：
- `get_*` 返回 None 不 raise（读路径友好）
- `update_*` raise LessonNotFound（写路径需明确契约）
- `add_lesson` 重复 raise LessonAlreadyExists
- 所有自定义异常继承 RuntimeMemoryError
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, FrozenSet, List, Optional

from nanoagent.evolution.runtime_memory.schema import (
    LessonStats,
    LessonStatus,
    RuntimeEpisode,
    RuntimeLesson,
)


# ============================================================
# 异常
# ============================================================


class RuntimeMemoryError(Exception):
    """Runtime memory 子系统的通用基类。"""


class LessonNotFound(RuntimeMemoryError):
    """update / 显式 get-with-raise 时 lesson_id 不存在。"""


class LessonAlreadyExists(RuntimeMemoryError):
    """add_lesson 时 lesson_id 已存在。upsert 走 update_lesson_metadata。"""


class IllegalStatusTransition(RuntimeMemoryError):
    """update_lesson_metadata 收到非法 status 转移时 raise。"""


# ============================================================
# 状态机：合法转移表
# ============================================================
#
# 出边为空 = 终态；自环（X → X）按定义非法（表里不写自环条目）。
# CLI 通过 promote/retire/reset/expire 4 个动作走这些转移；
# OutcomeTracker 走 status=None 路径，不进 validator。
LEGAL_TRANSITIONS: Dict[LessonStatus, FrozenSet[LessonStatus]] = {
    LessonStatus.CANDIDATE: frozenset({
        LessonStatus.PROBATION,  # 先升 PROBATION 试用，不直接 PROMOTED
        LessonStatus.PROMOTED,   # 保留：管理 CLI 可手动直 promote（紧急场景）
        LessonStatus.RETIRED, LessonStatus.EXPIRED,
    }),
    LessonStatus.PROMOTED: frozenset({
        LessonStatus.RETIRED, LessonStatus.CANDIDATE, LessonStatus.EXPIRED,
    }),
    LessonStatus.RETIRED: frozenset({
        LessonStatus.CANDIDATE, LessonStatus.EXPIRED,
    }),
    LessonStatus.EXPIRED: frozenset(),  # 终态
    LessonStatus.PROBATION: frozenset({
        LessonStatus.PROMOTED, LessonStatus.RETIRED,
        LessonStatus.CANDIDATE, LessonStatus.EXPIRED,
    }),
}


def validate_status_transition(from_: LessonStatus, to: LessonStatus) -> None:
    """raise IllegalStatusTransition if (from_ → to) not allowed."""
    allowed = LEGAL_TRANSITIONS.get(from_, frozenset())
    if to not in allowed:
        raise IllegalStatusTransition(
            f"非法状态转移：{from_.value} → {to.value}（合法目标：{sorted(s.value for s in allowed) or '无（终态）'}）"
        )


# ============================================================
# ABC 接口
# ============================================================


class MemoryBackend(ABC):
    """Episode + Lesson 的存储/检索接口。

    实现：
    - `InMemoryMemoryBackend`（dict 内存）
    - `SqliteMemoryBackend`（默认持久化 backend）

    所有实现必须满足"对象别名隔离"语义：`add_lesson` 之后 caller 持有的对象
    再 mutate 不应影响 backend 内部状态；`get_lesson` 返回的对象 mutate
    也不应回写——一切持久化更新必须走 `update_lesson_metadata` /
    `extend_lesson_evidence` 显式接口。
    """

    # ---- Episode ----

    @abstractmethod
    def add_episode(self, episode: RuntimeEpisode) -> str:
        """持久化 episode，返回 episode_id。重复 episode_id 行为由实现定义
        （当前 InMemory / SQLite 都是 overwrite/upsert 简化操作）。"""

    @abstractmethod
    def get_episode(self, episode_id: str) -> Optional[RuntimeEpisode]:
        """不存在返回 None。"""

    # ---- Lesson CRUD ----

    @abstractmethod
    def add_lesson(self, lesson: RuntimeLesson) -> str:
        """新增 lesson；若 lesson_id 已存在 raise LessonAlreadyExists。

        实现必须做对象隔离（深拷贝 / 序列化往返），caller 之后 mutate 传入的
        lesson 实例不应影响已存储数据。"""

    @abstractmethod
    def get_lesson(self, lesson_id: str) -> Optional[RuntimeLesson]:
        """不存在返回 None。

        返回值必须与内部存储对象隔离——caller mutate 返回值不能回写。"""

    @abstractmethod
    def search_lessons(
        self,
        filters: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None,
        limit: int = 50,
    ) -> List[RuntimeLesson]:
        """按 filters + query 搜索 lesson。

        filters grammar（InMemory + SQLite 共同覆盖）：
            filter   := leaf | logical
            leaf     := { "<dotted.field>": <value> }                # eq
                     | { "<dotted.field>": { "in": [...] } }
                     | { "<dotted.field>": { "ne": value } }
            logical  := { "AND": [filter, ...] }
                     | { "OR":  [filter, ...] }
                     | { "NOT": filter }

        字段路径用点号支持嵌套：`"trigger.tool_name"` / `"status"`——
        与 RuntimeLesson dataclass attr 路径一一对应（避免双轨）。
        `query` 当前在 InMemory + SQLite 走 advice substring fallback（memory_text
        已删）；未来增强（FTS5 / sqlite-vec / 语义检索）由具体 backend 决定。
        """

    def update_lesson_metadata(
        self,
        lesson_id: str,
        *,
        status: Optional[LessonStatus] = None,
        stats: Optional[LessonStats] = None,
        expires_on: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> RuntimeLesson:
        """部分字段更新；不存在 raise LessonNotFound。返回更新后的 lesson。
        None 字段表示 "不动"，不是 "清空"。

        模板方法（concrete）：
        - 若 `status is not None`：先 get_lesson 取当前态 + validate_status_transition；
          status=None（OutcomeTracker 路径）跳过 validator——这是显式契约。
        - 通过后 dispatch 给具体 backend 的 `_apply_lesson_metadata`。

        非法转移 raise IllegalStatusTransition；lesson_id 不存在 raise LessonNotFound。
        confidence 参数给 OutcomeTracker / PromotionGate 显式更新；不要靠 mutate
        get_lesson 返回的对象（实现保证别名隔离）。"""
        if status is not None:
            current = self.get_lesson(lesson_id)
            if current is None:
                raise LessonNotFound(f"lesson_id={lesson_id!r} not found")
            validate_status_transition(current.status, status)
        return self._apply_lesson_metadata(
            lesson_id,
            status=status,
            stats=stats,
            expires_on=expires_on,
            confidence=confidence,
        )

    @abstractmethod
    def _apply_lesson_metadata(
        self,
        lesson_id: str,
        *,
        status: Optional[LessonStatus] = None,
        stats: Optional[LessonStats] = None,
        expires_on: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> RuntimeLesson:
        """具体 backend 落盘 metadata 更新；不做 status 转移校验（已在模板方法层处理）。
        不存在仍需 raise LessonNotFound（防御性——template 已检查但避免双 fetch
        race / 直接调本方法时漏检）。"""

    @abstractmethod
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
        """累加跨 trace 的 evidence。

        - 若 episode_id 不在 evidence.source_episode_ids → 追加（幂等：已有则跳过）
        - 若 sample_* 字段非 None 且当前 lesson 的对应 sample_* 为空/默认 → 写入
          （第一次有 sample 时填，避免后续覆盖原始证据）
        - 若 example 非 None 且 lesson.example 为 None → 写入（首次有 canonical
          结构化示范时填；后续不覆盖，避免被 partial / 错误 schema 污染）
        - 不存在 raise LessonNotFound

        典型调用方：caller 先 add_lesson 抓 LessonAlreadyExists 后调本方法。
        """

    @abstractmethod
    def list_expired(self, today_iso: str) -> List[RuntimeLesson]:
        """返回 expires_on < today_iso 且 status != EXPIRED 的全部 lesson。

        today_iso 形如 "YYYY-MM-DD"。后续 cleanup pruner 用此找需要 mark
        EXPIRED 或物理删除的目标。
        """

    @abstractmethod
    def delete_lesson(self, lesson_id: str) -> bool:
        """物理删除 lesson；不存在返回 False（幂等清理）。"""
