"""runtime memory schema。

定义 trace-backed lesson 系统的核心 dataclass：
- `RuntimeEpisode` / `FailureEvent`：从 trace 抽出的"发生了什么"
- `RuntimeLesson` / `LessonTrigger` / `LessonEvidence` / `LessonStats`：可注入的"以后该怎么办"
- `LessonStatus` enum：候选 → 试用 → 升级 → 退役 / 过期 的生命周期

设计取舍：
- `@dataclass` 非 frozen（项目约定，参考 `runtime/failure_memory::FailureEntry`）
- `LessonStatus(str, Enum)` 便于 JSON 序列化
- `expires_on` 用 ISO date string ("YYYY-MM-DD")，TTL 比较只到天且 mem0
  metadata filter 对 string 友好；`created_at`/`updated_at` 用完整 ISO datetime
  保留秒级精度——不一致是有意的
- `memory_text` vs `recommendation` 双字段：前者给 mem0.add() 用（自然语言长描述），
  后者给 LessonInjector 用（短句注入 system prompt）
- 不引入 dataclass-json / Pydantic 等依赖，序列化自己写
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# LessonStatus 状态机
# ============================================================


class LessonStatus(str, Enum):
    """Lesson 生命周期。

    - CANDIDATE：刚生成，未经验证
    - PROBATION：试用观察期
    - PROMOTED：通过 promotion gate，可被 retriever 检索注入
    - RETIRED：累计负面 outcome 多 / 被更好 lesson 覆盖，停止注入但保留审计
    - EXPIRED：超过 expires_on 自动判定（走 cleanup pruner）
    """

    CANDIDATE = "candidate"
    PROBATION = "probation"
    PROMOTED = "promoted"
    RETIRED = "retired"
    EXPIRED = "expired"


# ============================================================
# Episode 侧：从 trace 抽出的"发生了什么"
# ============================================================


@dataclass
class FailureEvent:
    """单条失败事件。一个 episode 可能有 0..N 条。

    raw_action 取自 `core/trace_schema::ALL_ACTIONS`，常见值：
    - "tool_call_end"（output 含失败签名）
    - "failure_recovery_hint"（失败恢复 augment 触发）
    - "coverage_hint_injected"（硬覆盖不达标）
    - "evaluator_call_end"（副 LLM 判失败）
    """

    iteration: int
    tool_key: str  # "skill_exec:ai-digest/dup_check" / "load_skill" / "" (coverage/evaluator 无 tool)
    args_hash: str  # 12 char sha256 of canonical JSON args；coverage/evaluator 用 ""
    error_type: str  # "schema_mismatch" / "transient" / "unknown" / "coverage_gap" / "soft_quality"
    error_message: str  # 截断到 300 char
    timestamp: str
    raw_action: str  # ALL_ACTIONS 之一
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureEvent":
        return cls(**data)


@dataclass
class RuntimeEpisode:
    """一次 trace 的结构化产物。无 failure 的 trace 不会被建为 episode。"""

    episode_id: str  # sha1(trace_path + started_at)[:16]，确定性可重跑
    trace_path: str
    agent_name: str
    user_input: str  # 截断到 500 char
    started_at: str
    finished_at: str
    stop_reason: Optional[str]  # StopReason.value 或 None
    total_steps: int
    failures: List[FailureEvent]
    coverage_hits: List[Dict[str, Any]]  # 全部 coverage_check 事件原文（不只失败）
    evaluator_verdicts: List[Dict[str, Any]]  # 全部 evaluator_call_end 原文
    raw_summary: Dict[str, Any]  # run_summary 整段（tokens / total_steps 等）

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "trace_path": self.trace_path,
            "agent_name": self.agent_name,
            "user_input": self.user_input,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "stop_reason": self.stop_reason,
            "total_steps": self.total_steps,
            "failures": [f.to_dict() for f in self.failures],
            "coverage_hits": list(self.coverage_hits),
            "evaluator_verdicts": list(self.evaluator_verdicts),
            "raw_summary": dict(self.raw_summary),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeEpisode":
        return cls(
            episode_id=data["episode_id"],
            trace_path=data["trace_path"],
            agent_name=data["agent_name"],
            user_input=data["user_input"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            stop_reason=data.get("stop_reason"),
            total_steps=data["total_steps"],
            failures=[FailureEvent.from_dict(f) for f in data.get("failures", [])],
            coverage_hits=list(data.get("coverage_hits", [])),
            evaluator_verdicts=list(data.get("evaluator_verdicts", [])),
            raw_summary=dict(data.get("raw_summary", {})),
        )


# ============================================================
# Lesson 侧：可注入的"以后该怎么办"
# ============================================================


@dataclass
class LessonTrigger:
    """Lesson 何时该被检索/注入的条件。

    `error_class` 必填（5 种之一）；`tool_name` / `task_type` Optional——
    coverage_gap / soft_quality_issue 类不绑工具，task_type 等 router 接入再填。
    `failure_count_gte` 默认 1；只有 repeated_same_args_failure 模板会设 2。
    `scope` 用 string 不用 enum，留 "global" / "agent:<name>" / "skill:<name>" 扩展余地。
    """

    error_class: str
    task_type: Optional[str] = None
    tool_name: Optional[str] = None
    failure_count_gte: int = 1
    scope: str = "global"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonTrigger":
        return cls(**data)


@dataclass
class LessonEvidence:
    """Lesson 的"为什么"——可追溯到具体 trace。

    `repair_example`：本次失败现场 `_skill_arg_validator` 算出的完整结构化
    修复示例 `{"skill", "script", "args"}`。模板化 recommendation 会把错误信息
    截到 120 char 丢掉 example_call JSON 块；这里以 dict 完整保留跟着 evidence
    走。lesson 级聚合后的 canonical 修复示例放 RuntimeLesson.suggested_action。
    """

    source_episode_ids: List[str]
    sample_trace_path: str
    sample_failure_iteration: int
    sample_args_hash: str
    sample_error_message: str  # 截断 300 char
    repair_example: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonEvidence":
        # 旧 SQLite payload_json 没有 repair_example 字段；用 .get 兜底默认 None。
        return cls(
            source_episode_ids=list(data.get("source_episode_ids", [])),
            sample_trace_path=data.get("sample_trace_path", ""),
            sample_failure_iteration=data.get("sample_failure_iteration", -1),
            sample_args_hash=data.get("sample_args_hash", ""),
            sample_error_message=data.get("sample_error_message", ""),
            repair_example=data.get("repair_example"),
        )


@dataclass
class LessonStats:
    """Lesson 触发后的 outcome 统计。OutcomeTracker 填。

    `ineffective_count`：lesson 召回后 LLM 仍生成被 RepairGate 拦截的
    同类非法调用。**不进** success/failure，confidence 不变；累积到阈值再
    PROMOTED → CANDIDATE 自动降级（在 OutcomeTracker 里判定）。
    """

    hit_count: int = 0
    success_after_hit: int = 0
    failure_after_hit: int = 0
    ineffective_count: int = 0
    last_hit_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonStats":
        # 旧 SQLite payload_json 没有 ineffective_count 字段；用 .get 兜底默认 0。
        return cls(
            hit_count=data.get("hit_count", 0),
            success_after_hit=data.get("success_after_hit", 0),
            failure_after_hit=data.get("failure_after_hit", 0),
            ineffective_count=data.get("ineffective_count", 0),
            last_hit_at=data.get("last_hit_at"),
        )


@dataclass
class RuntimeLesson:
    """一条 conditional lesson。从 episode + error_class 用模板生成。

    `lesson_id` = sha1(error_class + tool_key + args_hash)[:16] —— **语义键**：
    同一种失败模式（不同 trace、不同 episode 但同 error_class + tool + args_hash）
    共享同一 lesson_id；跨 trace 的 evidence 累加靠 backend `extend_lesson_evidence`，
    `evidence.source_episode_ids` 是 list 即为支持这种聚合。
    """

    lesson_id: str
    memory_text: str  # 自然语言长描述，给 mem0.add() 用
    recommendation: str  # 短句，给 LessonInjector 注入 system prompt
    trigger: LessonTrigger
    evidence: LessonEvidence
    created_at: str  # ISO datetime
    updated_at: str  # ISO datetime
    expires_on: str  # ISO date "YYYY-MM-DD"
    status: LessonStatus = LessonStatus.CANDIDATE
    stats: LessonStats = field(default_factory=LessonStats)
    ttl_days: int = 14
    confidence: float = 0.0  # OutcomeTracker 计算 = times_helped / max(1, times_triggered)；PromotionGate 阈值 ~0.6
    tags: List[str] = field(default_factory=list)
    # 跨 evidence 聚合后的 canonical 修复示例。
    # 形如 {"skill", "script", "args"} dict（同 LessonEvidence.repair_example）。
    # LessonRetriever 在召回时附加 [structured-repair-hint] JSON 块给 LLM——
    # 区别于 recommendation 短文本：这是结构化"下一步该这么调"的具体方案。
    # LessonIngestor 决定何时刷新（首次写入 / 之前 None / 显式覆盖）。
    suggested_action: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "memory_text": self.memory_text,
            "recommendation": self.recommendation,
            "trigger": self.trigger.to_dict(),
            "evidence": self.evidence.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_on": self.expires_on,
            "status": self.status.value,
            "stats": self.stats.to_dict(),
            "ttl_days": self.ttl_days,
            "confidence": self.confidence,
            "tags": list(self.tags),
            "suggested_action": self.suggested_action,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeLesson":
        return cls(
            lesson_id=data["lesson_id"],
            memory_text=data["memory_text"],
            recommendation=data["recommendation"],
            trigger=LessonTrigger.from_dict(data["trigger"]),
            evidence=LessonEvidence.from_dict(data["evidence"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            expires_on=data["expires_on"],
            status=LessonStatus(data.get("status", LessonStatus.CANDIDATE.value)),
            stats=LessonStats.from_dict(data.get("stats", {})),
            ttl_days=data.get("ttl_days", 14),
            confidence=data.get("confidence", 0.0),
            tags=list(data.get("tags", [])),
            # 旧 SQLite payload_json 没有此字段；.get 兜底 None。
            suggested_action=data.get("suggested_action"),
        )
