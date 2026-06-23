"""runtime memory schema。

定义 trace-backed lesson 系统的核心 dataclass：
- `RuntimeEpisode` / `FailureEvent`：从 trace 抽出的"发生了什么"
- `RuntimeLesson` / `LessonTrigger` / `LessonEvidence` / `LessonStats`：可注入的"以后该怎么办"
- `LessonStatus` enum：候选 → 试用 → 升级 → 退役 / 过期 的生命周期

设计取舍：
- `@dataclass` 非 frozen（项目约定，参考 `runtime/failure_memory::FailureEntry`）
- `LessonStatus(str, Enum)` 便于 JSON 序列化
- `expires_on` 用 ISO date string ("YYYY-MM-DD")，TTL 比较只到天；
  `created_at`/`updated_at` 用完整 ISO datetime 保留秒级精度——不一致是有意的
- 内容层 = 必填 `advice`（散文指令，= 旧 recommendation）+ 可选 `example`
  （结构化示范，= 旧 suggested_action，仅"形状类"修复才有）。`memory_text`
  （为已废弃 mem0 而生的死字段）已删。
- `source_type` ∈ {template, reflector, research, human}：来源一等枚举，
  从"靠 tags/前缀反推"提为显式声明。
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

    `error_class` 必填（5 种之一）；`tool_name` Optional——coverage_gap /
    soft_quality_issue 类不绑工具。
    `failure_count_gte` 默认 1；只有 repeated_same_args_failure 模板会设 2。
    `scope` 用 string 不用 enum，是方法论归属的正式键：
    "global" / "agent:<name>" / "skill:<name>"。
    `cause_sig`：确定性提取的"这次错在哪"（如 "missing:args"），是 lesson_id
    第三成分，现以可读字段存储（原只烘进 lesson_id hash、读不回、比不了、调不了
    粒度）；当前召回不读它做匹配，"按 cause_sig 匹配"留待后续。

    （`task_type` 已删：唯一写入点写 None、零读取；`condition_hint` 已删：
    100% 派生自 error_class+tool_name+cause_sig，改由 retriever 召回时现渲染。）
    """

    error_class: str
    tool_name: Optional[str] = None
    failure_count_gte: int = 1
    scope: str = "global"
    cause_sig: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonTrigger":
        # 显式取已知字段，容忍并忽略旧 payload 残留的 task_type / condition_hint。
        return cls(
            error_class=data["error_class"],
            tool_name=data.get("tool_name"),
            failure_count_gte=data.get("failure_count_gte", 1),
            scope=data.get("scope", "global"),
            cause_sig=data.get("cause_sig", ""),
        )


@dataclass
class LessonEvidence:
    """Lesson 的"为什么"——可追溯到具体 trace。

    canonical 结构化修复示例已搬进 RuntimeLesson.example（content 层）；这里
    只留回溯指针 + 实例证据（sample_args_hash / sample_error_message /
    sample_trace_path），那次具体失败实例仍能靠它们回溯。
    （`repair_example` 字段已删——结构化示范搬进 content.example。）
    """

    source_episode_ids: List[str]
    sample_trace_path: str
    sample_failure_iteration: int
    sample_args_hash: str
    sample_error_message: str  # 截断 300 char

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonEvidence":
        return cls(
            source_episode_ids=list(data.get("source_episode_ids", [])),
            sample_trace_path=data.get("sample_trace_path", ""),
            sample_failure_iteration=data.get("sample_failure_iteration", -1),
            sample_args_hash=data.get("sample_args_hash", ""),
            sample_error_message=data.get("sample_error_message", ""),
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

    `lesson_id` = sha1(error_class + tool_key + cause_sig)[:16] —— **原因层语义键**：
    同一种失败原因（不同 trace、不同参数，但同 error_class + tool + 原因签名）
    共享同一 lesson_id；跨 trace 的 evidence 累加靠 backend `extend_lesson_evidence`，
    `evidence.source_episode_ids` 是 list 即为支持这种聚合。args_hash 只留在
    evidence 作实例证据，不参与键。
    """

    lesson_id: str
    advice: str  # 散文指令（= 旧 recommendation），命中后注入失败现场的"该怎么做"
    trigger: LessonTrigger
    evidence: LessonEvidence
    created_at: str  # ISO datetime
    updated_at: str  # ISO datetime
    expires_on: str  # ISO date "YYYY-MM-DD"
    # 来源一等枚举：template | reflector | research | human。让"置信度该不该带
    # 来源先验"从哲学之争变可配置项（PromotionGate 未来读它配先验，零 schema 变更）。
    source_type: str = "template"
    status: LessonStatus = LessonStatus.CANDIDATE
    stats: LessonStats = field(default_factory=LessonStats)
    # 方法论 lesson 几乎不随时间失效，TTL 退化为安全阀（原 14 → 90）；
    # 将来按 source_type 配差异（research/环境类短、方法论类长）。
    ttl_days: int = 90
    confidence: float = 0.0  # Laplace 平滑 success/max(1,success+failure+1)（见 outcome_tracker）；PromotionGate 阈值 ~0.6
    # 可选结构化示范（= 旧 suggested_action）。形如 {"skill","script","args"} dict，
    # 仅"形状类"修复才有；LessonRetriever 召回时附加 [learned-example] 块给 LLM。
    # 来源 = RepairGate 的 gate_repair_example 直传。example is None 本身即携带
    # "只有指令/无示范"的区分，无需模式标志位。
    example: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lesson_id": self.lesson_id,
            "advice": self.advice,
            "trigger": self.trigger.to_dict(),
            "evidence": self.evidence.to_dict(),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "expires_on": self.expires_on,
            "source_type": self.source_type,
            "status": self.status.value,
            "stats": self.stats.to_dict(),
            "ttl_days": self.ttl_days,
            "confidence": self.confidence,
            "example": self.example,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RuntimeLesson":
        return cls(
            lesson_id=data["lesson_id"],
            advice=data["advice"],
            trigger=LessonTrigger.from_dict(data["trigger"]),
            evidence=LessonEvidence.from_dict(data["evidence"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            expires_on=data["expires_on"],
            source_type=data.get("source_type", "template"),
            status=LessonStatus(data.get("status", LessonStatus.CANDIDATE.value)),
            stats=LessonStats.from_dict(data.get("stats", {})),
            ttl_days=data.get("ttl_days", 90),
            confidence=data.get("confidence", 0.0),
            example=data.get("example"),
        )
