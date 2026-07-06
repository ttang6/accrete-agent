"""runtime memory schema。

定义 trace-backed lesson 系统的核心 dataclass：
- `RuntimeEpisode` / `FailureEvent`：从 trace 抽出的"发生了什么"
- `RuntimeLesson` / `LessonTrigger` / `LessonEvidence` / `LessonStats`：可注入的"以后该怎么办"
  （无存储 status——注入资格由 `lesson_score` 从 LessonStats 账本现算,见该模块）

设计取舍：
- `@dataclass` 非 frozen（项目约定，参考 `runtime/failure_memory::FailureEntry`）
- `created_at`/`updated_at` 用完整 ISO datetime 保留秒级精度
- 内容层 = 必填 `advice`（散文指令，= 旧 recommendation）+ 可选 `example`
  （结构化示范，= 旧 suggested_action，仅"形状类"修复才有）。`memory_text`
  （为已废弃 mem0 而生的死字段）已删。
- `source_type` ∈ {template, reflector, research, human}：来源一等枚举，
  从"靠 tags/前缀反推"提为显式声明。
- 不引入 dataclass-json / Pydantic 等依赖，序列化自己写
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


# LessonStatus 状态机已删：lesson 不再存储 status，注入资格由
# `lesson_score.compute_score`（helped/hurt 账本派生）现算,展示标签见
# `lesson_score.status_label`（active/dormant）。


# ============================================================
# Episode 侧：从 trace 抽出的"发生了什么"
# ============================================================


@dataclass
class FailureEvent:
    """单条失败事件。一个 episode 可能有 0..N 条。

    raw_action 取自 `core/trace_schema::ALL_ACTIONS`，常见值：
    - "tool_call_end"（output 含失败签名）
    - "failure_recovery_hint"（失败恢复 augment 触发）
    - "evaluator_call_end"（副 LLM 判失败）
    """

    iteration: int
    op: str  # "skill_exec:ai-digest/dup_check" / "load_skill" / "" (evaluator 无 tool)
    args_hash: str  # 12 char sha256 of canonical JSON args；evaluator 用 ""
    error_type: str  # "schema_mismatch" / "transient" / "unknown" / "soft_quality"
    error_message: str  # 截断到 300 char
    timestamp: str
    raw_action: str  # ALL_ACTIONS 之一
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FailureEvent":
        # 3d 改名迁移窗口:旧 episode payload 用 tool_key,读时映射到 op。
        if "tool_key" in data and "op" not in data:
            data = {**{k: v for k, v in data.items() if k != "tool_key"},
                    "op": data["tool_key"]}
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
            evaluator_verdicts=list(data.get("evaluator_verdicts", [])),
            raw_summary=dict(data.get("raw_summary", {})),
        )


# ============================================================
# Lesson 侧：可注入的"以后该怎么办"
# ============================================================
#
# 键字段表（键重构·刀3·3d 词汇收敛后。op × args_hash 是正交两轴,其余是派生物）:
#
# | 字段          | 回答什么问题       | 谁计算                        | 何时可观测           |
# |---------------|--------------------|-------------------------------|----------------------|
# | op            | 哪个操作（粗）     | BaseTool.op（工具自声明）     | 调用时（首次即有）   |
# | args_hash     | 哪组参数（指纹）   | BaseTool.args_hash / SSoT     | 调用时               |
# | call_key      | 哪次具体调用（细） | 派生 = (op, args_hash)        | 调用时（熔断/计数用）|
# | failure_class | 失败的粗类         | classify_tool_failure（SSoT） | 失败时（召回键）     |
# | failure_reason| 失败的细由（症状） | _failure_reason（确定性提取） | 失败后（冷路径精化） |
# | lesson_id     | 这条经验的身份     | 派生 = hash(op|class|reason)  | 入库时               |
#
# 认识论边界:failure_reason 对 schema/transient 等 oracle 背书的类是校验器给的真原因,
# 对 unknown 族只是症状签名——advice 不许据此做因果断言（因果诊断是 grounded Reflector 专属）。
# 召回键只用召回时刻可观测的字段（op + failure_class);call_key/lesson_id 是派生物、不当召回键。


@dataclass
class LessonTrigger:
    """Lesson 何时该被检索/注入的条件。

    `failure_class` 必填（键重构后单轨:工具失败 base-3 schema_mismatch/transient/
    unknown + 质量类 soft_quality_issue/semantic_failure）；
    `op` Optional——质量类不绑工具。
    `failure_count_gte` 默认 1（键重构删了会设 2 的 repeated 模板,现恒为 1）。
    `scope` 用 string 不用 enum，是方法论归属的正式键：
    "global" / "agent:<name>" / "skill:<name>"。
    `failure_reason`：确定性提取的"这次错在哪"（如 "missing:args"），是 lesson_id
    第三成分，现以可读字段存储（原只烘进 lesson_id hash、读不回、比不了、调不了
    粒度）；当前召回不读它做匹配，"按 failure_reason 匹配"留待后续。

    （`task_type` 已删：唯一写入点写 None、零读取；`condition_hint` 已删：
    100% 派生自 failure_class+op+failure_reason，改由 retriever 召回时现渲染。）
    """

    failure_class: str
    op: Optional[str] = None
    failure_count_gte: int = 1
    scope: str = "global"
    failure_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonTrigger":
        # 显式取已知字段，容忍并忽略旧 payload 残留的 task_type / condition_hint。
        # 3d 改名迁移窗口:旧 payload 用 error_class / cause_sig / tool_name,读时兜底回退
        # （3e 迁移脚本重写后即无旧键;这里保留回退保护未迁移的历史数据）。
        return cls(
            failure_class=data.get("failure_class", data.get("error_class", "")),
            op=data.get("op", data.get("tool_name")),
            failure_count_gte=data.get("failure_count_gte", 1),
            scope=data.get("scope", "global"),
            failure_reason=data.get("failure_reason", data.get("cause_sig", "")),
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
    """Lesson 触发后的 outcome 账本。OutcomeTracker 填。lesson_score 由此派生注入分。

    `ineffective_count` 已删（刀4 折叠）:INEFFECTIVE 与 HURT 同权记 hurt_count,
    不留独立计数。判别的"被应用×有效"测量走 trace 事件,不靠 lesson 行。
    """

    hit_count: int = 0
    helped_count: int = 0
    hurt_count: int = 0
    last_hit_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LessonStats":
        # 旧 payload 残留 ineffective_count 被忽略（显式取已知字段）。
        return cls(
            hit_count=data.get("hit_count", 0),
            helped_count=data.get("helped_count", 0),
            hurt_count=data.get("hurt_count", 0),
            last_hit_at=data.get("last_hit_at"),
        )


@dataclass
class RuntimeLesson:
    """一条 conditional lesson。从 episode + failure_class 用模板生成。

    `lesson_id` = sha1(failure_class + op + failure_reason)[:16] —— **原因层语义键**：
    同一种失败原因（不同 trace、不同参数，但同 failure_class + tool + 原因签名）
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
    # 来源一等枚举：template | reflector | research | human。从"靠 tags/前缀反推"
    # 提为显式声明；未来可据此配来源先验（零 schema 变更）。
    source_type: str = "template"
    stats: LessonStats = field(default_factory=LessonStats)
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
            "source_type": self.source_type,
            "stats": self.stats.to_dict(),
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
            source_type=data.get("source_type", "template"),
            stats=LessonStats.from_dict(data.get("stats", {})),
            example=data.get("example"),
        )
