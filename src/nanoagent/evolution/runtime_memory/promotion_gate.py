"""PromotionGate —— candidate ↔ promoted ↔ retired 自动化规则引擎。

在状态机校验（LEGAL_TRANSITIONS）+ 手动 CLI（promote/retire/...）基础上加自动
决策器：每个 user-turn 完整结束（含 evaluator retry + Ingestor + OutcomeTracker）
后由 Harness 调一次 sweep()，用 stats / evidence_count 决定是否升降级。

为什么这是飞轮的最后一环：
- LessonRetriever 严格只召回 PROMOTED，但没人把 candidate 升 promoted
- 6 candidate / 1 retired / 0 promoted 的状态下，retriever 永远空召回 →
  OutcomeTracker 永远空跑 → 飞轮转不起来
- 即使 LessonIngestor 持续累积 evidence（同失败模式被多 trace 见过），也只是
  在 candidate 上累加 source_episode_ids，没有任何机制把它升 PROMOTED

为什么用 sweep 而不是 LessonIngestor / OutcomeTracker 内联触发：
- 解耦：Ingestor 关心"trace → candidate"，OutcomeTracker 关心"trace → stats"，
  PromotionGate 关心"stats → status 转移"，三者职责互不重叠
- 一致性：sweep 在所有数据写入完成后跑一次，避免 Ingestor 写完之后 OutcomeTracker
  还没更新就被 PromotionGate 看到中间态
- 性能：每个 user-turn 一次 sweep，扫候选 + promoted 子集（搜 lesson 已有 status
  filter）。生产 backend 现状 7 条 lesson，sweep 成本忽略

为什么当前阈值是保守拍脑袋值：
- 严格按"等真实数据校准"，飞轮永远转不起来 —— chicken-and-egg
- INEFFECTIVE 自动降级把"promote 错了"的代价压得很低（1 次 ineffective 就
  PROMOTED → CANDIDATE）。所以保守阈值 + 自我修正机制是可接受的开局
- 阈值默认 3/3/3，全部放 main.py 顶部常量，等真实 stats 沉淀后调

留待后续扩展（暂不做）：
- Bayesian Beta confidence + Wilson smoothing
- Cooldown / 重新 promote 的最小间隔
- 后台 cron sweep（不依赖 user-turn）
- LessonGenerator structured repair_template
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

from nanoagent.evolution.runtime_memory.backend import (
    IllegalStatusTransition,
    LessonNotFound,
    MemoryBackend,
)
from nanoagent.evolution.runtime_memory.schema import LessonStatus, RuntimeLesson

_logger = logging.getLogger("nanoagent.promotion_gate")

# env override 让 ops 改阈值不需改代码。module-level 一次性读，进程启动后
# 不支持热重载——production 改值需重启，与 main.py 顶部常量同等约束。
# 命名前缀 NANOAGENT_PROMOTION_* 与 main.py 的 PROMOTION_* 常量一一对应。
_DEFAULT_EVIDENCE_MIN = int(os.getenv("NANOAGENT_PROMOTION_EVIDENCE_MIN", "3"))
_DEFAULT_HELPED_MIN = int(os.getenv("NANOAGENT_PROMOTION_HELPED_MIN", "1"))
_DEFAULT_RETIRE_HURT_MIN = int(os.getenv("NANOAGENT_PROMOTION_RETIRE_HURT_MIN", "3"))
_DEFAULT_RETIRE_INEFFECTIVE_MIN = int(
    os.getenv("NANOAGENT_PROMOTION_RETIRE_INEFFECTIVE_MIN", "3")
)


@dataclass(frozen=True)
class PromotionDecision:
    """单条 lesson 的状态转移决策。reason 用枚举字符串便于 trace / log filter。

    `evidence_episode_ids`：命中 evidence_threshold 时记录支撑证据的
    episode IDs（截断 ≤10），让 ops 用 `nanoagent.lesson list --json` 校准阈值时
    能反推到具体 trace。frozen=True，需要补字段时用 dataclasses.replace。
    """
    lesson_id: str
    from_status: LessonStatus
    to_status: LessonStatus
    reason: str  # "evidence_threshold" / "hurt_threshold" / "ineffective_threshold"
    evidence_episode_ids: Tuple[str, ...] = field(default_factory=tuple)


# audit_callback 回调签名。caller 决定怎么记 PromotionDecision——可写 trace
# JSONL（如果 RunTracer 还活着）/audit log/stdout 都行。PromotionGate 自身不
# 依赖具体 tracer 类型，避免反向 import 与生命周期耦合。
AuditCallback = Callable[[PromotionDecision], None]


class PromotionGate:
    """Lesson 状态自动转移规则引擎。

    生命周期：main.py 装配一份共享实例，注入 Harness。Harness 在每个 user-turn
    完整结束（含 evaluator retry + Ingestor + OutcomeTracker）后调 sweep()。

    线程安全：内部无可变状态，全部走 backend，单进程读写不需锁。

    阈值默认值 fallback 顺序：
    1. 调用方显式传入 kwargs（最高优先级，main.py 走这条）
    2. env 变量 NANOAGENT_PROMOTION_* （ops 改值不需改代码）
    3. 硬编码 3/1/3/3（最后保底）
    """

    def __init__(
        self,
        backend: MemoryBackend,
        *,
        promote_evidence_min: int = _DEFAULT_EVIDENCE_MIN,
        promote_helped_min: int = _DEFAULT_HELPED_MIN,
        retire_hurt_min: int = _DEFAULT_RETIRE_HURT_MIN,
        retire_ineffective_min: int = _DEFAULT_RETIRE_INEFFECTIVE_MIN,
    ):
        self._backend = backend
        self._promote_evidence_min = promote_evidence_min
        self._promote_helped_min = promote_helped_min
        self._retire_hurt_min = retire_hurt_min
        self._retire_ineffective_min = retire_ineffective_min

    # ============================================================
    # 纯函数核心
    # ============================================================

    def evaluate(self, lesson: RuntimeLesson) -> Optional[PromotionDecision]:
        """给定 lesson 返回应该转移到的目标 status，无副作用。

        CANDIDATE 不直接升 PROMOTED，先去 PROBATION 试用层。
        PROBATION 期间被 LessonRetriever 召回（与 PROMOTED
        并列查询）→ OutcomeTracker 收集真实 helped/hurt → 决定升 PROMOTED 或
        回 CANDIDATE。只有"确认有效"才进 PROMOTED。

        规则（按优先级，命中即 break）：
        - candidate + hurt 达阈值 → RETIRED（高优先级 toxic 信号）
        - candidate + ineff 达阈值 → RETIRED
        - candidate + evidence 达阈值 → **PROBATION**（不是 PROMOTED）
        - probation + helped 达阈值 → PROMOTED（真用过且有效）
        - probation + ineff>=1 → CANDIDATE（试用失败，回到候选）
        - probation + hurt>=1 → CANDIDATE
        - promoted + hurt 达阈值 → RETIRED
        - 其他（retired / expired）→ 不动
        """
        status = lesson.status
        stats = lesson.stats
        evidence_count = len(lesson.evidence.source_episode_ids)

        if status == LessonStatus.CANDIDATE:
            if stats.failure_after_hit >= self._retire_hurt_min:
                return PromotionDecision(
                    lesson_id=lesson.lesson_id,
                    from_status=status,
                    to_status=LessonStatus.RETIRED,
                    reason="hurt_threshold",
                )
            if stats.ineffective_count >= self._retire_ineffective_min:
                return PromotionDecision(
                    lesson_id=lesson.lesson_id,
                    from_status=status,
                    to_status=LessonStatus.RETIRED,
                    reason="ineffective_threshold",
                )
            if evidence_count >= self._promote_evidence_min:
                return PromotionDecision(
                    lesson_id=lesson.lesson_id,
                    from_status=status,
                    to_status=LessonStatus.PROBATION,
                    reason="evidence_threshold",
                    # 填证据链。截断 ≤10 防 trace / audit log 膨胀
                    evidence_episode_ids=tuple(
                        lesson.evidence.source_episode_ids[:10]
                    ),
                )
            return None

        if status == LessonStatus.PROBATION:
            # 试用期负面信号优先 —— 一次 ineff 或 hurt 就回 candidate 重新积累
            if stats.ineffective_count >= 1:
                return PromotionDecision(
                    lesson_id=lesson.lesson_id,
                    from_status=status,
                    to_status=LessonStatus.CANDIDATE,
                    reason="probation_ineffective",
                )
            if stats.failure_after_hit >= 1:
                return PromotionDecision(
                    lesson_id=lesson.lesson_id,
                    from_status=status,
                    to_status=LessonStatus.CANDIDATE,
                    reason="probation_hurt",
                )
            if stats.success_after_hit >= self._promote_helped_min:
                return PromotionDecision(
                    lesson_id=lesson.lesson_id,
                    from_status=status,
                    to_status=LessonStatus.PROMOTED,
                    reason="helped_threshold",
                )
            return None

        if status == LessonStatus.PROMOTED:
            if stats.failure_after_hit >= self._retire_hurt_min:
                return PromotionDecision(
                    lesson_id=lesson.lesson_id,
                    from_status=status,
                    to_status=LessonStatus.RETIRED,
                    reason="hurt_threshold",
                )
            return None

        return None  # retired / expired：不动

    # ============================================================
    # IO 副作用
    # ============================================================

    def apply(self, decision: PromotionDecision) -> bool:
        """把 decision 应用到 backend。fail-open：异常 log warning 返回 False。

        正常路径：调 update_lesson_metadata（走 LEGAL_TRANSITIONS 校验）。
        IllegalStatusTransition / LessonNotFound 不抛 —— 这些表示 backend 状态在
        evaluate 之后被 race / CLI 等改了，不是程序 bug。
        """
        try:
            self._backend.update_lesson_metadata(
                decision.lesson_id, status=decision.to_status
            )
            _logger.info(
                f"[PromotionGate] {decision.lesson_id}: "
                f"{decision.from_status.value} → {decision.to_status.value} "
                f"(reason={decision.reason})"
            )
            return True
        except (IllegalStatusTransition, LessonNotFound) as e:
            _logger.warning(
                f"[PromotionGate] apply skip {decision.lesson_id} "
                f"({decision.from_status.value} → {decision.to_status.value}): {e}"
            )
            return False
        except Exception as e:
            _logger.warning(
                f"[PromotionGate] apply 异常 {decision.lesson_id}: {e}"
            )
            return False

    def sweep(
        self,
        audit_callback: Optional[AuditCallback] = None,
    ) -> List[PromotionDecision]:
        """扫 backend 全部 candidate + promoted lesson，evaluate + apply 每条。

        返回 applied decisions 列表（caller 用于日志 / trace / 单测断言）。

        只扫 candidate / promoted —— retired / expired 终态不动。用 backend 现成
        search_lessons + status filter，无额外索引开销。

        `audit_callback`：apply 成功后 fail-open 调一次。caller 决定怎么记
        （写 trace JSONL / audit file / external monitor）。callback 抛出异常被捕获
        log warning 不阻断 sweep，因为审计失败不应让飞轮停转。
        """
        applied: List[PromotionDecision] = []

        # 分三次查避免 OR 嵌套：candidate + probation + promoted 都参与规则评估
        # （retired / expired 终态跳过）
        targets: list[RuntimeLesson] = []
        try:
            for status in (
                LessonStatus.CANDIDATE,
                LessonStatus.PROBATION,
                LessonStatus.PROMOTED,
            ):
                targets.extend(
                    self._backend.search_lessons(
                        filters={"status": status.value}, limit=1000
                    )
                )
        except Exception as e:
            _logger.warning(f"[PromotionGate] sweep search 失败（已 fail-open）: {e}")
            return applied

        for lesson in targets:
            decision = self.evaluate(lesson)
            if decision is None:
                continue
            if not self.apply(decision):
                continue
            applied.append(decision)
            if audit_callback is not None:
                try:
                    audit_callback(decision)
                except Exception as e:
                    _logger.warning(
                        f"[PromotionGate] audit_callback 异常（已 fail-open）: {e}"
                    )
        return applied
