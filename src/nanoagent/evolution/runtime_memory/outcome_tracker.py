"""OutcomeTracker — 关闭自进化飞轮 + tri-state outcome 判定。

输入：单 trace 的 events list（或 trace 文件路径）
处理：扫 ACTION_LESSON_USED → look ahead 同 tool 后续事件 →
      判 HELPED / HURT / INEFFECTIVE → 更新 backend stats + Beta confidence

入门视角：
- 飞轮闭环关键一环：lesson 用过之后没人更新它，confidence 永远是死的 0
- 在每个 turn 完整结束（含 evaluator retry）后被 Harness 触发
- 不主动状态机流转 candidate → promoted（那是 PromotionGate 的事）

tri-state 判定：
- 对每条 ACTION_LESSON_USED 事件 E（按 lesson_id 去重，取首次）：
  1. 优先扫同 tool 后续 ACTION_TOOL_CALL_REPAIR_REQUIRED → INEFFECTIVE
     (lesson 已注入但 LLM 仍生成被 schema 拦截的同类非法调用 ——
      "relevant but ineffective"，不进 helped/hurt 计数，不污染 confidence)
  2. 否则扫 ACTION_TOOL_CALL_END：output 含失败签名 → HURT；全成功 → HELPED

关键："后续"判定 = iteration **严格大于** lesson_used.iteration：
- main_loop 在 iter N 的 emit 顺序是 [repair_required, lesson_used, tool_call_end]，
  整 iter 的所有事件都是 lesson 召回的前因 —— RR 是触发拦截的那次，TCE_failure
  是 lesson_retriever 命中的那次失败本身。LLM 看到 lesson hint 要等 iter N+1 prompt 渲染。
- 不能用 iteration >=（包括同 iter）：会把 TCE_failure 误算 HURT
- 不能用 events list 索引（idx > LU idx）：main_loop 的 TCE emit 晚于 LU emit，同 iter 的
  TCE 仍 idx 较大，仍会被误判 HURT
- 只有"iteration 严格大于"两层语义都对：同 iter 全部排除（前因），下 iter 全部纳入（结果）

confidence 公式（仅 HELPED/HURT 影响）：
- 简单 Beta：`success / (success + failure + 1)`
- 加 1 是 Laplace smoothing，防止 success=0 / failure=0 极端值

INEFFECTIVE 副作用：
- ineffective_count++（独立计数）
- confidence 不变（success/failure 不动）
- 若 lesson.status == PROMOTED → 自动降级 CANDIDATE
  （PROBATION/RETIRED/CANDIDATE/EXPIRED 状态不动，只 stats 累加）
- 自动降级理由：lesson 已无能力把 LLM 拉回合法调用 —— 文本表达失效。
  未来版本可用阈值控制（如 ineffective_count >= 3 才降）；当前一击降级
  保守一些防 false positive。

幂等：
- _processed_paths set 防同一 trace 重复处理
- harness 在 evaluator retry 完整结束后触发——保证拿到最终状态

异常路径：
- lesson_id 已 deleted / 不存在 → log debug + skip，不阻塞
- backend 异常 → fail-open（log warning，trace 仍写完整）
- trace 文件读失败 → skip + log warning

下游：
- PromotionGate 用 confidence + hit_count 判 candidate → promoted
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set

from nanoagent.core import trace_schema as ts
from nanoagent.evolution.runtime_memory.backend import (
    LessonNotFound,
    MemoryBackend,
)
from nanoagent.evolution.runtime_memory.schema import LessonStats, LessonStatus
from nanoagent.runtime.failure_memory import _is_tool_failure

_logger = logging.getLogger("nanoagent.outcome_tracker")


class TraceOutcome(str, Enum):
    """Lesson 在单条 trace 里的结局。"""
    HELPED = "helped"
    HURT = "hurt"
    INEFFECTIVE = "ineffective"


@dataclass(frozen=True)
class OutcomeUpdate:
    """单次 lesson outcome 更新结果——caller 用于日志 / 单测断言。

    `helped` 字段保留向后兼容：True iff outcome == HELPED；
    HURT 和 INEFFECTIVE 都返回 False。新代码用 `outcome` 字段做精确分支。
    """
    lesson_id: str
    outcome: TraceOutcome
    new_confidence: float
    new_hit_count: int

    @property
    def helped(self) -> bool:
        return self.outcome == TraceOutcome.HELPED


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _compute_confidence(success: int, failure: int) -> float:
    """Beta + Laplace smoothing：success / (success + failure + 1)。

    样本极小时（s=0, f=0）→ 0/1 = 0；
    s=1, f=0 → 1/2 = 0.5；
    s=10, f=0 → 10/11 ≈ 0.91；
    s=10, f=10 → 10/21 ≈ 0.48；
    分母 +1 是平滑项，避免极端 0/0 或 1.0 触发 PromotionGate 误升级。
    """
    return success / max(1, success + failure + 1)


class OutcomeTracker:
    """trace 写完后扫一遍 ACTION_LESSON_USED → 判 outcome → 更新 backend。

    生命周期：main.py 装配一份共享实例，注入 Harness。Harness 在每个
    user-turn 完整结束（含 evaluator retry）后调 process_trace_path()。

    线程安全：内部 _processed_paths 是 set，单进程读写不需锁。多 Harness
    共享同一实例时也只是写入唯一 trace path（每 turn 唯一），无冲突。
    """

    def __init__(self, backend: MemoryBackend):
        self._backend = backend
        self._processed_paths: Set[str] = set()

    # ============================================================
    # IO 入口
    # ============================================================

    def process_trace_path(self, path: Path) -> List[OutcomeUpdate]:
        """读 trace JSONL，扫 ACTION_LESSON_USED，更新 backend。

        幂等：同一 trace_path 不会处理两次。
        """
        path_str = str(path)
        if path_str in self._processed_paths:
            return []
        self._processed_paths.add(path_str)

        events = self._read_trace_events(path)
        if events is None:
            return []
        return self.process_trace_events(events)

    @staticmethod
    def _read_trace_events(path: Path) -> Optional[List[dict]]:
        """读 trace JSONL → 跳过 header / summary / 损坏行，返回 step events list。"""
        try:
            with open(path, encoding="utf-8") as f:
                events = []
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        e = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # 损坏行容忍
                    # 只关心带 action 字段的 step（跳过 run_header / run_summary）
                    if "action" in e:
                        events.append(e)
                return events
        except OSError as err:
            _logger.warning(f"OutcomeTracker 读 trace 失败 {path}: {err}")
            return None

    # ============================================================
    # 纯函数核心
    # ============================================================

    def process_trace_events(self, events: List[dict]) -> List[OutcomeUpdate]:
        """扫 events，按 lesson_id 去重，计算 helped/hurt，更新 backend。

        纯函数语义：除调 backend.update_lesson_metadata 外无 IO；输入 events
        重复传入也不影响结果（process_trace_path 通过 _processed_paths 保幂等
        而非这里的内部状态）。
        """
        # 1. 收集所有 lesson_used 事件，按 lesson_id 去重（首次出现）
        lesson_first_use: dict[str, dict] = {}
        for e in events:
            if e.get("action") != ts.ACTION_LESSON_USED:
                continue
            lid = e.get("lesson_id")
            if not lid or lid in lesson_first_use:
                continue
            lesson_first_use[lid] = e

        if not lesson_first_use:
            return []

        # 2. 对每条 lesson_used 判 tri-state outcome。用 iteration 严格大于：
        #    LLM 看到 lesson hint 要到下个 iter prompt 渲染才生效，所以同 iter
        #    所有事件（含 RR / TCE_failure）都是 lesson 召回的前因不算结果。
        updates: List[OutcomeUpdate] = []
        for lesson_id, lu_event in lesson_first_use.items():
            tool = lu_event.get("tool", "")
            iteration = lu_event.get("iteration", 0)
            outcome = self._judge_outcome(events, tool, iteration)
            update = self._update_backend(lesson_id, outcome)
            if update is not None:
                updates.append(update)
        return updates

    @staticmethod
    def _judge_outcome(
        events: List[dict], tool: str, lesson_used_iter: int
    ) -> TraceOutcome:
        """根据 iteration 严格大于 lesson_used_iter 的事件判定三态。

        判定优先级：
        1. 同 tool 的 ACTION_TOOL_CALL_REPAIR_REQUIRED → INEFFECTIVE
           (lesson 已注入但 LLM 仍触发 schema 拦截，文本建议未起作用)
        2. 否则扫 ACTION_TOOL_CALL_END：任一失败 → HURT
        3. 全成功 / 无后续 same-tool call → HELPED

        关键：iteration 严格大于（同 iter 全部排除）。
        - 同 iter 的 RR：是触发 lesson 召回的那次，LLM 还没看到 lesson hint
        - 同 iter 的 TCE_failure：是 lesson_retriever 命中的那次失败本身
        - LLM 真正"看到" lesson 是在下 iter prompt 渲染时
        """
        # 1. 优先 INEFFECTIVE
        for e in events:
            if e.get("action") != ts.ACTION_TOOL_CALL_REPAIR_REQUIRED:
                continue
            if e.get("tool") != tool:
                continue
            if e.get("iteration", 0) <= lesson_used_iter:
                continue
            return TraceOutcome.INEFFECTIVE

        # 2. helped/hurt
        for e in events:
            if e.get("action") != ts.ACTION_TOOL_CALL_END:
                continue
            if e.get("tool") != tool:
                continue
            if e.get("iteration", 0) <= lesson_used_iter:
                continue
            output = e.get("output", "")
            if isinstance(output, str) and _is_tool_failure(output):
                return TraceOutcome.HURT
        return TraceOutcome.HELPED

    def _update_backend(
        self, lesson_id: str, outcome: TraceOutcome
    ) -> Optional[OutcomeUpdate]:
        """根据 outcome 三态更新 backend stats。fail-open。

        HELPED / HURT：success 或 failure 计数 + Beta confidence 重算
        INEFFECTIVE：ineffective_count++，confidence 不变；PROMOTED 自动降级 CANDIDATE
        """
        try:
            old = self._backend.get_lesson(lesson_id)
        except Exception as e:
            _logger.warning(f"OutcomeTracker get_lesson 异常 {lesson_id}: {e}")
            return None
        if old is None:
            _logger.debug(f"OutcomeTracker skip：lesson {lesson_id} 不存在（已 deleted？）")
            return None

        if outcome == TraceOutcome.INEFFECTIVE:
            new_stats = LessonStats(
                hit_count=old.stats.hit_count + 1,
                success_after_hit=old.stats.success_after_hit,
                failure_after_hit=old.stats.failure_after_hit,
                ineffective_count=old.stats.ineffective_count + 1,
                last_hit_at=_now_iso(),
            )
            new_conf = old.confidence  # 不动
            update_kwargs: dict = {"stats": new_stats, "confidence": new_conf}
            # PROMOTED → CANDIDATE 自动降级；其他状态保持
            if old.status == LessonStatus.PROMOTED:
                update_kwargs["status"] = LessonStatus.CANDIDATE
                _logger.info(
                    f"OutcomeTracker INEFFECTIVE：lesson {lesson_id} "
                    f"PROMOTED → CANDIDATE（ineffective_count={new_stats.ineffective_count}）"
                )
        else:
            helped = outcome == TraceOutcome.HELPED
            new_stats = LessonStats(
                hit_count=old.stats.hit_count + 1,
                success_after_hit=old.stats.success_after_hit + (1 if helped else 0),
                failure_after_hit=old.stats.failure_after_hit + (0 if helped else 1),
                ineffective_count=old.stats.ineffective_count,  # 保持
                last_hit_at=_now_iso(),
            )
            new_conf = _compute_confidence(
                new_stats.success_after_hit, new_stats.failure_after_hit
            )
            update_kwargs = {"stats": new_stats, "confidence": new_conf}

        try:
            self._backend.update_lesson_metadata(lesson_id, **update_kwargs)
        except LessonNotFound:
            _logger.debug(f"OutcomeTracker update skip：lesson {lesson_id} 在 update 时已 deleted")
            return None
        except Exception as e:
            _logger.warning(f"OutcomeTracker update_lesson_metadata 异常 {lesson_id}: {e}")
            return None
        return OutcomeUpdate(
            lesson_id=lesson_id,
            outcome=outcome,
            new_confidence=new_conf,
            new_hit_count=new_stats.hit_count,
        )
