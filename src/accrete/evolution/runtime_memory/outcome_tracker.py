"""OutcomeTracker — 关闭自进化飞轮 + tri-state outcome 判定。

输入：单 trace 的 events list（或 trace 文件路径）
处理：扫 ACTION_LESSON_USED → look ahead 同 tool 后续事件 →
      判 HELPED / HURT / INEFFECTIVE → 更新 backend stats（helped/hurt/ineffective 计数）

入门视角：
- 飞轮闭环关键一环：lesson 用过之后没人更新它，helped/hurt 计数永远是死的 0
- 在每个 turn 完整结束（含 evaluator retry）后被 Harness 触发
- 只写 helped/hurt 账本;注入资格由 lesson_score 从账本派生,无状态机流转

tri-state 判定：
- 对每条 ACTION_LESSON_USED 事件 E（按 lesson_id 去重，取首次）：
  1. 优先扫同 tool 后续 ACTION_TOOL_CALL_REPAIR_REQUIRED → INEFFECTIVE
     (lesson 已注入但 LLM 仍生成被 schema 拦截的同类非法调用 ——
      "relevant but ineffective"，不进 helped/hurt 计数)
  2. 否则扫 ACTION_TOOL_CALL_END：output 含失败签名 → HURT；全成功 → HELPED

关键："后续"判定 = iteration **严格大于** lesson_used.iteration：
- main_loop 在 iter N 的 emit 顺序是 [repair_required, lesson_used, tool_call_end]，
  整 iter 的所有事件都是 lesson 召回的前因 —— RR 是触发拦截的那次，TCE_failure
  是 lesson_retriever 命中的那次失败本身。LLM 看到 lesson hint 要等 iter N+1 prompt 渲染。
- 不能用 iteration >=（包括同 iter）：会把 TCE_failure 误算 HURT
- 不能用 events list 索引（idx > LU idx）：main_loop 的 TCE emit 晚于 LU emit，同 iter 的
  TCE 仍 idx 较大，仍会被误判 HURT
- 只有"iteration 严格大于"两层语义都对：同 iter 全部排除（前因），下 iter 全部纳入（结果）

账本更新（刀4 折叠）：
- 判别仍产三态（INEFFECTIVE 从 REPAIR_REQUIRED trace 事件判出 —— 这是"被应用×有效"
  的测量,保留;走 trace 不走 lesson 行）
- 但**账本折叠**:HELPED → helped_count++;HURT / INEFFECTIVE 同权 → hurt_count++
  （两者都是"注入后没起作用",分数权重 −1）。不留独立 ineffective 计数、不碰 status
- 注入资格由 `lesson_score.compute_score` 从账本派生（score≥T），OutcomeTracker 不判晋降

幂等：
- _processed_paths set 防同一 trace 重复处理
- harness 在 evaluator retry 完整结束后触发——保证拿到最终状态

异常路径：
- lesson_id 已 deleted / 不存在 → log debug + skip，不阻塞
- backend 异常 → fail-open（log warning，trace 仍写完整）
- trace 文件读失败 → skip + log warning

下游：
- lesson_score.compute_score 用 helped/hurt 账本派生注入分（score≥T 才召回）
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional, Set

from accrete.core import trace_schema as ts
from accrete.evolution.runtime_memory.backend import (
    LessonNotFound,
    MemoryBackend,
)
from accrete.evolution.runtime_memory.schema import LessonStats
from accrete.runtime.failure_memory import _is_tool_failure

_logger = logging.getLogger("accrete.outcome_tracker")


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
    new_hit_count: int

    @property
    def helped(self) -> bool:
        return self.outcome == TraceOutcome.HELPED


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


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
        """根据 outcome 更新 backend 账本。fail-open。

        刀4 折叠:HELPED → helped_count++;HURT / **INEFFECTIVE 同权** → hurt_count++
        （两者语义都是"注入后没起作用",分数权重都 −1,不留独立计数、不改状态）。
        注入资格由 lesson_score 从账本派生,OutcomeTracker 不再碰 status。
        判别的"被应用×有效"测量走 trace 事件（REPAIR_REQUIRED 本就在 trace）,不靠
        lesson 行区分——账本只服务注入决策。
        """
        try:
            old = self._backend.get_lesson(lesson_id)
        except Exception as e:
            _logger.warning(f"OutcomeTracker get_lesson 异常 {lesson_id}: {e}")
            return None
        if old is None:
            _logger.debug(f"OutcomeTracker skip：lesson {lesson_id} 不存在（已 deleted？）")
            return None

        helped = outcome == TraceOutcome.HELPED
        new_stats = LessonStats(
            hit_count=old.stats.hit_count + 1,
            helped_count=old.stats.helped_count + (1 if helped else 0),
            hurt_count=old.stats.hurt_count + (0 if helped else 1),
            last_hit_at=_now_iso(),
        )
        update_kwargs = {"stats": new_stats}

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
            new_hit_count=new_stats.hit_count,
        )
