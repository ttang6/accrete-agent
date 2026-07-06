"""ReflectorMintRunner — 离线飞轮 mint（批/调度，**非 inline 热路径**）。

把 trace 里"非 schema、tool-grounded、且有可观测恢复"的失败，经 Reflector 产出
grounded `suggested_action`，mint 成 RuntimeLesson 入 backend——出生即 score=T 可被
LessonRetriever 召回，之后凭 helped/hurt 账本涨落。与 inline 模板 LessonIngestor 共用
同一 lesson_id 语义键（跨 trace / 跨 mint 路径自然去重 + evidence 累加）。

两半分流（对齐 docs/_eval_flywheel_hole_workflow_findings.md §7）：
- **grounded 那半**（有"在轨恢复验证"——同 op 失败后重执行成功）：mint 进 backend。
- **信号缺失那半**（无观测恢复）：仅当 `emit_soft=True`（默认关）才产**软** suggest →
  写 `SoftSuggestSink` 物理旁路 jsonl，**不进 backend、不被召回、钉 CANDIDATE**。
  consumption 闸是硬的（旁路 + flag + retriever 永不读此 jsonl），不靠 reflector 弃权。

门控（只 mint 重执行身份明确的 op）：
- 跳过 schema_mismatch（Tier-1 确定性 RepairGate 管）、soft_quality（不绑 tool）。
- 只收 `skill_exec:<skill>/<script>` 前缀的 op（episode op 已是脚本级、身份明确）。
  **排除 fetch**（多源 digest loop ~50% 归因混淆）**与 arxiv**（episode op 为裸
  `arxiv`、action 级身份未细分——留待后续给 episode_extractor 加细粒度键再开）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from nanoagent.core.logger import get_logger
from nanoagent.evolution.runtime_memory.backend import (
    LessonAlreadyExists,
    MemoryBackend,
)
from nanoagent.evolution.runtime_memory.episode_extractor import EpisodeExtractor
from nanoagent.evolution.runtime_memory.lesson_generator import (
    _failure_reason,
    _deterministic_lesson_id,
    _now_iso,
)
from nanoagent.evolution.runtime_memory.reflector import (
    Reflector,
    find_observed_recovery,
)
from nanoagent.evolution.runtime_memory.schema import (
    FailureEvent,
    LessonEvidence,
    LessonStats,
    LessonTrigger,
    RuntimeEpisode,
    RuntimeLesson,
)
from nanoagent.evolution.runtime_memory.trace_error_classifier import (
    TraceErrorClassifier,
)

_logger = get_logger("offline_mint")

# FailureEvent.error_type：这些不归 Reflector 的离线 mint 管
# （schema 走 Tier-1；soft 不绑 tool，op 也为空）
_SKIP_ERROR_TYPES = {"schema_mismatch", "soft_quality"}
_DEFAULT_ALLOWED_PREFIXES = ("skill_exec:",)


@dataclass
class MintReport:
    """单 trace / 批跑的 mint 统计——日志 + 单测断言用。"""
    grounded_minted: int = 0     # 新建 grounded CANDIDATE lesson
    grounded_extended: int = 0   # 已存在 → extend_lesson_evidence
    soft_emitted: int = 0        # 写入旁路 jsonl 的软 suggest
    skipped: int = 0             # 门控跳过（非范围 op / 非 tool 失败）
    abstained: int = 0           # 无恢复且未开 soft / reflect 失败 → 不产

    def merge(self, other: "MintReport") -> None:
        self.grounded_minted += other.grounded_minted
        self.grounded_extended += other.grounded_extended
        self.soft_emitted += other.soft_emitted
        self.skipped += other.skipped
        self.abstained += other.abstained


class SoftSuggestSink:
    """信号缺失那半的**物理旁路**：append 到独立 jsonl，LessonRetriever 永不读。"""

    def __init__(self, path: Path):
        self._path = Path(path)

    def write(self, record: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ReflectorMintRunner:
    """离线遍历 trace → 产 grounded CANDIDATE lesson（+ 可选软旁路）。"""

    def __init__(
        self,
        backend: MemoryBackend,
        reflector: Reflector,
        *,
        allowed_prefixes: Iterable[str] = _DEFAULT_ALLOWED_PREFIXES,
        soft_sink: Optional[SoftSuggestSink] = None,
        emit_soft: bool = False,
    ):
        self._backend = backend
        self._reflector = reflector
        self._allowed_prefixes = tuple(allowed_prefixes)
        self._soft_sink = soft_sink
        self._emit_soft = emit_soft
        self._extractor = EpisodeExtractor()
        self._classifier = TraceErrorClassifier()

    # ---- 批跑入口 ----

    def process_trace_paths(self, paths: Iterable[Path]) -> MintReport:
        total = MintReport()
        for p in paths:
            try:
                total.merge(self.process_trace_path(Path(p)))
            except Exception as e:  # 单条坏 trace 不阻断整批
                _logger.warning(f"[offline_mint] 处理 {p} 异常（跳过）: {e}")
        return total

    def process_trace_path(self, path: Path) -> MintReport:
        path = Path(path)
        if not path.exists():
            return MintReport()
        episode = self._extractor.extract_from_file(path)
        if episode is None:
            return MintReport()
        events = EpisodeExtractor._read_jsonl(path)
        return self.process_episode(episode, events)

    def process_episode(
        self, episode: RuntimeEpisode, events: List[Dict[str, Any]]
    ) -> MintReport:
        report = MintReport()
        try:
            self._backend.add_episode(episode)  # INSERT OR REPLACE，幂等
        except Exception as e:
            _logger.warning(f"[offline_mint] add_episode 失败 {episode.episode_id}: {e}")

        for fe in episode.failures:
            if not fe.op or fe.error_type in _SKIP_ERROR_TYPES:
                report.skipped += 1
                continue
            if not any(fe.op.startswith(p) for p in self._allowed_prefixes):
                report.skipped += 1  # 排除 fetch / arxiv / 多源 loop
                continue
            recovery = find_observed_recovery(events, fe)
            if recovery is not None:
                self._mint_grounded(episode, fe, recovery, report)
            elif self._emit_soft and self._soft_sink is not None:
                self._emit_soft_suggest(episode, fe, report)
            else:
                report.abstained += 1  # 无 grounding + soft 关 → 诚实弃权
        return report

    # ---- grounded 半：mint 进 backend ----

    def _mint_grounded(
        self, episode: RuntimeEpisode, fe: FailureEvent,
        recovery: Dict[str, Any], report: MintReport,
    ) -> None:
        result = self._reflector.reflect(fe, recovery)
        if result is None or not result.suggested_action:
            report.abstained += 1
            return
        failure_class = self._classifier.classify(fe, episode.failures)
        lesson = self._build_lesson(episode, fe, failure_class, result)
        try:
            self._backend.add_lesson(lesson)
            report.grounded_minted += 1
        except LessonAlreadyExists:
            try:
                self._backend.extend_lesson_evidence(
                    lesson.lesson_id,
                    episode_id=episode.episode_id,
                    sample_trace_path=lesson.evidence.sample_trace_path,
                    sample_failure_iteration=lesson.evidence.sample_failure_iteration,
                    sample_error_message=lesson.evidence.sample_error_message,
                    example=lesson.example,
                )
                report.grounded_extended += 1
            except Exception as e:
                _logger.warning(f"[offline_mint] extend 失败 {lesson.lesson_id}: {e}")
        except Exception as e:
            _logger.warning(f"[offline_mint] add_lesson 失败 {lesson.lesson_id}: {e}")

    def _build_lesson(
        self, episode: RuntimeEpisode, fe: FailureEvent,
        failure_class: str, result,
    ) -> RuntimeLesson:
        failure_reason = _failure_reason(fe, failure_class)
        lesson_id = _deterministic_lesson_id(
            failure_class=failure_class, op=fe.op, failure_reason=failure_reason,
        )
        now = _now_iso()
        trigger = LessonTrigger(
            failure_class=failure_class, op=fe.op,
            failure_count_gte=1, scope=f"agent:{episode.agent_name}",
            failure_reason=failure_reason,
        )
        evidence = LessonEvidence(
            source_episode_ids=[episode.episode_id],
            sample_trace_path=episode.trace_path,
            sample_failure_iteration=fe.iteration,
            sample_args_hash=fe.args_hash,
            sample_error_message=fe.error_message[:300],
        )
        # 救 diagnosis 进 advice（原寄生在 memory_text 的 [reflector] 前缀里，是
        # recommendation 没有的真信息）；模板套话丢弃。来源标 source_type=reflector。
        advice = result.recommendation
        if result.diagnosis:
            advice = f"{advice}（根因：{result.diagnosis}）"
        return RuntimeLesson(
            lesson_id=lesson_id,
            advice=advice,
            trigger=trigger, evidence=evidence,
            source_type="reflector",
            created_at=now, updated_at=now,
            stats=LessonStats(),
            example=result.suggested_action,
        )

    # ---- 信号缺失半：软 suggest → 物理旁路 jsonl ----

    def _emit_soft_suggest(
        self, episode: RuntimeEpisode, fe: FailureEvent, report: MintReport
    ) -> None:
        result = self._reflector.reflect_soft(fe)
        if result is None:
            report.abstained += 1
            return
        self._soft_sink.write({
            "grounded": False,  # 软 suggest 物理旁路,永不进 backend / 召回
            "episode_id": episode.episode_id,
            "trace_path": episode.trace_path,
            "iteration": fe.iteration,
            "op": fe.op,
            "error_type": fe.error_type,
            "error_message": fe.error_message[:300],
            "diagnosis": result.diagnosis,
            "recommendation": result.recommendation,
            "minted_at": _now_iso(),
        })
        report.soft_emitted += 1
