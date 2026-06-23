"""LessonIngestor — trace 写完后自动抽取 episode + 生成 candidate lessons。

职责：
- 读 trace（JSONL 文件 or events list）
- 调 EpisodeExtractor 把含 failure 的 trace 转成 RuntimeEpisode
- 调 TraceErrorClassifier 把每个 failure 分类
- 调 LessonGenerator 产 candidate lesson
- upsert 进 backend：先 add_lesson，已存在则 extend_lesson_evidence 累加 evidence

设计要点：
- 封装 4 个组件（extractor / classifier / generator / backend），Harness 只接一个 ingestor
- 幂等：_processed_paths set 防同一 trace 重复 ingest（同 OutcomeTracker 模式）
- Fail-open：任何异常路径返回 IngestResult.error，不阻塞 user-turn 主流程
- 不主动状态机流转——所有新生成的 lesson 都是 CANDIDATE，等 PromotionGate 升级
- 不消费 ACTION_LESSON_USED——那是 OutcomeTracker 的职责，两者读 disjoint 维度

为什么独立成一个类而不是 Harness 直接调 4 个组件：
- 4 组件依赖给 Harness 注入太长（违反 Harness "channel-agnostic" 简洁性）
- 单元测试更简单（只 mock LessonIngestor 而非 4 个）
- main.py 装配点更对称（OutcomeTracker / LessonRetriever / LessonIngestor 共享 backend）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

from nanoagent.evolution.runtime_memory.backend import (
    LessonAlreadyExists,
    MemoryBackend,
)
from nanoagent.evolution.runtime_memory.episode_extractor import EpisodeExtractor
from nanoagent.evolution.runtime_memory.lesson_generator import (
    LessonGenerator,
    _cause_signature,
)
from nanoagent.evolution.runtime_memory.trace_error_classifier import (
    TraceErrorClassifier,
)

_logger = logging.getLogger("nanoagent.lesson_ingestor")


@dataclass(frozen=True)
class IngestResult:
    """单 trace ingest 的结果——caller 用于日志 / 单测断言。"""
    episode_id: Optional[str] = None
    lessons_added: int = 0  # 新创建（add_lesson 成功）
    lessons_extended: int = 0  # 已存在 → extend_lesson_evidence
    skipped_reason: Optional[str] = None  # already_processed / no_failures / trace_not_found
    error: Optional[str] = None  # 异常字符串（fail-open 后给 caller 记日志）

    @property
    def total_lessons_touched(self) -> int:
        return self.lessons_added + self.lessons_extended


class LessonIngestor:
    """trace 关闭后扫一遍，把失败抽成 candidate lessons 入 backend。

    生命周期：main.py 装配一份共享实例，注入 Harness。Harness 在每个
    user-turn 完整结束（含 evaluator retry）后调 process_trace_path()。

    线程安全：内部 _processed_paths 是 set，单进程读写不需锁。
    """

    def __init__(self, backend: MemoryBackend, *, ttl_days: int = 14):
        self._backend = backend
        self._extractor = EpisodeExtractor()
        self._classifier = TraceErrorClassifier()
        self._generator = LessonGenerator(ttl_days=ttl_days)
        self._processed_paths: Set[str] = set()

    def process_trace_path(self, path: Path) -> IngestResult:
        """读 trace JSONL，抽 episode，生成并 upsert lessons 进 backend。

        幂等：同一 trace_path 不会处理两次（_processed_paths 集合）。

        异常路径：
        - trace 文件不存在 → IngestResult(skipped_reason="trace_not_found")
        - extractor 异常 → IngestResult(error=...)
        - episode 无 failure → IngestResult(skipped_reason="no_failures")
        - 单条 lesson 生成 / upsert 失败 → log warning 跳过，继续处理其他 failures
        """
        path_str = str(path)
        if path_str in self._processed_paths:
            return IngestResult(skipped_reason="already_processed")
        self._processed_paths.add(path_str)

        try:
            episode = self._extractor.extract_from_file(Path(path))
        except FileNotFoundError:
            return IngestResult(skipped_reason="trace_not_found")
        except Exception as e:
            _logger.warning(f"LessonIngestor 抽 episode 失败 {path}: {e}")
            return IngestResult(error=f"{type(e).__name__}: {e}")

        if episode is None:
            return IngestResult(skipped_reason="no_failures")

        # add_episode 在两个 backend 都是 INSERT OR REPLACE，幂等
        try:
            self._backend.add_episode(episode)
        except Exception as e:
            _logger.warning(f"LessonIngestor add_episode 失败 {episode.episode_id}: {e}")
            return IngestResult(
                episode_id=episode.episode_id,
                error=f"add_episode: {type(e).__name__}: {e}",
            )

        added = 0
        extended = 0
        for fe in episode.failures:
            try:
                error_class = self._classifier.classify(fe, episode.failures)
                lesson = self._generator.generate(episode, fe, error_class)
                # 键审计：cause_sig 错位（错分类/提取失败）比 args_hash 键更隐蔽，
                # 留四元组日志可查
                _logger.info(
                    f"lesson key: id={lesson.lesson_id} tool={fe.tool_key} "
                    f"class={error_class} sig={_cause_signature(fe, error_class)!r} "
                    f"args_hash={fe.args_hash}"
                )
            except Exception as e:
                _logger.warning(
                    f"LessonIngestor lesson 生成失败 (episode={episode.episode_id}, "
                    f"failure_iter={fe.iteration}): {e}"
                )
                continue

            try:
                self._backend.add_lesson(lesson)
                added += 1
            except LessonAlreadyExists:
                try:
                    self._backend.extend_lesson_evidence(
                        lesson.lesson_id,
                        episode_id=episode.episode_id,
                        sample_trace_path=lesson.evidence.sample_trace_path,
                        sample_failure_iteration=lesson.evidence.sample_failure_iteration,
                        sample_error_message=lesson.evidence.sample_error_message,
                        # generator 已把 gate 的结构化示范填进 lesson.example；
                        # 透到 backend 让 extend 首次写入（跨 trace 聚合同一条 lesson）。
                        example=lesson.example,
                    )
                    extended += 1
                except Exception as e:
                    _logger.warning(
                        f"LessonIngestor extend_evidence 失败 lesson_id={lesson.lesson_id}: {e}"
                    )
                    continue
            except Exception as e:
                _logger.warning(
                    f"LessonIngestor add_lesson 失败 lesson_id={lesson.lesson_id}: {e}"
                )
                continue

        return IngestResult(
            episode_id=episode.episode_id,
            lessons_added=added,
            lessons_extended=extended,
        )
