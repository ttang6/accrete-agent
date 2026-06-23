"""编排一次偏好蒸馏：decide → build context → run sub-agent → commit。

返回新 marker 供 caller 写回 session.meta。runner 失败或写入 conflict 时不推进
marker（留待下次重试）；其余情况推进。fail-open。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from nanoagent.core.logger import get_logger
from nanoagent.core.prompt_assets import load_prompt
from nanoagent.evolution.preference_pipeline.context import DistillContextBuilder
from nanoagent.evolution.preference_pipeline.committer import PreferenceCommitter
from nanoagent.evolution.preference_pipeline.policy import (
    POLICY_OUTPUT_SCHEMA,
    POLICY_SYSTEM_PROMPT,
)
from nanoagent.evolution.preference_pipeline.scheduler import (
    DistillScheduler,
    make_marker,
    read_marker,
)
from nanoagent.evolution.skill_preference_store import SkillPreferenceStore
from nanoagent.runtime.subagent import SubAgentContextBuilder, SubAgentRunner

_logger = get_logger("distill_pipeline")


class PreferenceDistillPipeline:
    def __init__(
        self,
        *,
        store: SkillPreferenceStore,
        scheduler: DistillScheduler,
        context_builder: DistillContextBuilder,
        runner: SubAgentRunner,
        committer: PreferenceCommitter,
        overlap: int = 3,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_iterations: int = 1,
    ):
        self._store = store
        self._scheduler = scheduler
        self._context_builder = context_builder
        self._runner = runner
        self._committer = committer
        self._overlap = overlap
        self._model = model
        self._provider = provider
        self._max_iterations = max_iterations

    def maybe_distill(
        self,
        *,
        skill: str,
        event: str,
        history: list[dict],
        meta: dict,
        feedback_history: list[str],
        now: datetime,
    ) -> Optional[dict]:
        """返回新 marker（caller 写回 meta）或 None（不触发 / 不推进）。"""
        marker = read_marker(meta)
        reason = self._scheduler.decide(event=event, marker=marker, history=history, now=now)
        if reason is None:
            return None
        try:
            current = self._store.get(skill)
            current_summary = current.value if current else ""
            base_updated_at = current.updated_at if current else ""

            ctx = self._context_builder.build(
                history=history, marker=marker,
                current_summary=current_summary, feedback_history=feedback_history,
                overlap=self._overlap,
            )
            request = SubAgentContextBuilder.build(
                system_prompt=load_prompt("preference_distiller", POLICY_SYSTEM_PROMPT),
                blocks=ctx.blocks,
                allowed_tools=(),
                output_schema=POLICY_OUTPUT_SCHEMA,
                model=self._model,
                provider=self._provider,
                max_iterations=self._max_iterations,
            )
            result = self._runner.run(request)
            if not result.ok or result.structured is None:
                return None  # runner 失败 → 不推进
            commit = self._committer.commit(
                skill=skill, decision=result.structured,
                base_updated_at=base_updated_at, valid_turn_ids=ctx.valid_turn_ids,
                trigger=reason,
            )
            if commit.reason == "conflict":
                return None  # 数据被改 → 不推进
        except Exception as e:
            _logger.warning(f"[distill-pipeline] 异常（fail-open，不推进 marker）: {type(e).__name__}: {e}")
            return None

        return make_marker(history, now)
