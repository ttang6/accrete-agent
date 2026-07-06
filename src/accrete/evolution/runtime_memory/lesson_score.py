"""Derived injection score that gates lesson recall.

A lesson stores only its raw ledger (helped/hurt counts plus evidence episode
ids); its recall eligibility is derived on read as a single score, replacing the
old stored status state machine. Design rationale (why the ledger is stored raw,
credit-cap semantics, the CAP-strikes behavior, no time decay) lives in the
Phase 0 plan doc and the relevant commit messages, not here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accrete.evolution.runtime_memory.schema import RuntimeLesson

INIT_SCORE: int = 1
INJECT_THRESHOLD: int = 1
SCORE_CAP: int = 3


def _evidence_count(lesson: "RuntimeLesson") -> int:
    return len(lesson.evidence.source_episode_ids) or 1


def compute_score(lesson: "RuntimeLesson") -> int:
    """Compute a lesson's injection score from its ledger.

    score = min(CAP, INIT + helped + (evidence - 1)) - hurt

    Args:
        lesson: The lesson whose stats and evidence back the score.

    Returns:
        The score. Not clamped at the low end, so it may be negative.
    """
    credit = min(
        SCORE_CAP,
        INIT_SCORE + lesson.stats.helped_count + (_evidence_count(lesson) - 1),
    )
    return credit - lesson.stats.hurt_count


def is_active(lesson: "RuntimeLesson") -> bool:
    """Return whether the lesson is eligible for recall (score >= threshold)."""
    return compute_score(lesson) >= INJECT_THRESHOLD


def status_label(lesson: "RuntimeLesson") -> str:
    """Return a display label ("active" or "dormant") derived from the score."""
    return "active" if is_active(lesson) else "dormant"
