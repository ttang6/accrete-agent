"""PreferenceCommitter：偏好写入的**存储安全**层。

职责边界（关键）：
- **语义判断**（值不值得写）= policy sub-agent 的 `write_allowed`，不在这里。
- **存储安全**（数据完整性 + 并发安全）= 本类。哪怕 sub-agent 说 `write_allowed=True`，
  committer 仍**独立**把关：schema / update 时 new_summary 非空 / evidence turn_id 真实 /
  长度上限 / 乐观锁 / 原子写。sub-agent 一句话绕不过这层。

乐观锁用 `updated_at` 当弱版本号（sync 现状下 inert，是廉价保险；上异步时升级为 revision 字段）。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from nanoagent.evolution.skill_preference_audit import DistillAuditRecord, NullAuditWriter
from nanoagent.evolution.skill_preference_store import (
    PreferenceEvidence,
    SkillPreference,
    SkillPreferenceStore,
)

_MAX_SUMMARY_CHARS = 500


@dataclass
class CommitResult:
    committed: bool
    reason: str   # update / clear / keep / invalid / conflict
    stored: Optional[SkillPreference] = None


class PreferenceCommitter:
    def __init__(self, store: SkillPreferenceStore, audit_writer=None):
        self._store = store
        self._audit = audit_writer if audit_writer is not None else NullAuditWriter()

    def commit(
        self,
        *,
        skill: str,
        decision: dict,
        base_updated_at: str,
        valid_turn_ids: set[str],
        trigger: str = "periodic",
    ) -> CommitResult:
        """把 policy sub-agent 的 decision 落盘（或拒绝）。fail-safe：任何不确定都不写。

        decision: {action, new_summary, write_allowed, confidence, evidence[], why_changed, risk}
        base_updated_at: 组 job 时读到的 store updated_at，用于弱乐观锁
        valid_turn_ids: 输入窗口里真实存在的 turn_id 集合，用于过滤 evidence
        """
        old = self._store.get(skill)
        old_value = old.value if old else ""
        action = decision.get("action")
        why = str(decision.get("why_changed", ""))

        # --- 无写路径 ---
        if action not in ("keep", "update", "clear"):
            self._write_audit(skill, trigger, "invalid", old_value, why="bad_action")
            return CommitResult(False, "invalid")

        if action == "keep" or not bool(decision.get("write_allowed")):
            # action=keep，或 sub-agent 语义 gate 判定不该写 → 不写
            self._write_audit(skill, trigger, "keep", old_value, why=why)
            return CommitResult(False, "keep")

        # --- 弱乐观锁：写前 re-check updated_at ---
        current = self._store.get(skill)
        current_updated_at = current.updated_at if current else ""
        if current_updated_at != base_updated_at:
            self._write_audit(skill, trigger, "conflict", old_value, why="updated_at mismatch")
            return CommitResult(False, "conflict")

        # --- clear ---
        if action == "clear":
            if current is not None:
                self._store.delete(skill)
            self._write_audit(skill, trigger, "clear", old_value, why=why)
            return CommitResult(True, "clear")

        # --- update：硬校验 ---
        new_summary = str(decision.get("new_summary", "")).strip()
        if not new_summary:
            self._write_audit(skill, trigger, "invalid", old_value, why="empty new_summary on update")
            return CommitResult(False, "invalid")
        new_summary = new_summary[:_MAX_SUMMARY_CHARS]

        evidence = self._clean_evidence(decision.get("evidence"), valid_turn_ids)
        confidence = decision.get("confidence", "low")
        if confidence not in ("low", "medium", "high"):
            confidence = "low"

        pref = SkillPreference(
            value=new_summary,
            source="auto_distilled",
            updated_at=datetime.now().isoformat(timespec="seconds"),
            confidence=confidence,
            evidence=evidence,
        )
        self._store.set(skill, pref)
        self._write_audit(
            skill, trigger, "update", old_value,
            new_summary=new_summary, confidence=confidence, why=why,
            evidence_turn_ids=[e.turn_id for e in evidence],
        )
        return CommitResult(True, "update", stored=pref)

    # ------------------------------------------------------------------

    @staticmethod
    def _clean_evidence(raw, valid_turn_ids: set[str]) -> tuple[PreferenceEvidence, ...]:
        """只保留 turn_id 真实存在于输入窗口的 evidence（丢弃 LLM 编造的）。"""
        if not isinstance(raw, list):
            return ()
        out: list[PreferenceEvidence] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            tid = str(item.get("turn_id", ""))
            if tid and tid in valid_turn_ids:
                out.append(PreferenceEvidence(turn_id=tid, signal=str(item.get("signal", ""))[:200]))
        return tuple(out)

    def _write_audit(
        self, skill, trigger, action, old_summary, *,
        new_summary="", confidence="low", why="", evidence_turn_ids=None,
    ) -> None:
        self._audit.write(DistillAuditRecord(
            skill=skill,
            trigger=trigger,
            action=action,
            confidence=confidence,
            old_summary=old_summary,
            new_summary=new_summary,
            why=why,
            evidence_turn_ids=evidence_turn_ids or [],
        ))
