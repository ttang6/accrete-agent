"""SkillPreferenceAuditWriter：PreferenceDistiller 决策的 JSONL 审计 sink。

每次 distill（含 keep / update / clear / gate_blocked / debounce / parse_error）
追加一行到 data/memory/skill_preferences_audit.jsonl，事后用 jq / tail 反推
distiller 行为：哪条 evidence 触发了 update、哪些 update 被 gate 拦下。

风格对齐 promotion_audit.py：append-only，open-write-close，无常驻句柄。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class DistillAuditRecord:
    """一次 distill 调用的完整审计记录。"""

    skill: str
    trigger: str                              # skill_switch / session_end / manual
    action: str                               # keep / update / clear / gate_blocked / debounce / parse_error / not_eligible
    confidence: str = "low"
    old_summary: str = ""
    new_summary: str = ""
    why: str = ""                             # LLM 给的 why_changed 或 gate 阻挡原因
    evidence_turn_ids: list[str] = field(default_factory=list)
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp or datetime.now(timezone.utc).isoformat(),
            "skill": self.skill,
            "trigger": self.trigger,
            "action": self.action,
            "confidence": self.confidence,
            "old_summary": self.old_summary,
            "new_summary": self.new_summary,
            "why": self.why,
            "evidence_turn_ids": list(self.evidence_turn_ids),
        }


class SkillPreferenceAuditWriter:
    """JSONL append-only sink。fail-open，写失败不抛。"""

    def __init__(self, log_path: Path):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: DistillAuditRecord) -> None:
        try:
            with self._log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except OSError:
            pass


class NullAuditWriter:
    """装配未启用 audit 时的 no-op 占位（避免 distiller 内部到处 if writer is not None）。"""

    def write(self, record: DistillAuditRecord) -> None:  # noqa: ARG002
        pass
