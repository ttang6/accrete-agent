"""SkillPreferenceStore：自动推断偏好的存储（独立于 UserFacts）。

定位：
  - 存放 distiller 自动推断出的 skill 级 NL preference summary
  - **物理隔离 UserFacts**——UserFacts 哲学是 explicit-only，自动推断走独立文件
  - 风格对齐 UserFacts（原子写 + lazy load）；语义对齐 ReflexionStore.render_for_skill
    （返回 markdown body，由 SkillLoader 拼标题块）

文件：
    data/memory/skill_preferences.json
    {
      "ai-digest": {
        "value": "...NL summary...",
        "source": "auto_distilled",
        "updated_at": "2026-05-09T15:30:00",
        "confidence": "medium",
        "evidence": [{"turn_id": "m-3", "signal": "..."}]
      }
    }
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final, Literal, Optional

from nanoagent.core.logger import get_logger

_logger = get_logger("skill_preference")

Confidence = Literal["low", "medium", "high"]

_PREFERENCE_HEADER: Final[str] = (
    "## 用户偏好（自动推断，软约束）\n\n"
    "以下内容来自近期对话与显式反馈的自动归纳，可能不完整或过时。"
    "仅在不冲突于当前用户请求、显式反馈和本 skill 硬性规则时使用。\n\n"
)


@dataclass(frozen=True)
class PreferenceEvidence:
    turn_id: str
    signal: str

    def to_dict(self) -> dict:
        return {"turn_id": self.turn_id, "signal": self.signal}

    @classmethod
    def from_dict(cls, data: dict) -> "PreferenceEvidence":
        return cls(turn_id=str(data.get("turn_id", "")), signal=str(data.get("signal", "")))


@dataclass(frozen=True)
class SkillPreference:
    value: str
    source: str = "auto_distilled"
    updated_at: str = ""
    confidence: Confidence = "low"
    evidence: tuple[PreferenceEvidence, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "value": self.value,
            "source": self.source,
            "updated_at": self.updated_at,
            "confidence": self.confidence,
            "evidence": [e.to_dict() for e in self.evidence],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SkillPreference":
        evidence_raw = data.get("evidence") or []
        evidence: list[PreferenceEvidence] = []
        if isinstance(evidence_raw, list):
            for item in evidence_raw:
                if isinstance(item, dict):
                    evidence.append(PreferenceEvidence.from_dict(item))
        conf = data.get("confidence", "low")
        if conf not in ("low", "medium", "high"):
            conf = "low"
        return cls(
            value=str(data.get("value", "")),
            source=str(data.get("source", "auto_distilled")),
            updated_at=str(data.get("updated_at", "")),
            confidence=conf,  # type: ignore[arg-type]
            evidence=tuple(evidence),
        )


class SkillPreferenceStore:
    """skill 级 distilled preference 的持久化层。

    用法：
        store = SkillPreferenceStore(Path("data/memory/skill_preferences.json"))
        pref = SkillPreference(value="...", confidence="medium", evidence=(...,))
        store.set("ai-digest", pref)
        block = store.render_for_skill("ai-digest")  # markdown 给 SkillLoader 拼
    """

    def __init__(self, path: Path):
        self._path = Path(path)
        self._prefs: dict[str, SkillPreference] = {}
        self._load()

    # ============================================================
    # 读 / 写 / 删
    # ============================================================

    def get(self, skill: str) -> Optional[SkillPreference]:
        return self._prefs.get(skill)

    def set(self, skill: str, preference: SkillPreference) -> None:
        # 自动补 updated_at（caller 没设）
        if not preference.updated_at:
            preference = SkillPreference(
                value=preference.value,
                source=preference.source,
                updated_at=datetime.now().isoformat(timespec="seconds"),
                confidence=preference.confidence,
                evidence=preference.evidence,
            )
        self._prefs[skill] = preference
        self._save()

    def delete(self, skill: str) -> bool:
        if skill in self._prefs:
            del self._prefs[skill]
            self._save()
            return True
        return False

    def clear_all(self) -> int:
        n = len(self._prefs)
        self._prefs.clear()
        self._save()
        return n

    def list_skills(self) -> list[str]:
        return list(self._prefs.keys())

    def __len__(self) -> int:
        return len(self._prefs)

    def __contains__(self, skill: str) -> bool:
        return skill in self._prefs

    # ============================================================
    # 渲染（供 SkillLoader 后置注入）
    # ============================================================

    def render_for_skill(self, skill: str) -> str:
        """返回带标题的完整 preference 块；无记录返回空串。

        由 SkillLoader 直接拼到 body 之后（区别于 reflexion 的前置策略）。
        """
        pref = self._prefs.get(skill)
        if pref is None or not pref.value.strip():
            return ""
        return _PREFERENCE_HEADER + pref.value.strip()

    # ============================================================
    # 持久化
    # ============================================================

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            _logger.warning(f"{self._path} 加载失败: {type(e).__name__}: {e}；从空开始")
            return
        if not isinstance(data, dict):
            _logger.warning(f"{self._path} 格式不符（非 dict），忽略")
            return
        valid: dict[str, SkillPreference] = {}
        for skill, entry in data.items():
            if not isinstance(entry, dict) or "value" not in entry:
                _logger.warning(f"{self._path} 跳过损坏条目 {skill!r}")
                continue
            valid[skill] = SkillPreference.from_dict(entry)
        self._prefs = valid

    def _save(self) -> None:
        """原子写：tempfile + os.replace。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {skill: pref.to_dict() for skill, pref in self._prefs.items()}
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self._path.stem}_",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp_name, self._path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as e:
            _logger.warning(f"写 {self._path} 失败（已忽略）: {e}")
