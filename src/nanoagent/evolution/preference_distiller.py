"""PreferenceDistiller-Lite：从 skill 对话历史 + 显式反馈中蒸馏出 NL preference summary。

定位：
  - POPI-inspired NL summary（不是 KV schema），跟 nanoagent "schema 只传建议、
    不强制约束" 哲学一致
  - Conservative gate：LLM 输出 update 但**不一定**采纳——4 个准入条件任一满足才写
  - Evidence-backed：每次决策记录 turn_id + signal，写 audit JSONL，便于事后核验
  - Fail-open：副 LLM 异常 / JSON 损坏 / 不 eligible / debounce 全部走 audit + 返回 keep

跟飞轮（trace 层 lesson）的世界观对照：
  飞轮：失败修复 → 可解释 / 可量化 → PromotionGate 状态机
  本类：偏好漂移 → 自然演化 / nuance → LLM rewrite + conservative gate
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Final, Literal, Optional, Protocol

from nanoagent.core.logger import get_logger
from nanoagent.evolution.skill_preference_audit import (
    DistillAuditRecord,
    NullAuditWriter,
)
from nanoagent.evolution.skill_preference_store import (
    PreferenceEvidence,
    SkillPreference,
    SkillPreferenceStore,
)

_logger = get_logger("preference_distiller")

Action = Literal["keep", "update", "clear"]


# 强偏好词（lexical hints）：单次出现就让 conservative gate 通过 update。
# 跟 ObligationTracker / FailureMemory 同款 substring 匹配机制——零状态 / 可审计。
STRONG_PREFERENCE_KEYWORDS: Final[tuple[str, ...]] = (
    "以后", "别再", "不要", "更多", "少一点", "少放点", "少放些",
    "不喜欢", "优先", "重点关注", "记住", "永远", "不再", "更喜欢",
)


@dataclass(frozen=True)
class DistillDecision:
    """PreferenceDistiller.distill 返回结构。"""
    action: Action
    new_summary: str = ""                    # action=update 时填新内容；clear/keep 留空
    confidence: Literal["low", "medium", "high"] = "low"
    evidence: tuple[PreferenceEvidence, ...] = field(default_factory=tuple)
    why_changed: str = ""


class _LLMLike(Protocol):
    """副 LLM 接口最小约定（LLMClient.think 已满足）。"""

    def think(self, messages: list[dict], temperature: float = 0) -> str: ...  # noqa: E704


_DISTILLER_PROMPT: Final[str] = """你是一个 user preference distiller。基于用户跟某个 skill 的近期对话，
输出该 skill 的 NL preference summary。

# 当前 summary（可能为空）
{current_summary}

# 最近 N turn 用户消息（含 turn_id，按时间从早到晚）
{recent_messages}

# 用户显式 /feedback 历史（最近若干条）
{feedback_history}

# 任务
判断当前对话信号是否足以更新 summary：
- keep：信号弱不更新；new_summary 留空
- update：用户表现出新的稳定偏好；new_summary 给出 ≤150 字中文 NL summary
- clear：旧偏好显然已不适用；new_summary 留空

# 输出严格 JSON（**必须**只返回一个 JSON 对象，不要任何其它文字、解释、Markdown 代码块）
{{
  "action": "keep | update | clear",
  "new_summary": "更新后的 NL summary（≤150 字中文，action=update 才必填）",
  "confidence": "low | medium | high",
  "evidence": [
    {{"turn_id": "<上面列表里出现的 turn_id>", "signal": "<具体信号描述>"}}
  ],
  "why_changed": "为什么这样决定（一句话）"
}}

# 规则
- 信号不足时返回 action=keep
- 不要编造 turn_id（必须出自上面"最近 N turn 用户消息"列表）
- new_summary 是给主 LLM 看的 soft guidance，不是 hard rule
- 兴趣漂移自然衰减：旧 summary 里长期无新证据的偏好可在 update 时删除
- confidence=high 仅在 ≥ 3 个独立证据 + 跨 ≥ 2 turn 时给
"""


@dataclass
class _SignalContext:
    """从输入信号推导出的 conservative gate 准入维度。"""
    has_explicit_feedback: bool = False
    has_strong_keyword: bool = False
    valid_evidence_count: int = 0  # LLM 回的 evidence 中 turn_id 真存在的条数


class PreferenceDistiller:
    """skill 级偏好蒸馏器。

    用法：
        distiller = PreferenceDistiller(llm=qwen_flash_llm, store=pref_store, audit_writer=audit)
        distiller.maybe_distill(skill="ai-digest",
                                recent_user_messages=[{"turn_id": "m-1", "content": "..."}],
                                feedback_history=["以后日报每条 ≤50 字"],
                                trigger="skill_switch")

    fail-open：所有异常路径不抛，写 audit 后返回 None。
    """

    def __init__(
        self,
        llm: _LLMLike,
        store: SkillPreferenceStore,
        audit_writer=None,
        *,
        min_turns: int = 10,
        debounce_minutes: int = 30,
    ):
        self._llm = llm
        self._store = store
        self._audit = audit_writer if audit_writer is not None else NullAuditWriter()
        self._min_turns = min_turns
        self._debounce_minutes = debounce_minutes

    # ============================================================
    # 主入口
    # ============================================================

    def maybe_distill(
        self,
        skill: str,
        recent_user_messages: list[dict],
        *,
        feedback_history: Optional[list[str]] = None,
        trigger: str = "manual",
    ) -> Optional[DistillDecision]:
        """eligibility / debounce 检查 + LLM 调用 + conservative gate + 写入 store/audit。

        Returns:
            DistillDecision 当**真**对 store 做了更改（update / clear）；
            None 当 keep / 不 eligible / debounce / gate_blocked / 异常降级。
        """
        feedback_history = feedback_history or []

        # 1. eligibility
        if len(recent_user_messages) < self._min_turns:
            self._audit.write(DistillAuditRecord(
                skill=skill,
                trigger=trigger,
                action="not_eligible",
                why=f"recent_user_messages={len(recent_user_messages)} < min_turns={self._min_turns}",
            ))
            return None

        # 2. debounce
        existing = self._store.get(skill)
        if existing and self._is_debounced(existing.updated_at):
            self._audit.write(DistillAuditRecord(
                skill=skill,
                trigger=trigger,
                action="debounce",
                old_summary=existing.value,
                why=f"距上次 distill < {self._debounce_minutes} min",
            ))
            return None

        # 3. 调 LLM
        try:
            raw = self._call_llm(skill, existing, recent_user_messages, feedback_history)
        except Exception as e:
            self._audit.write(DistillAuditRecord(
                skill=skill,
                trigger=trigger,
                action="parse_error",
                old_summary=existing.value if existing else "",
                why=f"llm_call_failed: {type(e).__name__}: {e}",
            ))
            return None

        decision = self._parse_decision(raw, recent_user_messages)
        if decision is None:
            self._audit.write(DistillAuditRecord(
                skill=skill,
                trigger=trigger,
                action="parse_error",
                old_summary=existing.value if existing else "",
                why="invalid_json_or_schema",
            ))
            return None

        # 4. conservative gate（仅对 action=update 检查）
        signals = self._derive_signals(
            recent_user_messages=recent_user_messages,
            feedback_history=feedback_history,
            decision=decision,
        )

        if decision.action == "update":
            if not self._passes_gate(signals):
                self._audit.write(DistillAuditRecord(
                    skill=skill,
                    trigger=trigger,
                    action="gate_blocked",
                    confidence=decision.confidence,
                    old_summary=existing.value if existing else "",
                    new_summary=decision.new_summary,
                    why=(
                        f"gate_failed: feedback={signals.has_explicit_feedback} "
                        f"keyword={signals.has_strong_keyword} "
                        f"evidence_count={signals.valid_evidence_count}"
                    ),
                    evidence_turn_ids=[e.turn_id for e in decision.evidence],
                ))
                return None
            self._apply_update(skill, decision)
            self._audit.write(DistillAuditRecord(
                skill=skill,
                trigger=trigger,
                action="update",
                confidence=decision.confidence,
                old_summary=existing.value if existing else "",
                new_summary=decision.new_summary,
                why=decision.why_changed,
                evidence_turn_ids=[e.turn_id for e in decision.evidence],
            ))
            return decision

        if decision.action == "clear":
            old_value = existing.value if existing else ""
            if existing:
                self._store.delete(skill)
            self._audit.write(DistillAuditRecord(
                skill=skill,
                trigger=trigger,
                action="clear",
                confidence=decision.confidence,
                old_summary=old_value,
                why=decision.why_changed,
                evidence_turn_ids=[e.turn_id for e in decision.evidence],
            ))
            return decision

        # action == "keep"
        self._audit.write(DistillAuditRecord(
            skill=skill,
            trigger=trigger,
            action="keep",
            confidence=decision.confidence,
            old_summary=existing.value if existing else "",
            why=decision.why_changed or "signal_too_weak",
            evidence_turn_ids=[e.turn_id for e in decision.evidence],
        ))
        return None

    # ============================================================
    # eligibility / debounce
    # ============================================================

    def _is_debounced(self, updated_at_iso: str) -> bool:
        if not updated_at_iso:
            return False
        try:
            last = datetime.fromisoformat(updated_at_iso)
        except ValueError:
            return False
        return datetime.now() - last < timedelta(minutes=self._debounce_minutes)

    # ============================================================
    # LLM 调用 + 解析
    # ============================================================

    def _call_llm(
        self,
        skill: str,
        existing: Optional[SkillPreference],
        recent_user_messages: list[dict],
        feedback_history: list[str],
    ) -> str:
        current_summary = existing.value if existing else "(尚无)"
        msg_lines = [
            f"- [{m.get('turn_id', '?')}] {m.get('content', '')}"
            for m in recent_user_messages
        ]
        fb_lines = [f"- {fb}" for fb in feedback_history] or ["(无)"]
        prompt = _DISTILLER_PROMPT.format(
            current_summary=current_summary,
            recent_messages="\n".join(msg_lines) if msg_lines else "(无)",
            feedback_history="\n".join(fb_lines),
        )
        return self._llm.think(
            messages=[
                {"role": "system", "content": f"分析 skill={skill} 的用户偏好。"},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

    @staticmethod
    def _parse_decision(
        raw: str,
        recent_user_messages: list[dict],
    ) -> Optional[DistillDecision]:
        """容忍 LLM 在 JSON 外面包 ```json``` / 解释文字。失败返 None。"""
        text = (raw or "").strip()
        # 剥 markdown 代码块
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        # 抓最外层 JSON 对象
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(data, dict):
            return None

        action = data.get("action")
        if action not in ("keep", "update", "clear"):
            return None

        confidence = data.get("confidence", "low")
        if confidence not in ("low", "medium", "high"):
            confidence = "low"

        # 过滤 hallucinated turn_id
        valid_ids = {str(m.get("turn_id", "")) for m in recent_user_messages}
        evidence_raw = data.get("evidence") or []
        evidence: list[PreferenceEvidence] = []
        if isinstance(evidence_raw, list):
            for item in evidence_raw:
                if not isinstance(item, dict):
                    continue
                tid = str(item.get("turn_id", ""))
                if tid and tid in valid_ids:
                    evidence.append(PreferenceEvidence(
                        turn_id=tid,
                        signal=str(item.get("signal", ""))[:200],
                    ))

        return DistillDecision(
            action=action,  # type: ignore[arg-type]
            new_summary=str(data.get("new_summary", "")).strip()[:500],
            confidence=confidence,  # type: ignore[arg-type]
            evidence=tuple(evidence),
            why_changed=str(data.get("why_changed", ""))[:200],
        )

    # ============================================================
    # Conservative gate
    # ============================================================

    @staticmethod
    def _derive_signals(
        recent_user_messages: list[dict],
        feedback_history: list[str],
        decision: DistillDecision,
    ) -> _SignalContext:
        has_keyword = any(
            kw in (m.get("content") or "")
            for m in recent_user_messages
            for kw in STRONG_PREFERENCE_KEYWORDS
        ) or any(
            kw in fb for fb in feedback_history for kw in STRONG_PREFERENCE_KEYWORDS
        )
        return _SignalContext(
            has_explicit_feedback=bool(feedback_history),
            has_strong_keyword=has_keyword,
            valid_evidence_count=len(decision.evidence),
        )

    @staticmethod
    def _passes_gate(signals: _SignalContext) -> bool:
        """4 准入条件简化为 3 个可观测信号（同方向 ≥2 / mark+对话双支持
        合到 valid_evidence_count ≥ 2 这一条里——MVP 不做 mark history 单独维度）：
          1. 显式 /feedback 出现
          2. 强偏好关键词
          3. LLM 给出 ≥ 2 条有效 evidence
        """
        return (
            signals.has_explicit_feedback
            or signals.has_strong_keyword
            or signals.valid_evidence_count >= 2
        )

    # ============================================================
    # 写入
    # ============================================================

    def _apply_update(self, skill: str, decision: DistillDecision) -> None:
        if not decision.new_summary.strip():
            # update 但 summary 空——降级 keep（gate 之前的 LLM 输出格式异常）
            return
        pref = SkillPreference(
            value=decision.new_summary.strip(),
            source="auto_distilled",
            updated_at=datetime.now().isoformat(timespec="seconds"),
            confidence=decision.confidence,
            evidence=decision.evidence,
        )
        self._store.set(skill, pref)
