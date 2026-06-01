"""偏好蒸馏的 WHEN 层（纯决策）：session_end / skill_switch / turn_end 三信号，
都先过 floor guard（新增 user 轮次 >= floor）；cadence = (轮次 >= N) OR (距上次 >= T 分钟)。

marker 放 session.meta["distill"]，按 message index 记：
    {"last_message_index": int, "last_distilled_at": ISO str}
时间分支仅在有 last_distilled_at 时生效，否则退化为纯轮次驱动。
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

EVENT_TURN_END = "turn_end"
EVENT_SKILL_SWITCH = "skill_switch"
EVENT_SESSION_END = "session_end"

_MARKER_KEY = "distill"


def read_marker(meta: dict) -> dict:
    """从 session.meta 取 distill marker；无则返回零值 marker。"""
    m = meta.get(_MARKER_KEY)
    if isinstance(m, dict):
        return m
    return {"last_message_index": 0, "last_distilled_at": ""}


def make_marker(history: list[dict], now: datetime) -> dict:
    """蒸馏发生后推进 marker 到当前位置 + 时间。"""
    return {
        "last_message_index": len(history),
        "last_distilled_at": now.isoformat(timespec="seconds"),
    }


def new_user_turns(history: list[dict], marker: dict) -> int:
    """自上次 marker 以来新增的 user 消息条数。"""
    idx = int(marker.get("last_message_index", 0) or 0)
    return sum(1 for m in history[idx:] if m.get("role") == "user")


def elapsed_minutes(marker: dict, now: datetime) -> Optional[float]:
    """距上次蒸馏的分钟数；marker 无 last_distilled_at 时返回 None（时间分支不生效）。"""
    last = marker.get("last_distilled_at")
    if not last:
        return None
    try:
        t = datetime.fromisoformat(last)
    except (ValueError, TypeError):
        return None
    return (now - t).total_seconds() / 60.0


class DistillScheduler:
    def __init__(self, *, floor: int = 3, cadence_turns_n: int = 30, cadence_minutes_t: int = 30):
        self._floor = floor
        self._cadence_turns_n = cadence_turns_n
        self._cadence_minutes_t = cadence_minutes_t

    def decide(self, *, event: str, marker: dict, history: list[dict], now: datetime) -> Optional[str]:
        """返回 trigger reason（"cadence"/"skill_switch"/"session_end"）或 None。"""
        turns = new_user_turns(history, marker)
        if turns < self._floor:
            return None  # floor guard，所有信号共用

        if event in (EVENT_SKILL_SWITCH, EVENT_SESSION_END):
            return event  # 边界收尾：过 floor 即触发

        if event == EVENT_TURN_END:
            if turns >= self._cadence_turns_n:
                return "cadence"
            elapsed = elapsed_minutes(marker, now)
            if elapsed is not None and elapsed >= self._cadence_minutes_t:
                return "cadence"
            return None

        return None
