"""偏好蒸馏的 HOW 层：产出具名 block（current_summary / recent_window / feedback）。
窗口从 marker.last_message_index - overlap 起切（留 overlap 防边界信号丢失）；
turn_id 用 history 全局 index。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DistillContext:
    blocks: list[tuple[str, str]]
    valid_turn_ids: set[str]   # 窗口里真实存在的 turn_id，给 committer 过滤 evidence


class DistillContextBuilder:
    def build(
        self,
        *,
        history: list[dict],
        marker: dict,
        current_summary: str,
        feedback_history: list[str],
        overlap: int,
    ) -> DistillContext:
        start = max(0, int(marker.get("last_message_index", 0) or 0) - overlap)
        window = [
            (f"m-{i}", str(m.get("content", "")))
            for i, m in enumerate(history)
            if i >= start and m.get("role") == "user"
        ]
        valid_ids = {tid for tid, _ in window}
        window_text = "\n".join(f"- [{tid}] {content}" for tid, content in window) or "(无)"
        fb_text = "\n".join(f"- {fb}" for fb in (feedback_history or [])) or "(无)"

        blocks = [
            ("current_summary", current_summary.strip() if current_summary else "(尚无)"),
            ("recent_window", window_text),
            ("feedback_history", fb_text),
        ]
        return DistillContext(blocks=blocks, valid_turn_ids=valid_ids)
