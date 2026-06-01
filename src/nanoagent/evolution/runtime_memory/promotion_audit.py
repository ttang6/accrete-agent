"""PromotionGate audit log sink。

把 `PromotionGate.sweep()` 的每次状态转移决策追加写入 JSONL，让 ops 用
`tail -f data/runtime/lessons/promotion_audit.jsonl` 或 jq 反推飞轮历史
（candidate→probation 的 evidence ID、retire 触发原因等）。

为什么 JSONL 不是 SQLite：
- Trace 体系（data/runtime/logs/traces/）已经是 JSONL，保持一致；
- 单进程 Harness sweep 串行追加写不存在并发问题；每行独立 JSON 易 grep / jq；
- 未来要做分布查询再引外部 ETL（DuckDB / 直接 sqlite_backend 加表），
  现在不预设抽象。

callback 抛异常的 fail-open 由 `PromotionGate.sweep` 兜底，本类不做异常吞咽。
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from nanoagent.evolution.runtime_memory.promotion_gate import PromotionDecision


class JsonlAuditWriter:
    """`PromotionGate.sweep(audit_callback=...)` 的默认 sink。

    每次 __call__ 追加一行 JSON：timestamp / lesson_id / from_status /
    to_status / reason / evidence_episode_ids。父目录构造时确保存在，单次写
    open-write-close（无常驻句柄，与 ToolOutputStore 一致）。
    """

    def __init__(self, log_path: Path):
        self._log_path = Path(log_path)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, decision: PromotionDecision) -> None:
        row = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "lesson_id": decision.lesson_id,
            "from_status": decision.from_status.value,
            "to_status": decision.to_status.value,
            "reason": decision.reason,
            "evidence_episode_ids": list(decision.evidence_episode_ids),
        }
        with self._log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
