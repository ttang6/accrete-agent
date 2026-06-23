"""dup_check.py — ai-digest 的历史去重 script（只剩 check）。

LLM 调用——拉到 candidates 后，过滤已报过的：

     skill_exec(skill="ai-digest", script="dup_check", args={
         "action": "check",
         "fingerprints": ["2510.12345", "https://blog.x/post", "owner/repo", ...]
     })

写历史（原 action=mark）已退役：v3 阶段三-2b 把登记下沉到确定性的发布流程
（`runtime/publish.py` 按日报末尾机器块自动写）。模型只查重、不落库，故本 script
不再暴露 mark——保留 mark 只会重开"靠 LLM 自觉调工具登记"那条老缝。

存储：`data/memory/digest_reported.jsonl`，全局单例（跨 channel 共享）。
每行 JSON：{fingerprint, source, title, reported_at}（由发布流程写）。

MVP 不做 TTL——个人助手场景 10 年数据也只几 MB，真要加一行筛选即可。
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from _common import ensure_utf8_stdout, print_error, read_args


# skill 根目录 → scripts/ → ai-digest/ → skills/ → 项目根
_SKILL_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SKILL_ROOT.parent.parent

# 默认全局单例；env override 可把历史隔离到独立路径，避免并行跑（如批量测试）
# 之间 dup 表互相污染。
_HISTORY_PATH_DEFAULT = _PROJECT_ROOT / "data" / "memory" / "digest_reported.jsonl"
_HISTORY_PATH = Path(
    os.getenv("NANOAGENT_DIGEST_HISTORY_PATH", str(_HISTORY_PATH_DEFAULT))
)


def _load_history() -> dict[str, dict]:
    """读取历史记录 → {fingerprint: {source, title, reported_at}}。

    同一 fingerprint 多次记录时保留最早的 reported_at（首次报道时间）。
    文件不存在 / 损坏行容忍：坏行跳过，不让一条脏数据拖垮整个查询。
    """
    if not _HISTORY_PATH.exists():
        return {}

    history: dict[str, dict] = {}
    try:
        with open(_HISTORY_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                fp = record.get("fingerprint")
                if not isinstance(fp, str) or not fp:
                    continue
                # 保留首次报道记录（后续重复 mark 不覆盖 reported_at）
                if fp not in history:
                    history[fp] = record
    except OSError:
        return {}
    return history


def _days_ago(reported_at_str: str) -> str:
    """把 ISO 时间戳格式化为"N 天前" / "今天" / "昨天"，方便 LLM 读。"""
    if not reported_at_str:
        return "未知时间"
    try:
        reported = datetime.fromisoformat(reported_at_str)
    except ValueError:
        return "未知时间"
    delta_days = (datetime.now() - reported).days
    if delta_days <= 0:
        return "今天"
    if delta_days == 1:
        return "昨天"
    return f"{delta_days} 天前"


def _handle_check(args: dict) -> int:
    """LLM 传入 fingerprints list → 返回哪些已报过、哪些新鲜。"""
    fingerprints = args.get("fingerprints") or []
    if not isinstance(fingerprints, list):
        print_error("fingerprints 必须是字符串列表")
        return 1
    fingerprints = [fp for fp in fingerprints if isinstance(fp, str) and fp]
    if not fingerprints:
        print("# dedup 查询：无输入 fingerprint")
        return 0

    history = _load_history()
    reported = []
    fresh = []
    for fp in fingerprints:
        if fp in history:
            record = history[fp]
            reported.append(
                {
                    "fingerprint": fp,
                    "title": record.get("title") or "",
                    "days_ago": _days_ago(record.get("reported_at") or ""),
                    "source": record.get("source") or "",
                }
            )
        else:
            fresh.append(fp)

    lines = [f"# dedup 查询结果（已报 {len(reported)} / 新鲜 {len(fresh)}）"]
    lines.append("")

    if reported:
        lines.append("## 已报过（应跳过，除非有重大新进展值得重报）")
        for item in reported:
            title = item["title"] or "（无标题）"
            meta = f"{item['days_ago']}，来源 {item['source']}" if item["source"] else item["days_ago"]
            lines.append(f"- `{item['fingerprint']}` — {title}（{meta}）")
        lines.append("")

    if fresh:
        lines.append("## 新鲜（可入日报）")
        for fp in fresh:
            lines.append(f"- `{fp}`")
        lines.append("")

    print("\n".join(lines).rstrip())
    return 0


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()
    action = (args.get("action") or "").strip().lower()

    if action == "check":
        return _handle_check(args)

    if action == "mark":
        print_error(
            "action=mark 已退役：本期采纳条目由发布流程按日报末尾机器块自动登记，"
            "无需调用 dup_check 写历史。dup_check 只用于选题前 check 去重。"
        )
        return 1

    print_error(f"未知 action '{action}'，可选：check")
    return 1


if __name__ == "__main__":
    sys.exit(main())
