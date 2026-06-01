"""dup_check.py — ai-digest 的历史去重 script。

LLM 调用：

  1) 拉到 candidates 后，过滤已报过的：
     skill_exec(skill="ai-digest", script="dup_check", args={
         "action": "check",
         "fingerprints": ["2510.12345", "https://blog.x/post", "owner/repo", ...]
     })

  2) 日报定稿后，把采纳的条目写入历史：
     skill_exec(skill="ai-digest", script="dup_check", args={
         "action": "mark",
         "items": [
             {"fingerprint": "2510.12345", "source": "fetch_hf", "title": "..."},
             {"fingerprint": "https://blog.x/post", "source": "fetch_rss", "title": "..."},
         ]
     })

存储：`data/memory/digest_reported.jsonl`，全局单例（跨 channel 共享）。
每行 JSON：{fingerprint, source, title, reported_at}。

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

# 默认全局单例；env override 让 dose-response / dogfood driver 把每个 sandbox 的
# 历史隔离到独立路径——避免跨实验 dup 表污染（详见 docs/codex_dogfood/
# preference_dose_response_driver.py 的 setup_isolation 注释）。
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


def _handle_mark(args: dict) -> int:
    """LLM 传入本次日报采纳的条目列表 → 追加写入 JSONL 历史。

    items 每条要求 `fingerprint`（必填），可选 `source` / `title`。
    同一 fingerprint 已存在 → 不重复写（保留首次报道记录）。
    """
    items = args.get("items") or []
    if not isinstance(items, list):
        print_error("items 必须是对象列表")
        return 1

    valid: list[dict] = []
    for raw in items:
        if not isinstance(raw, dict):
            continue
        fp = raw.get("fingerprint")
        if not isinstance(fp, str) or not fp:
            continue
        valid.append(
            {
                "fingerprint": fp,
                "source": raw.get("source") or "",
                "title": raw.get("title") or "",
            }
        )

    if not valid:
        print_error("items 中无合法条目（需至少含 fingerprint 字段）")
        return 1

    history = _load_history()
    existing_fps = set(history.keys())
    now_iso = datetime.now().isoformat(timespec="seconds")

    _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    with open(_HISTORY_PATH, "a", encoding="utf-8") as f:
        for item in valid:
            fp = item["fingerprint"]
            if fp in existing_fps:
                skipped += 1
                continue
            record = {
                "fingerprint": fp,
                "source": item["source"],
                "title": item["title"],
                "reported_at": now_iso,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            existing_fps.add(fp)
            written += 1

    lines = [f"# dedup 记录完成：新增 {written} 条，跳过 {skipped} 条（已存在于历史）"]
    if written > 0:
        lines.append("")
        lines.append("新写入：")
        for item in valid:
            if item["fingerprint"] not in history:
                title = item["title"] or "（无标题）"
                lines.append(f"- `{item['fingerprint']}` — {title}")
    print("\n".join(lines))
    return 0


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()
    action = (args.get("action") or "").strip().lower()

    if action == "check":
        return _handle_check(args)
    if action == "mark":
        return _handle_mark(args)

    print_error(f"未知 action '{action}'，可选：check / mark")
    return 1


if __name__ == "__main__":
    sys.exit(main())
