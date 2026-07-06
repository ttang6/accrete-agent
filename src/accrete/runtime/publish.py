"""发布流程的副作用层：从日报机器块抽采用条目 → 出处核对 → 写去重历史。

设计锚点（与用户定案一致）：
- **机器块是唯一真值**。模型在日报末尾附一个 ```json 块（`adopted_items`），副作用
  层只读它；散文日报给人看、不参与抽取，也不校验两者一致——以机器块为准。这样抽取
  确定、不靠解析散文那种脆。
- **出处核对走最小版**：一条采用条目的 fingerprint 必须出现在本轮抓回的候选原文里，
  否则判为凭空编造、剔除（不写进去重历史）。链接 resolve / 日期窗等更强校验留后。
- **mark 由流程自动写**，不再是模型工具：发布轮才跑到这里，结构上杜绝随手查污染历史。

这些都是纯函数 / 轻 IO，harness 的发布流程按序调用；与"日报"业务知识隔离在 skill
侧的调用者，本模块只认 fingerprint/source/title 三元组。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from accrete.core.logger import get_logger
from accrete.core.paths import data_dir

_logger = get_logger("publish")

# 去重历史文件（与 skills/ai-digest/scripts/dup_check.py 同一份，格式对齐）。
_HISTORY_PATH = data_dir("memory") / "digest_reported.jsonl"

# 抓 ```json ... ``` 围栏块（DOTALL 跨行）。机器块在散文之外、由模型显式产出，
# 这里只在"最终答案"上扫一次，不是热路径的脆正则往返。
_JSON_FENCE_RE = re.compile(r"```json\s*\n(.*?)\n```", re.DOTALL)


def extract_adopted_items(answer: str) -> List[Dict[str, Any]]:
    """从日报答案里抽机器块的 `adopted_items`。

    取**最后一个**含 `adopted_items` 的 ```json 块（约定附在答案末尾）。每条规整成
    {fingerprint, source, title}；缺 fingerprint 的条目丢弃。无机器块 / 解析失败 →
    返回空列表（发布流程据此知道"本期没有可登记条目"，不报错）。
    """
    if not isinstance(answer, str) or not answer:
        return []
    found: List[Dict[str, Any]] = []
    for m in _JSON_FENCE_RE.finditer(answer):
        try:
            parsed = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("adopted_items"), list):
            found = parsed["adopted_items"]  # 后出现者覆盖，取最后一个块
    out: List[Dict[str, Any]] = []
    for raw in found:
        if not isinstance(raw, dict):
            continue
        fp = str(raw.get("fingerprint") or "").strip()
        if not fp:
            continue
        out.append({
            "fingerprint": fp,
            "source": str(raw.get("source") or "").strip(),
            "title": str(raw.get("title") or "").strip(),
        })
    return out


def check_provenance(
    items: List[Dict[str, Any]], candidates: List[str]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """出处核对（最小版）：fingerprint 必须在本轮候选原文里出现，否则剔除。

    返回 (kept, dropped)。candidates 是本轮所有 tool 成功输出的原文（AgentRuntimeState
    攒的）。判据是子串命中——便宜、防的是"凭空造一条候选集里根本没有的 paper"这种
    最丢人的失败；不做链接 resolve（留后）。无候选（candidates 空）时一律判 dropped，
    宁可不登记也不登记来路不明的条目。
    """
    blob = "\n".join(candidates) if candidates else ""
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for it in items:
        if blob and it["fingerprint"] in blob:
            kept.append(it)
        else:
            dropped.append(it)
    return kept, dropped


def mark_published_items(items: List[Dict[str, Any]]) -> int:
    """把通过核对的条目写进去重历史（append JSONL，按 fingerprint 去重）。

    这是原 dup_check action=mark 的发布流程版——发布轮自动写，模型不再调。返回新写入
    条数。任何 IO 异常吞掉（发布的日报已产出，登记失败不该影响交付）。
    """
    if not items:
        return 0
    try:
        existing = _load_existing_fingerprints()
        now_iso = datetime.now().isoformat(timespec="seconds")
        written = 0
        _HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _HISTORY_PATH.open("a", encoding="utf-8") as f:
            for it in items:
                fp = it["fingerprint"]
                if fp in existing:
                    continue
                existing.add(fp)
                f.write(json.dumps({
                    "fingerprint": fp,
                    "source": it["source"],
                    "title": it["title"],
                    "reported_at": now_iso,
                }, ensure_ascii=False) + "\n")
                written += 1
        return written
    except OSError as e:
        _logger.warning(f"[publish] 写去重历史失败（已 fail-open）：{e}")
        return 0


def _load_existing_fingerprints() -> set:
    out: set = set()
    if not _HISTORY_PATH.is_file():
        return out
    try:
        with _HISTORY_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                fp = rec.get("fingerprint")
                if fp:
                    out.add(fp)
    except OSError:
        pass
    return out
