"""桶观察员 —— 离线扫历史 trace,把工具失败聚簇成"失败家族 + 归一化签名",产人读报告。

定位(总路线图 Track E):**报告生成器,不是学习系统**。它不铸类、不写 lesson——只把
9千条历史 trace 里的失败按 (op, 归一化签名) 聚簇、数复现,unknown 桶单列供人判。铸不铸
新类由人按三件套(typed 检测 + 处置 + advice 模板)裁。

"旧→新标签 shim"就在这里落地为**复用当前分类器**:`tool_failure.is_tool_failure` 判失败、
`classify_tool_failure` 打 base-3 家族——把今天的词汇套在旧 trace 上,无需为旧 schema 建
一套并行标签。

blind-test 验收(路线图):已知类**重现率**(失败被归进 base-3 而非落 unknown 的占比)是及格线;
unknown 桶里翻出的**新聚簇**才是真发现。两个结果都如实报。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from nanoagent.runtime.tool_failure import classify_tool_failure, is_tool_failure

# 签名归一化:把可变部分抹平,只留可聚簇的结构。顺序有讲究(hex 先于纯数字)。
_NORM_SUBS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"https?://\S+"), "<url>"),
    (re.compile(r"\b[0-9a-f]{6,}\b", re.I), "<hex>"),
    (re.compile(r"\b\d{4}\.\d{4,5}\b"), "<arxiv>"),      # arxiv id
    (re.compile(r"[a-zA-Z]:\\[^\s]+|/[\w./-]{6,}"), "<path>"),
    (re.compile(r"\d+"), "N"),
    (re.compile(r"\s+"), " "),
]
_SIG_MAX = 160


def derive_op(tool: str, input_str: str) -> str:
    """从 (tool, input JSON) 还原操作身份;对齐 BaseTool.op 的粒度。

    skill_exec → `skill_exec:<skill>/<script>`;其余 → tool 名。解析失败回退 tool。
    """
    tool = (tool or "").strip() or "?"
    if tool == "skill_exec" and input_str:
        try:
            args = json.loads(input_str)
            skill = str(args.get("skill", "")).strip()
            script = str(args.get("script", "")).strip()
            if skill and script:
                return f"skill_exec:{skill}/{script}"
        except (json.JSONDecodeError, ValueError, AttributeError):
            pass
    return tool


def normalize_signature(output: str) -> str:
    """把失败 output 抹成可聚簇的归一化签名(去 url/hex/数字/路径,压空白)。"""
    text = (output or "").strip()
    for pat, repl in _NORM_SUBS:
        text = pat.sub(repl, text)
    return text.strip()[:_SIG_MAX]


@dataclass
class FailureObservation:
    op: str
    family: str          # base-3: schema_mismatch / transient / unknown
    signature: str       # 归一化签名
    trace: str           # 来源 trace 路径(算复现用)


def observe_events(events: Iterable[dict], trace: str) -> List[FailureObservation]:
    """从一条 trace 的事件流里抽工具失败观测。读旧/新两种工具事件。"""
    out: List[FailureObservation] = []
    for e in events:
        if e.get("action") not in ("tool_call_end", "tool_op"):
            continue
        output = e.get("output")
        if not isinstance(output, str) or not is_tool_failure(output):
            continue
        op = derive_op(e.get("tool", ""), e.get("input") or e.get("args") or "")
        out.append(FailureObservation(
            op=op,
            family=classify_tool_failure(output),
            signature=normalize_signature(output),
            trace=trace,
        ))
    return out


@dataclass
class Cluster:
    key: Tuple[str, str]           # (op, signature)
    family: str
    count: int = 0                 # 总出现次数
    traces: set = field(default_factory=set)  # 去重 trace(复现宽度)

    @property
    def recurrence(self) -> int:
        return len(self.traces)


def cluster(observations: Iterable[FailureObservation]) -> Dict[Tuple[str, str], Cluster]:
    clusters: Dict[Tuple[str, str], Cluster] = {}
    for o in observations:
        key = (o.op, o.signature)
        c = clusters.get(key)
        if c is None:
            c = clusters[key] = Cluster(key=key, family=o.family)
        c.count += 1
        c.traces.add(o.trace)
    return clusters


def summarize(clusters: Dict[Tuple[str, str], Cluster], total_traces: int,
              traces_with_failure: int) -> dict:
    """聚簇 → 报告 dict:家族分布 + 已知类重现率(blind-test 及格线)+ unknown 桶。"""
    from collections import Counter
    fam_obs = Counter()
    for c in clusters.values():
        fam_obs[c.family] += c.count
    total_obs = sum(fam_obs.values()) or 1
    known = total_obs - fam_obs.get("unknown", 0)
    return {
        "total_traces": total_traces,
        "traces_with_failure": traces_with_failure,
        "total_failure_observations": total_obs,
        "family_distribution": dict(fam_obs),
        "known_class_redetection_rate": round(known / total_obs, 4),  # 及格线指标
        "cluster_count": len(clusters),
        "unknown_cluster_count": sum(1 for c in clusters.values() if c.family == "unknown"),
    }


def top_clusters(clusters: Dict[Tuple[str, str], Cluster], family: Optional[str] = None,
                 n: int = 15) -> List[Cluster]:
    items = [c for c in clusters.values() if family is None or c.family == family]
    return sorted(items, key=lambda c: (-c.recurrence, -c.count))[:n]
