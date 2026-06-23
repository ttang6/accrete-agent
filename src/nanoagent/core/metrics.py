"""MetricsRegistry —— 进程内薄指标聚合。

可观测三柱里 trace 早有（RunTracer，为离线学习特化），log 也有（零散分级行），
唯独 metric 此前完全没有——算"近 1h 工具失败率"得去扫一堆 per-run trace 文件，
那是拿错了载体。这里补上缺得最彻底的这根：一个内存里累加的计数器/直方图，turn
结束落一行快照。

三条纪律贯穿全文：
- **轻档零依赖**：不引 Prometheus/OTel SDK，dict 累加 + JSONL 快照，单进程作品够用；
  但指标命名对齐 OTel GenAI 语义（tool/model/status/latency），将来换 sink 即可接。
- **各 sink 自持存储，绝不读学习 trace 文件**：metric 与 trace 生命周期冲突（一个要
  聚合丢明细、一个要长留给飞轮），合并会互相拖累，所以物理分目录、各自演进。
- **fire-and-forget**：所有写入 try/except 包住，绝不抛进主循环；`ENABLE_OBSERVABILITY`
  关掉则整体 no-op，向后兼容。
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from typing import Any, Dict, Tuple

from nanoagent.core.paths import data_dir

_ENABLED_ENV = "ENABLE_OBSERVABILITY"

# histogram 固定分桶（上界，ms / token 数）：预聚合丢明细，只留落桶计数 + count/sum。
_LATENCY_BUCKETS_MS: Tuple[float, ...] = (50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000)
_TOKEN_BUCKETS: Tuple[float, ...] = (500, 1000, 2500, 5000, 10000, 20000, 50000, 100000)

_LabelKey = Tuple[str, Tuple[Tuple[str, str], ...]]


def observability_enabled() -> bool:
    """env 开关，默认开；关掉则 log_event / metrics 全 no-op。"""
    return os.getenv(_ENABLED_ENV, "1").strip().lower() not in ("0", "false", "off", "no")


def _key(name: str, labels: Dict[str, Any]) -> _LabelKey:
    # label 值统一转 str 并排序，保证同一组 labels 命中同一序列（与顺序无关）。
    return name, tuple(sorted((str(k), str(v)) for k, v in labels.items()))


def _buckets_for(name: str) -> Tuple[float, ...]:
    return _TOKEN_BUCKETS if "token" in name else _LATENCY_BUCKETS_MS


class MetricsRegistry:
    """counter / histogram / gauge 三类指标的内存聚合。线程安全、fire-and-forget。"""

    def __init__(self) -> None:
        self._counters: Dict[_LabelKey, float] = {}
        self._hist: Dict[_LabelKey, Dict[str, Any]] = {}
        self._gauges: Dict[_LabelKey, float] = {}
        self._lock = threading.Lock()

    def incr(self, name: str, value: float = 1, **labels: Any) -> None:
        if not observability_enabled():
            return
        try:
            k = _key(name, labels)
            with self._lock:
                self._counters[k] = self._counters.get(k, 0) + value
        except Exception:
            pass

    def observe(self, name: str, value: float, **labels: Any) -> None:
        if not observability_enabled():
            return
        try:
            k = _key(name, labels)
            bounds = _buckets_for(name)
            with self._lock:
                h = self._hist.get(k)
                if h is None:
                    h = {"count": 0, "sum": 0.0, "buckets": [0] * (len(bounds) + 1)}
                    self._hist[k] = h
                h["count"] += 1
                h["sum"] += value
                # 落第一个 value<=上界的桶；都不命中归最后的 +Inf 桶
                idx = next((i for i, b in enumerate(bounds) if value <= b), len(bounds))
                h["buckets"][idx] += 1
        except Exception:
            pass

    def gauge(self, name: str, value: float, **labels: Any) -> None:
        if not observability_enabled():
            return
        try:
            with self._lock:
                self._gauges[_key(name, labels)] = value
        except Exception:
            pass

    def snapshot(self) -> Dict[str, Any]:
        """当前累计值的可序列化快照（cumulative-since-start）。"""
        def _rows(store: Dict[_LabelKey, Any]):
            out = []
            for (name, labels), value in store.items():
                out.append({"name": name, "labels": dict(labels), "value": value})
            return out

        with self._lock:
            return {
                "counters": _rows(self._counters),
                "histograms": _rows(self._hist),
                "gauges": _rows(self._gauges),
            }

    def dump(self) -> None:
        """turn 结束时把快照追加一行到 data/logs/metrics/<date>.jsonl。fire-and-forget。"""
        if not observability_enabled():
            return
        try:
            snap = self.snapshot()
            snap["ts"] = datetime.now().isoformat()
            path = data_dir("logs/metrics") / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(snap, ensure_ascii=False) + "\n")
        except Exception:
            pass


# 模块级单例：装配层与各 emit 点直接 import 它，免去层层穿线。
metrics = MetricsRegistry()
