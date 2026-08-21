"""按模型价格表计算 token 成本。"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import yaml

from infra.core.types import Cost


def load_pricing(path: Path) -> dict:
    """读取价格表。"""
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def region_for(pricing: dict, base_url: str | None) -> str:
    """按端点选择价格区域，未知端点回退默认区域。"""
    billing = pricing.get("billing") or {}
    host = urlparse(base_url).hostname if base_url else None
    return (billing.get("endpoint_region") or {}).get(host) or billing.get("default_region", "cn")


def cost_of(pricing: dict, model: str, region: str, usage: dict) -> Cost | None:
    """计算指定模型和区域的 token 成本；价格缺失时返回 None。"""
    entry = (pricing.get("models") or {}).get(model)
    price = ((entry or {}).get("prices") or {}).get(region)
    if not price:
        return None

    cached = usage.get("cached_input") or 0
    input_tokens = usage.get("input") or 0
    fresh_input = max(0, input_tokens - cached)
    cached_rate = price.get("cached_input_per_1m")
    if cached_rate is None:
        # 未配置缓存价时按普通输入价计费。
        fresh_input, cached = fresh_input + cached, 0
        cached_rate = 0.0

    long_context = price.get("long_context") or {}
    if input_tokens > long_context.get("threshold_input_tokens", float("inf")):
        input_multiplier = long_context.get("input_multiplier", 1.0)
        output_multiplier = long_context.get("output_multiplier", 1.0)
    else:
        input_multiplier = output_multiplier = 1.0

    amount = (fresh_input * price["input_per_1m"] * input_multiplier
              + cached * cached_rate * input_multiplier
              + (usage.get("output") or 0) * price["output_per_1m"] * output_multiplier) / 1_000_000
    return Cost(amount, price["currency"], price.get("confidence", "unknown"))


def total_by_currency(costs: list[Cost]) -> dict[str, float]:
    """按币种分别汇总成本。"""
    totals: dict[str, float] = {}
    for cost in costs:
        totals[cost.currency] = totals.get(cost.currency, 0.0) + cost.amount
    return dict(sorted(totals.items()))
