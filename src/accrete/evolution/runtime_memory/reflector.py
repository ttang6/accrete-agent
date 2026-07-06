"""Reflector — 离线 Tier-2 失败复盘器：给一条失败轨迹，LLM 诊断 + 打包出可复用的
`suggested_action`。由 `evals/reflector_prototype.py` 产品化而来。

两条铁律：
1. **只在离线跑**（批/调度，非 inline 热路径）——见 `offline_mint.py`。
2. **在轨恢复验证门**（recovery-verified gate）：`reflect()` 只在 trace 里能看到
   "同一 op 失败后又被重新执行且成功"（= 可观测的恢复证据）时才反思——LLM 不发明
   修复，只**诊断 + 抽象**这次已观测到的恢复。没有这个硬 grounding 就别调
   `reflect()`（调用方先 `find_observed_recovery` 判定，None 即弃权）。

`reflect_soft()` 是给"信号缺失那半"的**软**通道（无观测恢复，best-effort 猜测）：
故意违反弃权，但产物**只**写物理旁路 jsonl、不进 backend、不被召回（见 offline_mint
的 consumption 闸）——安全性全靠那道硬闸，不靠这里弃权。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from accrete.core import trace_schema as ts
from accrete.core.logger import get_logger
from accrete.evolution.runtime_memory.schema import FailureEvent
from accrete.runtime.tool_failure import _is_tool_failure

_logger = get_logger("reflector")


@dataclass
class ReflectionResult:
    """Reflector 产物。grounded 模式必有 suggested_action；soft 模式可能 None。"""
    diagnosis: str
    recommendation: str
    suggested_action: Optional[Dict[str, Any]]
    grounded: bool  # True=基于观测恢复；False=无恢复的软猜测


# ============================================================
# 1) 在轨恢复验证：失败之后，同一 op 第一次成功的调用 = 观测到的恢复
# ============================================================


def _parse_tool_call(ev: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """tool_call_end 事件 → (op, args dict)。与 episode_extractor 同口径
    （skill_exec 用 skill/script 前缀，其它用裸 tool 名）。"""
    tool = ev.get("tool", "unknown")
    raw = ev.get("input", "") or ""
    try:
        parsed = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        parsed = {}
    if not isinstance(parsed, dict):
        parsed = {}
    if tool == "skill_exec":
        key = f"skill_exec:{parsed.get('skill', '?')}/{parsed.get('script', '?')}"
        args = parsed.get("args", {}) if isinstance(parsed.get("args"), dict) else {}
        return key, args
    return tool, parsed


def find_observed_recovery(
    events: List[Dict[str, Any]], failure: FailureEvent
) -> Optional[Dict[str, Any]]:
    """在失败之后，找同一 op 第一次**成功**的 tool_call_end，返回其 args dict。

    没有 → None（= 这次失败没被恢复，无 grounding，调用方应弃权产 grounded lesson）。
    这就是"在轨恢复验证门"：只认 trace 里实际重执行成功的硬证据，不靠 LLM 猜。
    """
    seen_failure = False
    for ev in events:
        if ev.get("action") != ts.ACTION_TOOL_CALL_END:
            continue
        key, args = _parse_tool_call(ev)
        if key != failure.op:
            continue
        out = ev.get("output", "") or ""
        if not seen_failure:
            # 等到越过那条失败本身（按 iteration 对齐）
            if ev.get("iteration", -1) >= failure.iteration and _is_tool_failure(out):
                seen_failure = True
            continue
        if not _is_tool_failure(out):
            return args  # 失败之后第一次成功 = 恢复
    return None


# ============================================================
# 2) Reflector：grounding prompt → LLM → suggested_action
# ============================================================

_REFLECT_INSTRUCTION = """你是一个离线的"失败复盘器"(Reflector)。给你一次 agent 运行里的【一个工具失败】\
和它后来【实际观测到的成功恢复调用】。基于**实际观测**（不是你的猜测）总结一条可复用教训，\
让 agent 下次遇到同类失败时能直接照做。

【失败】
- 工具: {op}
- 错误类型: {error_type}
- 错误信息: {error_message}
- 失败时参数: {failed_args}

【实际观测到的成功恢复】（同一工具，后续这次调用成功了）
- 恢复调用参数: {recovery_args}

只输出一个 JSON（不要 markdown 代码块），字段：
{{
  "diagnosis": "一句话：为什么失败、为什么这个恢复有效",
  "recommendation": "一句中文短建议，未来注入给 agent",
  "suggested_action": {{"skill": "...", "script": "...", "args": {{...}}}}
}}

规则：
- 只基于上面观测，不编造没发生的步骤。
- 若失败参数与恢复参数**相同** → 这是瞬时故障，recommendation 应说"直接原样重试，勿降级跳过"。
- suggested_action 必须是一个具体可照抄的正确调用（skill/script 来自 op，args 取恢复参数）。"""

_SOFT_INSTRUCTION = """你是一个离线的"失败复盘器"(Reflector)。给你一次 agent 运行里的【一个工具失败】，\
但这次**没有观测到后续成功的恢复**。请基于失败信息给一条 best-effort 的猜测性建议——\
明确这是未经验证的猜测。

【失败】
- 工具: {op}
- 错误类型: {error_type}
- 错误信息: {error_message}
- 失败时参数: {failed_args}

只输出一个 JSON（不要 markdown 代码块），字段：
{{
  "diagnosis": "一句话：可能的失败原因（标注为猜测）",
  "recommendation": "一句中文短建议（未经验证）"
}}

规则：只给一条简短建议，不要编造 suggested_action（没有观测恢复可照抄）。"""


def _strip_json(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
        if t.startswith("json"):
            t = t[4:]
    return t.strip()


def _failed_args(failure: FailureEvent) -> str:
    return json.dumps((failure.extras or {}).get("raw_input", ""), ensure_ascii=False)


class Reflector:
    """离线失败复盘器。注入 LLM（需 `.think(messages)`），便于 mock 单测。"""

    def __init__(self, llm):
        self._llm = llm

    def reflect(
        self, failure: FailureEvent, recovery_args: Dict[str, Any]
    ) -> Optional[ReflectionResult]:
        """grounded 反思：基于观测到的恢复 → {diagnosis, recommendation, suggested_action}。

        只在 `find_observed_recovery` 返回非 None（有硬 grounding）时调用。
        LLM 输出解析失败 → None。
        """
        prompt = _REFLECT_INSTRUCTION.format(
            op=failure.op,
            error_type=failure.error_type,
            error_message=failure.error_message[:200],
            failed_args=_failed_args(failure),
            recovery_args=json.dumps(recovery_args, ensure_ascii=False),
        )
        parsed = self._call_and_parse(prompt)
        if parsed is None:
            return None
        sa = parsed.get("suggested_action")
        return ReflectionResult(
            diagnosis=str(parsed.get("diagnosis", "")),
            recommendation=str(parsed.get("recommendation", "")),
            suggested_action=sa if isinstance(sa, dict) else None,
            grounded=True,
        )

    def reflect_soft(self, failure: FailureEvent) -> Optional[ReflectionResult]:
        """软（无 grounding）反思：无观测恢复时的 best-effort 猜测。产物只能进
        物理旁路 jsonl（见 offline_mint），不进 backend、不被召回。suggested_action 恒 None。"""
        prompt = _SOFT_INSTRUCTION.format(
            op=failure.op,
            error_type=failure.error_type,
            error_message=failure.error_message[:200],
            failed_args=_failed_args(failure),
        )
        parsed = self._call_and_parse(prompt)
        if parsed is None:
            return None
        return ReflectionResult(
            diagnosis=str(parsed.get("diagnosis", "")),
            recommendation=str(parsed.get("recommendation", "")),
            suggested_action=None,
            grounded=False,
        )

    def _call_and_parse(self, prompt: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._llm.think([{"role": "user", "content": prompt}])
        except Exception as e:
            _logger.warning(f"[reflector] LLM 调用异常: {e}")
            return None
        try:
            parsed = json.loads(_strip_json(raw or ""))
        except (json.JSONDecodeError, ValueError):
            _logger.warning(f"[reflector] JSON 解析失败，原文片段: {(raw or '')[:200]}")
            return None
        return parsed if isinstance(parsed, dict) else None
