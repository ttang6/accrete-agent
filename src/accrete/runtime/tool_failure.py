"""tool_failure — tool-output 失败签名 / 分类 / args_hash 的单一来源（SSoT）。

之前 `runtime/failure_memory.py` 与 `evolution/runtime_memory/episode_extractor.py`
各自维护一份 `_FAILURE_SIGNATURES` / `_*_PATTERNS` / `_is_tool_failure` /
`_canonical_args_hash`，靠"注释约定"同步。真跑已经踩过 drift bug——dup_check
schema_mismatch 在 RepairGate 拦下后，episode_extractor 因
`_FAILURE_SIGNATURES` 没同步加 `[tool-call-repair-required]` 而忽略该事件，
LessonIngestor 不入库 → 飞轮断链。

实测发现的真实 drift：
- `failure_memory._SCHEMA_MISMATCH_PATTERNS` 含 6 项（多 `"action"` 处理空字符串场景）
- `episode_extractor._SCHEMA_MISMATCH_HINTS` 只有 5 项
- 命名也不同（`_PATTERNS` vs `_HINTS`）

本模块统一所有常量与函数。两个旧模块通过 `from tool_failure import ...` re-export
保持向后兼容（`from accrete.runtime.failure_memory import _is_tool_failure` 等
import 路径**不变**），但物理上共享同一对象（`is` 相等）。

不在范围：
- `_SUGGESTED_NEXT_ACTION` —— 仅 `failure_memory.maybe_augment` 用，不是重复，留在原地
- `_failure_key` / `_build_tool_key` —— 两边 schema 不同（tuple vs str），保持原样

API 公开：
- `is_tool_failure(output) -> bool`
- `classify_tool_failure(output) -> Literal["schema_mismatch", "transient", "unknown"]`
- `canonical_args_hash(raw_args) -> str`

私有别名（向后兼容旧 import 路径）：
- `_FAILURE_SIGNATURES` / `_TRANSIENT_PATTERNS` / `_SCHEMA_MISMATCH_PATTERNS`
- `_is_tool_failure` / `_classify_tool_failure` / `_canonical_args_hash`
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Final, Tuple

from accrete.runtime.context_sources import MARKER_GATE_INVALID_CALL


# ============================================================
# 失败签名常量（SSoT）
# ============================================================

# tool-output 失败签名：任一前缀命中即视为 failure。
# `[gate-invalid-call]`（MARKER_GATE_INVALID_CALL，原 [tool-call-repair-required]）
# 是 RepairGate 的输出标记 —— 也归为 failure，让现有 lesson 召回 + 重复失败 + 3rd 次
# stop_condition 保护链对 RepairGate 拦下的协议级错误同样生效。这里 import 同一常量
# （B∩C 共读单一来源），避免历史上 emit/match 不同步致飞轮断链的 drift。
# OutcomeTracker 通过独立的 ACTION_TOOL_CALL_REPAIR_REQUIRED trace 事件
# 把它从 helped/hurt 计数中剥离出来（判 ineffective_application）。
FAILURE_SIGNATURES: Final[Tuple[str, ...]] = (
    "[skill_exec 错误]",
    "[参数错误]",
    "[执行错误]",
    "[error]",
    MARKER_GATE_INVALID_CALL,
    "错误:",
    "退出码",
    "Traceback",
    "Exception",
)

# schema_mismatch 关键词。`"action"` 处理 "action ''" 等空字符串场景
# （之前 episode_extractor 漏了这项 → 真实 drift）。
SCHEMA_MISMATCH_PATTERNS: Final[Tuple[str, ...]] = (
    "未知 action",
    "action",
    "required",
    "missing",
    "参数",
    "[参数错误]",
)

TRANSIENT_PATTERNS: Final[Tuple[str, ...]] = (
    "timeout",
    "超时",
    "429",
    "rate limit",
    "5xx",
    "500",
    "502",
    "503",
    "504",
    "server error",
)


# ============================================================
# 公开函数
# ============================================================


def is_tool_failure(output: str) -> bool:
    """基于 tool 输出前缀/关键词判定是否失败。

    扫描首 200 字符（lower-case）找任一签名命中。
    """
    head = output[:200].lower() if output else ""
    for sig in FAILURE_SIGNATURES:
        if sig.lower() in head:
            return True
    return False


def classify_tool_failure(output: str) -> str:
    """把失败 tool_output 分类为 schema_mismatch / transient / unknown。

    transient 优先级高于 schema_mismatch（"timeout" 含 "i" 可能误匹配未来
    schema 关键词，先排 transient 防误判）。
    """
    msg = (output or "").lower()
    if any(p.lower() in msg for p in TRANSIENT_PATTERNS):
        return "transient"
    if any(p.lower() in msg for p in SCHEMA_MISMATCH_PATTERNS):
        return "schema_mismatch"
    return "unknown"


def canonical_args_hash(raw_args: str) -> str:
    """对 raw_args（JSON 字符串）做 canonical hash。

    先尝试 json parse → sort_keys dump → sha256[:12]。parse 失败 fallback
    到 raw 字符串 strip 后直接 hash——保证 whitespace 微变不影响 hash。
    跨进程稳定：同样输入产同样 hash。
    """
    try:
        parsed: Any = json.loads(raw_args) if raw_args else {}
    except json.JSONDecodeError:
        parsed = (raw_args or "").strip()
    canonical = json.dumps(
        parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# ============================================================
# 私有别名（向后兼容旧 import 路径，物理上同一对象）
# ============================================================

_FAILURE_SIGNATURES = FAILURE_SIGNATURES
_TRANSIENT_PATTERNS = TRANSIENT_PATTERNS
_SCHEMA_MISMATCH_PATTERNS = SCHEMA_MISMATCH_PATTERNS
_is_tool_failure = is_tool_failure
_classify_tool_failure = classify_tool_failure
_canonical_args_hash = canonical_args_hash
