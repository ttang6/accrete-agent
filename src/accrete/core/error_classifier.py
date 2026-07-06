"""错误分类器：把 exception 映射成短标签 + transient 标志位。

骨架位预留。evolution ReflexionGate 用这个字段做 pre-filter
（transient 错误视为噪音跳过，permanent 错误才进 reflexion 候选池）。

约定：
  - error_class: 短标签（如 "network_timeout" / "invalid_input"）
  - is_transient: True = 重试可能成功；False = 结构性错误

precedence（优先级，从高到低）：
  1. 消息里带 "rate limit" / "429" 等 → rate_limit transient
  2. 消息里带 "5xx" / "HTTPError: 5" / "server error" → http_5xx transient
  3. exception 类型名直接在 ERROR_CLASS_LABELS 表 → 按表
  4. 走 MRO 链找父类匹配
  5. fallback ("unknown", False)
"""

from typing import Final

# type_name -> (label, is_transient)
ERROR_CLASS_LABELS: Final[dict[str, tuple[str, bool]]] = {
    # Transient: 网络 / 服务
    "TimeoutError": ("network_timeout", True),
    "ConnectionError": ("network_error", True),
    "ConnectionResetError": ("network_reset", True),
    "ConnectionRefusedError": ("network_refused", True),
    "ConnectionAbortedError": ("network_aborted", True),
    # Permanent: 用户输入 / 结构性问题
    "ValueError": ("invalid_input", False),
    "KeyError": ("missing_key", False),
    "FileNotFoundError": ("not_found", False),
    "PermissionError": ("permission_denied", False),
    "TypeError": ("type_error", False),
    "AttributeError": ("attribute_error", False),
    "NotImplementedError": ("not_implemented", False),
}

_RATE_LIMIT_PATTERNS: Final[tuple[str, ...]] = (
    "429",
    "rate limit",
    "quota exceeded",
    "too many requests",
)
_HTTP_5XX_PATTERNS: Final[tuple[str, ...]] = (
    "500",
    "502",
    "503",
    "504",
    "server error",
)


def classify_error(exc: BaseException) -> tuple[str, bool]:
    """Classify an exception into (error_class_label, is_transient)。

    未识别 → ("unknown", False) fallback。
    """
    type_name = type(exc).__name__
    msg = str(exc).lower()

    # 1. rate limit 消息特征
    if any(p in msg for p in _RATE_LIMIT_PATTERNS):
        return ("rate_limit", True)

    # 2. HTTP 5xx 消息特征
    if any(p in msg for p in _HTTP_5XX_PATTERNS):
        return ("http_5xx", True)

    # 3. 类型名直接查表
    if type_name in ERROR_CLASS_LABELS:
        return ERROR_CLASS_LABELS[type_name]

    # 4. 走 MRO 链找父类匹配（Exception / BaseException 自己不在表里，自然走到 5）
    for base_cls in type(exc).__mro__:
        base_name = base_cls.__name__
        if base_name in ERROR_CLASS_LABELS:
            return ERROR_CLASS_LABELS[base_name]

    return ("unknown", False)
