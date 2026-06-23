"""OnlineReflector — 在线微反思：同 op 第 2 次失败时基于报错产一个修复假设。

docs/81 §5.4 在线/离线铁律的修正案（增补，不替换确定性地板）：
- 触发窄：同 op_key 第 2 次失败 + 飞轮无可召回的 suggested_action + 每 turn 上限
- 产出是假设不是经验——hint 标注语明示未验证，主 LLM 决定采纳与否，
  下一次工具调用即环境验证
- fail-open：副 LLM 任何异常 → None，行为与未装配时完全一致
- 不持有治理：不写 backend；假设被采纳且成功后由既有 episode/ingest 管道沉淀

装配开关在 main.py（ENABLE_ONLINE_REFLECTOR，默认关——eval 可比性 + 待小测验证）。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Final, Optional

from nanoagent.core.logger import get_logger
from nanoagent.core.prompt_assets import load_prompt
from nanoagent.runtime.context_sources import (
    MARKER_LEARNED_FIX,
    MARKER_LEARNED_HYPOTHESIS,
)

_logger = get_logger("online_reflector")

# hint 内 diagnosis 的硬上限，对齐 lesson_retriever 的 recommendation 截断策略
_HINT_MAX_DIAGNOSIS_CHARS: Final[int] = 200
# 喂给副 LLM 的现场材料截断（args / 报错原文）
_MAX_RAW_ARGS_CHARS: Final[int] = 500
_MAX_ERROR_OUTPUT_CHARS: Final[int] = 1200


PROPOSE_REPAIR_SCHEMA: Final[dict] = {
    "type": "function",
    "function": {
        "name": "propose_repair",
        "description": (
            "基于失败的工具调用与报错原文，提出一个修复假设。"
            "证据不足以定位时省略 suggested_args。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "diagnosis": {
                    "type": "string",
                    "description": "失败根因假设，≤120 字，必须引用报错原文中的具体证据",
                },
                "suggested_args": {
                    "type": "object",
                    "description": (
                        "按修复假设改好的完整调用参数（与给出的『调用参数』同形状、"
                        "整体替换）。只要 diagnosis 给出了具体假设就必须填；"
                        "仅当完全无线索时省略此字段"
                    ),
                },
            },
            # suggested_args 不进 required：union type ["object","null"] 在
            # dashscope function calling 上实测恒返 null，改为可省略字段
            "required": ["diagnosis"],
        },
    },
}


_REFLECTOR_SYSTEM_PROMPT: Final[str] = """你是工具失败诊断副助手。
给你一次失败的工具调用（工具、参数、报错原文），提出一个修复假设。

规则：
- 只基于报错原文里的证据推理（HTTP 状态码、URL、报错字段名等），不要编造接口文档
- 优先怀疑调用方参数而非服务端故障——4xx 几乎总是请求本身的问题
- suggested_args = 按假设改好的完整调用参数（与输入的『调用参数』同形状，整体替换）。
  锁定了可疑参数但不确定正确值时，首选直接删掉该可疑参数（让接口走默认值）
- 只有报错完全不足以定位时才省略 suggested_args，并在 diagnosis 说明缺什么信息
- 必须调用 propose_repair function，不要直接回复文字"""


def _parse_raw_args(raw_args: str) -> Optional[dict]:
    """原调用 args 的 JSON 解析；非法 / 非 dict → None（护栏比较自然不命中）。"""
    try:
        parsed = json.loads(raw_args or "")
    except (json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


@dataclass
class ReflectorSuggestion:
    """try_suggest 的返回结构。suggested_args=None 表示只有诊断没有可重试 args。"""

    diagnosis: str
    suggested_args: Optional[dict] = None


class OnlineReflector:
    """副 LLM 失败诊断器。stateless，每次 try_suggest 独立调用。

    用法（装配见 main.py build_loop）：
        reflector_llm = LLMClient(model="qwen3.6-flash", provider="dashscope", timeout=15)
        reflector = OnlineReflector(llm=reflector_llm)
    """

    def __init__(self, llm, system_prompt: Optional[str] = None):
        self._llm = llm
        self._system_prompt = system_prompt or load_prompt(
            "online_reflector", _REFLECTOR_SYSTEM_PROMPT
        )

    def try_suggest(
        self, tool_key: str, raw_args: str, error_output: str
    ) -> Optional[ReflectorSuggestion]:
        """对一次失败现场产一个修复假设。任何异常路径 → None（fail-open）。"""
        try:
            return self._suggest_impl(tool_key, raw_args, error_output)
        except Exception as e:
            _logger.warning(
                f"[online_reflector] suggest failed (fail-open): {type(e).__name__}: {e}"
            )
            return None

    def _suggest_impl(
        self, tool_key: str, raw_args: str, error_output: str
    ) -> Optional[ReflectorSuggestion]:
        user_content = (
            f"工具: {tool_key}\n"
            f"调用参数: {(raw_args or '')[:_MAX_RAW_ARGS_CHARS]}\n"
            f"报错原文:\n{(error_output or '')[:_MAX_ERROR_OUTPUT_CHARS]}"
        )
        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]
        response = self._llm.think_with_tools(
            messages=messages,
            tools=[PROPOSE_REPAIR_SCHEMA],
            temperature=0,
        )

        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls or tool_calls[0].function.name != "propose_repair":
            return None
        args = json.loads(tool_calls[0].function.arguments)
        diagnosis = str(args.get("diagnosis") or "").strip()
        suggested = args.get("suggested_args")
        if not isinstance(suggested, dict):
            suggested = None
        # 护栏：建议 args 与原 args 相同 → 丢弃建议只留诊断（同参重试正是
        # repeated_same_args_failure 要防的，不能由反思自己诱发）
        if suggested is not None and suggested == _parse_raw_args(raw_args):
            suggested = None
        if not diagnosis and suggested is None:
            return None
        return ReflectorSuggestion(diagnosis=diagnosis, suggested_args=suggested)

    def format_hint(self, suggestion: ReflectorSuggestion) -> str:
        """拼成可附加到 tool_result 末尾的 hint。与 [learned-lesson] 通道平行：

            [learned-fix]（修复假设，未经验证）{diagnosis}
            [learned-hypothesis]（建议的完整 args，重试一次即可验证）
            {json}
        """
        diagnosis = suggestion.diagnosis[:_HINT_MAX_DIAGNOSIS_CHARS]
        parts = [f"\n\n{MARKER_LEARNED_FIX}（修复假设，未经验证）{diagnosis}"]
        if suggestion.suggested_args is not None:
            try:
                args_json = json.dumps(
                    suggestion.suggested_args, ensure_ascii=False, indent=2
                )
            except (TypeError, ValueError):
                args_json = ""
            if args_json:
                parts.append(
                    f"\n{MARKER_LEARNED_HYPOTHESIS}"
                    "（建议的完整 args，重试一次即可验证）"
                    f"\n{args_json}"
                )
        return "".join(parts)
