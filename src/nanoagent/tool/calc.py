"""strict baseline probe：纯函数四则运算 + 完全静态 schema。

存在意义不是产品功能（LLM 自己会算 1+1），而是**独立验证**
`BaseTool.strict_mode` 基础设施在真实 OpenAI tool 调用下生效。

为什么需要：
- SkillExecTool 想开 strict 但被 OpenAI API 拒（args 是 free-form object，
  与 strict 要求的 additionalProperties:false 不兼容）
- CalcTool 是反例 baseline——schema 完全静态、所有字段必填、无 free-form 子段，
  满足 strict 模式所有约束（all properties required + additionalProperties:false +
  无 pattern/format/minLength）
- 先用 CalcTool 验证 strict_mode 真生效，再回头看 SkillExecTool 撞墙是协议
  不兼容（virtual script tools 的工程动机），不是基础设施 bug

不走 skill_exec：CalcTool 是顶层 BaseTool，与任何 skill 无关。
"""

from __future__ import annotations

from typing import Final

from nanoagent.tool.base import BaseTool

_OPERATIONS: Final = {
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "multiply": lambda a, b: a * b,
    # divide 单独处理 zero 防 ZeroDivisionError 抛进 BaseTool.run 兜底
}


class CalcTool(BaseTool):
    """四则运算。`operation` enum + 两个 number，OpenAI strict 模式安全。"""

    strict_mode = True

    @property
    def name(self) -> str:
        return "calc"

    @property
    def description(self) -> str:
        return "执行基础四则运算（add / subtract / multiply / divide）。"

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["add", "subtract", "multiply", "divide"],
                    "description": "运算类型",
                },
                "a": {"type": "number", "description": "第一个数"},
                "b": {"type": "number", "description": "第二个数"},
            },
            "required": ["operation", "a", "b"],
            "additionalProperties": False,
        }

    def _execute(self, *, operation: str, a, b, **_kwargs) -> str:
        if operation == "divide":
            if b == 0:
                return "[CalcTool] divide-by-zero"
            return str(a / b)
        op = _OPERATIONS.get(operation)
        if op is None:
            return f"[CalcTool] 未知 operation={operation!r}"
        return str(op(a, b))
