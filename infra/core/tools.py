# tools.py
from __future__ import annotations

import json
from typing import Any, Literal, Protocol

from jsonschema import Draft202012Validator, SchemaError
from .types import EnvironmentFailure, ToolCall, ToolResult

PermissionGroup = Literal["read_only", "mutating", "network"]
_PERMISSION_GROUPS = frozenset({"read_only", "mutating", "network"})


class Tool(Protocol):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema
    permission_group: PermissionGroup

    def execute(self, arguments: dict[str, Any]) -> ToolResult: ...


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._validators: dict[str, Draft202012Validator] = {}
        self._disabled: set[str] = set()
        self._permission_groups: dict[str, PermissionGroup] = {}

    def register(self, tool: Tool) -> None:
        """注册工具，并在装配期拒绝无效的 JSON Schema。"""
        try:
            validator = Draft202012Validator(tool.parameters)
            validator.check_schema(tool.parameters)
        except SchemaError as error:
            raise ValueError(f"工具 {tool.name!r} 的 parameters 不是有效 JSON Schema: {error.message}") from error
        permission_group = getattr(tool, "permission_group", "mutating")
        if permission_group not in _PERMISSION_GROUPS:
            raise ValueError(f"工具 {tool.name!r} 的 permission_group 无效: {permission_group!r}")
        self._tools[tool.name] = tool
        self._validators[tool.name] = validator
        self._permission_groups[tool.name] = permission_group

    def disable(self, name: str) -> None:
        """禁用一个已注册工具，使后续模型调用和实际执行都不可用。"""
        self._require_registered(name)
        self._disabled.add(name)

    def enable(self, name: str) -> None:
        """重新启用一个已注册工具。"""
        self._require_registered(name)
        self._disabled.discard(name)

    def is_enabled(self, name: str) -> bool:
        """返回已注册工具当前是否可用。"""
        self._require_registered(name)
        return name not in self._disabled

    def permission_group(self, name: str) -> PermissionGroup:
        """返回已注册工具的权限组；未声明的外部工具保守视为 mutating。"""
        self._require_registered(name)
        return self._permission_groups[name]

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                },
            }
            for name, t in self._tools.items()
            if name not in self._disabled
        ]

    def dispatch(self, call: ToolCall) -> ToolResult:
        tool = self._tools.get(call.function.name)
        if tool is None:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Unknown tool: {call.function.name}",
                is_error=True,
                error_type="schema_error",
            )
        try:
            args = json.loads(call.function.arguments or "{}")
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Invalid JSON arguments: {e}",
                is_error=True,
                error_type="schema_error",
            )
        if call.function.name in self._disabled:
            return ToolResult(
                tool_call_id=call.id,
                content=f"Tool disabled: {call.function.name}",
                is_error=True,
                error_type="permission",
            )
        validation_error = next(self._validators[tool.name].iter_errors(args), None)
        if validation_error is not None:
            location = "/".join(str(part) for part in validation_error.absolute_path)
            where = f" at {location}" if location else ""
            return ToolResult(
                tool_call_id=call.id,
                content=f"Invalid arguments for tool {tool.name!r}{where}: {validation_error.message}",
                is_error=True,
                error_type="schema_error",
            )
        try:
            result = tool.execute(args)
            result.tool_call_id = call.id
            return result
        # 环境死亡不是工具错误。翻译成 ToolResult 就等于喂回模型让它对着尸体空转，
        # 必须排在兜底 except 之前。
        except EnvironmentFailure:
            raise
        except Exception as e:
            return ToolResult(
                tool_call_id=call.id,
                content=str(e),
                is_error=True,
                error_type="exec_error",
            )

    def _require_registered(self, name: str) -> None:
        """确认工具名已注册，避免状态操作静默拼错。"""
        if name not in self._tools:
            raise KeyError(f"未注册工具: {name}")
