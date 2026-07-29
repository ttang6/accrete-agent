"""工具注册、Schema 子集校验与分发。"""

from copy import deepcopy

from .tool import AccessType, BaseTool
from .types import ToolCall, ToolResult


class ToolRegistry:
    """维护工具实例并在分发前执行确定性校验。"""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """注册工具；重名属于组装期错误。"""
        if not tool.name or tool.name in self._tools:
            raise ValueError(f"工具重复注册或名称为空: {tool.name!r}")
        if not isinstance(tool.access_type, AccessType):
            raise TypeError("工具 access_type 必须是 AccessType 枚举值")
        self._tools[tool.name] = tool

    def to_openai_schemas(self) -> list[dict]:
        """导出 OpenAI function calling 所需的规整 Schema。"""
        result = []
        for tool in self._tools.values():
            parameters = deepcopy(tool.parameters)
            parameters.setdefault("type", "object")
            parameters.setdefault("properties", {})
            result.append({"type": "function", "function": {
                "name": tool.name, "description": tool.description, "parameters": parameters,
                "strict_compatible": tool.strict_compatible,
            }})
        return result

    def dispatch(self, call: ToolCall) -> ToolResult:
        """校验并执行一次工具调用。"""
        tool = self._tools.get(call.name)
        if tool is None:
            return ToolResult(False, "", f"未注册工具: {call.name}", "schema_error")
        error = self._validate(tool.parameters, call.arguments)
        if error:
            return ToolResult(False, "", error, "schema_error")
        return tool.execute(call.arguments)

    @staticmethod
    def _validate(schema: dict, value: object, path: str = "arguments") -> str | None:
        """校验 P0 JSON Schema 子集并返回第一条错误。"""
        expected = schema.get("type")
        types = expected if isinstance(expected, list) else [expected] if expected else []
        if types and not any(ToolRegistry._matches_type(value, item) for item in types):
            return f"{path} 期望类型 {types}，实际收到 {type(value).__name__}"
        if "enum" in schema and value not in schema["enum"]:
            return f"{path} 期望为 {schema['enum']} 之一，实际收到 {value!r}"
        if isinstance(value, dict):
            for key in schema.get("required", []):
                if key not in value:
                    return f"{path}.{key} 是必填字段，但实际未提供"
            properties = schema.get("properties", {})
            if schema.get("additionalProperties") is False:
                extra = set(value) - set(properties)
                if extra:
                    return f"{path} 不允许额外字段，实际收到 {sorted(extra)}"
            for key, child in properties.items():
                if key in value:
                    error = ToolRegistry._validate(child, value[key], f"{path}.{key}")
                    if error:
                        return error
        return None

    @staticmethod
    def _matches_type(value: object, expected: str) -> bool:
        return {
            "object": lambda: isinstance(value, dict), "array": lambda: isinstance(value, list),
            "string": lambda: isinstance(value, str), "integer": lambda: isinstance(value, int) and not isinstance(value, bool),
            "number": lambda: isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": lambda: isinstance(value, bool), "null": lambda: value is None,
        }.get(expected, lambda: True)()
