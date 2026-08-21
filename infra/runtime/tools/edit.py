"""edit 工具：通过 Environment 做精确字符串替换。"""

from __future__ import annotations

from typing import Any

from infra.core.tools import Tool
from infra.core.types import ToolResult
from infra.runtime.environments.base import Environment, WorkspacePathError


class EditTool(Tool):
    """只在匹配唯一时替换，避免模型在错误位置盲改。"""

    name = "edit"
    permission_group = "mutating"
    description = "在已有 UTF-8 文件中精确替换字符串；默认要求 old_string 只匹配一次。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 workspace 的目标文件路径"},
            "old_string": {"type": "string", "description": "必须逐字符匹配的原文本"},
            "new_string": {"type": "string", "description": "替换文本；空串表示删除"},
            "replace_all": {"type": "boolean", "default": False},
        },
        "required": ["path", "old_string", "new_string"],
        "additionalProperties": False,
    }

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        old_string = arguments.get("old_string")
        new_string = arguments.get("new_string")
        replace_all = arguments.get("replace_all", False)
        if not isinstance(path, str) or not path.strip():
            return _error("参数 path 必须是相对 workspace 的非空字符串", "schema_error")
        if not isinstance(old_string, str) or not old_string:
            return _error("old_string 必须是非空字符串", "schema_error")
        if not isinstance(new_string, str) or not isinstance(replace_all, bool):
            return _error("new_string 必须是字符串，replace_all 必须是布尔值", "schema_error")
        try:
            if self.environment.path_kind(path) != "file":
                return _error(f"文件不存在: {path}", "not_found")
            content = self.environment.read_file(path)
        except WorkspacePathError as error:
            return _error(str(error), "permission")
        except (OSError, UnicodeDecodeError) as error:
            return _error(f"读取失败 {path}: {error}", "exec_error")

        count = content.count(old_string)
        if count == 0:
            return _error("未找到 old_string 的精确匹配；请先 read 核对内容。", "exec_error")
        if count > 1 and not replace_all:
            return _error(f"old_string 匹配了 {count} 处；请扩大上下文或设 replace_all=true。", "exec_error")
        try:
            self.environment.write_file(path, content.replace(old_string, new_string))
        except OSError as error:
            return _error(f"写入失败 {path}: {error}", "exec_error")
        replaced = count if replace_all else 1
        return ToolResult(tool_call_id="", content=f"已对 {path} 完成 {replaced} 处替换\n{_summary(old_string, new_string)}")


def _summary(old: str, new: str) -> str:
    """返回紧凑的替换摘要，供模型在下一步核对。"""
    return f"  - {_single_line(old)}\n  + {_single_line(new)}"


def _single_line(text: str, limit: int = 60) -> str:
    """把多行字符串压成不刷屏的一行。"""
    flattened = text.replace("\n", "\\n")
    return flattened if len(flattened) <= limit else flattened[:limit] + "…"


def _error(message: str, error_type: str) -> ToolResult:
    return ToolResult(tool_call_id="", content=message, is_error=True, error_type=error_type)
