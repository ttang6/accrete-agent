"""write 工具：通过 Environment 新建或覆盖 UTF-8 文件。"""

from __future__ import annotations

from typing import Any

from infra.core.tools import Tool
from infra.core.types import ToolResult
from infra.runtime.environments.base import Environment, WorkspacePathError


class WriteTool(Tool):
    """整文件写入；已有文件的局部修改应优先使用 edit。"""

    name = "write"
    permission_group = "mutating"
    description = "新建或整文件覆盖写入 UTF-8 内容；路径相对 workspace。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 workspace 的目标文件路径"},
            "content": {"type": "string", "description": "完整文件内容"},
        },
        "required": ["path", "content"],
        "additionalProperties": False,
    }

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        content = arguments.get("content")
        if not isinstance(path, str) or not path.strip():
            return _error("参数 path 必须是相对 workspace 的非空字符串", "schema_error")
        if not isinstance(content, str):
            return _error("参数 content 必须是字符串", "schema_error")
        try:
            if self.environment.path_kind(path) == "directory":
                return _error(f"目标是目录，不能写入: {path}", "exec_error")
            size = self.environment.write_file(path, content)
        except WorkspacePathError as error:
            return _error(str(error), "permission")
        except OSError as error:
            return _error(f"写入失败 {path}: {error}", "exec_error")
        return ToolResult(tool_call_id="", content=f"已写入 {path}（{size} 字节）")


def _error(message: str, error_type: str) -> ToolResult:
    return ToolResult(tool_call_id="", content=message, is_error=True, error_type=error_type)
