"""read 工具：通过 Environment 读取 UTF-8 文件或列出目录。"""

from __future__ import annotations

from typing import Any

from infra.core.tools import Tool
from infra.core.types import ToolResult
from infra.runtime.environments.base import Environment, WorkspacePathError

DEFAULT_LIMIT = 400


class ReadTool(Tool):
    """按行读取文件，或列出目录的一层条目。"""

    name = "read"
    permission_group = "read_only"
    description = "读取 UTF-8 文件（带行号、可分页）或列出目录；路径相对 workspace。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "相对 workspace 的文件或目录路径"},
            "offset": {"type": "integer", "minimum": 0, "default": 0},
            "limit": {"type": "integer", "minimum": 1, "default": DEFAULT_LIMIT},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        path = arguments.get("path")
        offset = arguments.get("offset", 0)
        limit = arguments.get("limit", DEFAULT_LIMIT)
        if not isinstance(path, str) or not path.strip():
            return _error("参数 path 必须是相对 workspace 的非空字符串", "schema_error")
        if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
            return _error("offset 必须是非负整数", "schema_error")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            return _error("limit 必须是正整数", "schema_error")
        try:
            kind = self.environment.path_kind(path)
            if kind == "missing":
                return _error(f"文件或目录不存在: {path}", "not_found")
            if kind == "directory":
                entries = self.environment.list_dir(path)
                lines = [f"{'dir' if entry.kind == 'directory' else 'file'} {entry.path}" for entry in entries]
                return ToolResult(tool_call_id="", content=f"目录: {path}（{len(lines)} 项）\n" + "\n".join(lines))
            content = self.environment.read_file(path)
        except WorkspacePathError as error:
            return _error(str(error), "permission")
        except UnicodeDecodeError:
            return _error(f"文件不是合法 UTF-8，无法读取: {path}", "exec_error")
        except OSError as error:
            return _error(f"读取文件失败 {path}: {error}", "exec_error")

        lines = content.splitlines()
        end = min(offset + limit, len(lines))
        shown = [f"{line_no + 1}|{line}" for line_no, line in enumerate(lines[offset:end], start=offset)]
        tail = f"\n[已截断，还剩 {len(lines) - end} 行，用 offset={end} 继续]" if end < len(lines) else ""
        return ToolResult(
            tool_call_id="",
            content=f"文件: {path}\n共 {len(lines)} 行，显示 {offset + 1}-{end} 行\n" + "\n".join(shown) + tail,
        )


def _error(message: str, error_type: str) -> ToolResult:
    return ToolResult(tool_call_id="", content=message, is_error=True, error_type=error_type)
