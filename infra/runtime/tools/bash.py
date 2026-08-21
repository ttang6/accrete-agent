"""bash 工具：在注入的 Environment 中执行命令。"""

from __future__ import annotations

from typing import Any

from infra.core.tools import Tool
from infra.core.types import ToolResult
from infra.runtime.environments.base import Environment, WorkspacePathError

DEFAULT_TIMEOUT_S = 60.0


class BashTool(Tool):
    """运行命令；退出码非零仍作为模型可纠正的任务信息返回。"""

    name = "bash"
    permission_group = "mutating"
    description = "在 workspace 内执行 shell 命令，用于搜索、构建和运行测试。"
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "要执行的 shell 命令"},
            "workdir": {"type": "string", "description": "相对 workspace 根目录的可选子目录"},
            "timeout_s": {"type": "number", "description": f"超时秒数，默认 {DEFAULT_TIMEOUT_S}"},
        },
        "required": ["command"],
        "additionalProperties": False,
    }

    def __init__(self, environment: Environment) -> None:
        self.environment = environment

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        command = arguments.get("command")
        if not isinstance(command, str) or not command.strip():
            return _error("参数 command 必须是非空字符串", "schema_error")
        workdir = arguments.get("workdir")
        if workdir is not None and (not isinstance(workdir, str) or not workdir.strip()):
            return _error("参数 workdir 必须是非空字符串", "schema_error")
        timeout_s = arguments.get("timeout_s", DEFAULT_TIMEOUT_S)
        if not isinstance(timeout_s, (int, float)) or isinstance(timeout_s, bool) or timeout_s <= 0:
            return _error("timeout_s 必须是正数", "schema_error")
        try:
            result = self.environment.execute(command, cwd=workdir, timeout_s=float(timeout_s))
        except WorkspacePathError as error:
            return _error(str(error), "permission")

        parts = [f"$ {command}"]
        if result.timed_out:
            parts.append(f"[命令超时（>{timeout_s}s），已终止]")
        if result.stdout:
            parts.append(f"--- stdout ---\n{result.stdout}")
        if result.stderr:
            parts.append(f"--- stderr ---\n{result.stderr}")
        parts.append(f"exit: {result.exit_code}")
        return ToolResult(
            tool_call_id="",
            content="\n".join(parts),
            is_error=result.timed_out or result.exit_code != 0,
            error_type="timeout" if result.timed_out else ("exec_error" if result.exit_code else None),
            attributes={"exit_code": result.exit_code, "timed_out": result.timed_out, "truncated": result.truncated},
        )


def _error(message: str, error_type: str) -> ToolResult:
    return ToolResult(tool_call_id="", content=message, is_error=True, error_type=error_type)
