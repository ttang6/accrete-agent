"""Workspace tools —— M0 CI/onboarding 域的三个最小工具。

`run_command` / `read_file` / `write_file`，都**圈禁在一个 workspace_root**（= clone 出来的
仓库根）里：路径解析后必须仍落在 root 之下，`..` 穿越 / 绝对路径一律拒。这与 glob/grep 的
"锁 cwd"不同——onboarding runner 每个仓库构造一套工具、把 root 设成该仓库目录。

圈禁强度说明（M0 起步档，见 `docs/_M0_*` §5）：这是**subprocess + cwd 圈禁 + 超时**级别,
不是真沙箱——`run_command` 走 shell,理论上仍能 `cd ..` 逃逸。起步只跑热门大库(风险有限),
上量再套 WSL2 / 容器。路径类工具(read/write)的圈禁是硬的(resolve 后 relative_to 校验)。

错误约定:
- 工具级失败(超时 / 启动子进程失败 / 路径越界 / 文件不存在)→ `[<tool> 错误] ...` 前缀,
  被热路径失败签名识别、走熔断/召回链。
- **命令非零退出不是工具错误,是数据**(pytest 挂了是合法结果,agent 要看到 exit code
  才知道测试过没过)→ 结构化返回 exit code + stdout + stderr,不带错误前缀。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from nanoagent.tool.base import BaseTool, FailureKlass, ToolFailure

# 单次返回上限,防一次 pytest / 大文件把整棵输出灌进上下文。
_MAX_OUTPUT_CHARS = 5000
# run_command 默认超时(秒)。装依赖可能慢,给宽一点;挂死由超时兜底。
_DEFAULT_TIMEOUT_S = 180


def _resolve_in_workspace(root: Path, rel: str) -> Optional[Path]:
    """把相对路径解析到 workspace 内的绝对路径;越界 / 绝对路径 / 空 → None。

    Args:
        root: 已 resolve 的 workspace 根。
        rel: 相对 root 的路径(不接受绝对路径)。

    Returns:
        落在 root 之下的绝对 Path;非法则 None。
    """
    if not isinstance(rel, str) or not rel.strip():
        return None
    if Path(rel).is_absolute():
        return None  # 绝对路径直接拒,不给逃逸口
    try:
        resolved = (root / rel).resolve()
    except (OSError, ValueError, RuntimeError):
        return None
    try:
        resolved.relative_to(root)
    except ValueError:
        return None  # resolve 后逃出了 root（.. 穿越 / 软链）
    return resolved


def _truncate(text: str, limit: int = _MAX_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（共 {len(text)} 字,已截断到前 {limit}）"


class RunCommandTool(BaseTool):
    """在 workspace 根里跑一条 shell 命令,带超时。

    非零退出**不算工具错误**——结构化返回 exit code + stdout + stderr,让 agent 自己判
    测试过没过。只有超时 / 启动失败才 `[run_command 错误]`。
    """

    def __init__(
        self,
        workspace_root: Path,
        *,
        timeout_seconds: int = _DEFAULT_TIMEOUT_S,
        max_output_chars: int = _MAX_OUTPUT_CHARS,
    ):
        self._root = Path(workspace_root).resolve()
        self._timeout = timeout_seconds
        self._max_output = max_output_chars

    @property
    def name(self) -> str:
        return "run_command"

    @property
    def description(self) -> str:
        return (
            "在当前仓库工作目录里执行一条 shell 命令（如 `pip install -e .`、"
            "`python -m pytest -q`），带超时。返回退出码 + stdout + stderr。"
            "退出码非 0 是正常结果（如测试失败），不是工具错误。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "要执行的 shell 命令,在仓库根目录下运行",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    def validate(self, **kwargs) -> Optional[str]:
        command = kwargs.get("command")
        if not isinstance(command, str) or not command.strip():
            return "command 必填且非空"
        return None

    def _execute(self, *, command: str, **_kwargs) -> str:
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            return f"[run_command 错误] 命令超时（{self._timeout}s）: {command}"
        except (OSError, ValueError) as e:
            return f"[run_command 错误] 启动子进程失败: {type(e).__name__}: {e}"

        stdout = _truncate(result.stdout or "", self._max_output)
        stderr = _truncate(result.stderr or "", self._max_output)
        return (
            f"exit_code: {result.returncode}\n"
            f"--- stdout ---\n{stdout or '(空)'}\n"
            f"--- stderr ---\n{stderr or '(空)'}"
        )

    def classify_failure(
        self, kwargs: dict, output: str, exc: Optional[BaseException] = None
    ) -> ToolFailure:
        """超时归 transient（值得重试）,其余工具级失败说不清（klass=None）。"""
        klass: Optional[FailureKlass] = None
        if isinstance(output, str) and "命令超时" in output:
            klass = "transient"
        return ToolFailure(call_key=self.call_key(kwargs), klass=klass)


class ReadFileTool(BaseTool):
    """读 workspace 内一个文本文件（相对路径,圈禁在仓库根之下）。"""

    strict_mode = True

    def __init__(self, workspace_root: Path, *, max_output_chars: int = _MAX_OUTPUT_CHARS):
        self._root = Path(workspace_root).resolve()
        self._max_output = max_output_chars

    @property
    def name(self) -> str:
        return "read_file"

    @property
    def description(self) -> str:
        return (
            "读取仓库工作目录下一个文本文件的内容（传相对路径,如 "
            "`pyproject.toml` 或 `src/pkg/__init__.py`）。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对仓库根的文件路径",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    def _execute(self, *, path: str, **_kwargs) -> str:
        target = _resolve_in_workspace(self._root, path)
        if target is None:
            return f"[read_file 错误] 路径越界或非法: {path}"
        if not target.exists():
            return f"[read_file 错误] 文件不存在: {path}"
        if not target.is_file():
            return f"[read_file 错误] 不是文件: {path}"
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"[read_file 错误] 读取失败: {type(e).__name__}: {e}"
        return _truncate(text, self._max_output)


class WriteFileTool(BaseTool):
    """写 workspace 内一个文本文件（相对路径,圈禁;自动建父目录;覆盖写）。"""

    strict_mode = True
    is_mutating = True  # 写副作用 → 熔断器 fail-fast,不自动重试

    def __init__(self, workspace_root: Path):
        self._root = Path(workspace_root).resolve()

    @property
    def name(self) -> str:
        return "write_file"

    @property
    def description(self) -> str:
        return (
            "把内容写入仓库工作目录下一个文件（传相对路径,覆盖写,自动建父目录）。"
            "用于打补丁 / 建配置文件。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "相对仓库根的文件路径",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的完整文件内容",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    def _execute(self, *, path: str, content: str, **_kwargs) -> str:
        target = _resolve_in_workspace(self._root, path)
        if target is None:
            return f"[write_file 错误] 路径越界或非法: {path}"
        if target.is_dir():
            return f"[write_file 错误] 目标是目录: {path}"
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content if isinstance(content, str) else str(content),
                              encoding="utf-8")
        except OSError as e:
            return f"[write_file 错误] 写入失败: {type(e).__name__}: {e}"
        return f"已写入 {path}（{len(content)} 字）"
