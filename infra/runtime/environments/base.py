"""环境契约：工具触达命令和文件系统的唯一边界。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


PathKind = Literal["file", "directory", "missing"]


class WorkspacePathError(ValueError):
    """工具请求的相对路径或工作目录越出 workspace。"""


@dataclass(frozen=True)
class DirectoryEntry:
    """工作目录内单层条目。"""

    path: str
    kind: Literal["file", "directory"]


@dataclass
class CommandResult:
    """一次命令执行的完整结果。

    非零退出码是任务信息。只有环境不可用时，实现才抛出
    ``EnvironmentFailure``。输出截断由环境完成，所有工具看到同一份证据。
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False
    truncated: bool = False
    extra: dict = field(default_factory=dict)


class Environment(ABC):
    """命令和文件操作的后端接口；所有路径均相对 workspace 根目录。"""

    @abstractmethod
    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        """执行命令并返回输出；超时返回 ``timed_out=True``。"""

    @abstractmethod
    def path_kind(self, path: str) -> PathKind:
        """返回 workspace 内 path 的类型；越界路径抛出 ``WorkspacePathError``。"""

    @abstractmethod
    def read_file(self, path: str) -> str:
        """读取 workspace 内的 UTF-8 文本文件。"""

    @abstractmethod
    def write_file(self, path: str, content: str) -> int:
        """覆盖写入 UTF-8 文本并返回写入字节数。"""

    @abstractmethod
    def list_dir(self, path: str) -> list[DirectoryEntry]:
        """列出 workspace 内目录的一层条目。"""

    def close(self) -> None:
        """释放环境资源；本机实现无需操作。"""


def truncate_output(text: str, *, max_chars: int = 50_000) -> tuple[str, bool]:
    """截断过长命令输出，保留头尾以便定位失败。"""
    if len(text) <= max_chars:
        return text, False
    head_size = max_chars // 2
    tail_size = max_chars - head_size
    omitted = len(text) - max_chars
    return (
        text[:head_size] + f"\n[中间 {omitted} 个字符已省略]\n" + text[-tail_size:],
        True,
    )
