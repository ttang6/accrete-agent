"""本机环境实现。"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from infra.core.types import EnvironmentFailure

from .base import (CommandResult, DirectoryEntry, Environment, PathKind,
                   WorkspacePathError, truncate_output)


class LocalEnvironment(Environment):
    """在指定 workspace 内执行命令和文件操作。"""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        workdir = self._resolve(cwd or ".")
        try:
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                start_new_session=os.name == "posix",
            )
        except OSError as error:
            raise EnvironmentFailure(f"无法启动命令: {error}") from error

        try:
            stdout_raw, stderr_raw = process.communicate(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            _kill_process_tree(process)
            stdout_raw, stderr_raw = process.communicate()
            timed_out = True

        stdout, stdout_truncated = truncate_output(_decode(stdout_raw or b""))
        stderr, stderr_truncated = truncate_output(_decode(stderr_raw or b""))
        return CommandResult(
            exit_code=process.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            truncated=stdout_truncated or stderr_truncated,
        )

    def path_kind(self, path: str) -> PathKind:
        target = self._resolve(path)
        if target.is_dir():
            return "directory"
        if target.is_file():
            return "file"
        return "missing"

    def read_file(self, path: str) -> str:
        return self._resolve(path).read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> int:
        target = self._resolve(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8")
        target.write_bytes(data)
        return len(data)

    def list_dir(self, path: str) -> list[DirectoryEntry]:
        target = self._resolve(path)
        return [
            DirectoryEntry(
                path=entry.relative_to(self.workspace_root).as_posix(),
                kind="directory" if entry.is_dir() else "file",
            )
            for entry in sorted(target.iterdir(), key=lambda item: (not item.is_dir(), item.name))
        ]

    def _resolve(self, path: str) -> Path:
        raw = Path(path)
        if raw.is_absolute():
            raise WorkspacePathError(f"路径必须相对 workspace: {path}")
        target = (self.workspace_root / raw).resolve()
        if target != self.workspace_root and self.workspace_root not in target.parents:
            raise WorkspacePathError(f"路径越出 workspace: {path}")
        return target


def _decode(data: bytes) -> str:
    """尽量按 UTF-8 解码命令输出，保留不可解码字节的提示。"""
    return data.decode("utf-8", errors="replace")


def _kill_process_tree(process: subprocess.Popen) -> None:
    """终止超时命令及其子进程，避免宿主机残留后台进程。"""
    try:
        if os.name == "posix":
            os.killpg(os.getpgid(process.pid), 9)
        else:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                check=False,
            )
    except (OSError, ProcessLookupError):
        pass
