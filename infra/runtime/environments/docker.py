"""Docker 环境实现：四个工具在同一容器 workspace 内操作。"""

from __future__ import annotations

import posixpath
import subprocess
from pathlib import Path, PurePosixPath
from uuid import uuid4

from infra.core.types import EnvironmentFailure

from .base import (CommandResult, DirectoryEntry, Environment, PathKind,
                   WorkspacePathError, truncate_output)


class DockerEnvironment(Environment):
    """在无网络的保活容器中执行命令与文件操作。

    Args:
        image: 本次运行使用的镜像。
        workspace_root: 容器内唯一允许工具访问的根目录。
        interpreter: 容器内解释 bash 命令的可执行文件及参数。
    """

    def __init__(
        self,
        image: str,
        *,
        workspace_root: str = "/workspace",
        network: str = "none",
        container_lifetime: str = "2h",
        interpreter: tuple[str, ...] = ("sh", "-c"),
        start_timeout_s: float = 300.0,
    ) -> None:
        root = PurePosixPath(workspace_root)
        if not root.is_absolute():
            raise ValueError("workspace_root 必须是容器内绝对路径")
        self.image = image
        self.workspace_root = root
        self.interpreter = interpreter
        self.container_name = f"accrete-{uuid4().hex[:12]}"
        self._closed = False
        started = subprocess.run(
            [
                "docker", "run", "-d", "--name", self.container_name,
                f"--network={network}", image, "sleep", container_lifetime,
            ],
            capture_output=True,
            timeout=start_timeout_s,
        )
        if started.returncode != 0:
            raise EnvironmentFailure(
                f"容器启动失败: {_decode(started.stderr).strip()}"
            )
        self.container_id = _decode(started.stdout).strip()
        try:
            created = self._run(["mkdir", "-p", str(self.workspace_root)], timeout_s=30)
            self._require_alive(created)
            if created.returncode != 0:
                raise EnvironmentFailure(f"无法创建容器工作目录: {_decode(created.stderr).strip()}")
        except BaseException:
            self.close()
            raise

    def execute(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float | None = None,
    ) -> CommandResult:
        workdir = self._resolve(cwd or ".")
        # timeout 在容器内执行，避免杀掉 docker 客户端却留下测试进程。
        command_line = [
            "timeout", "-s", "KILL", str(max(timeout_s or 60.0, 0.1)),
            *self.interpreter, command,
        ]
        result = self._run(command_line, cwd=workdir, timeout_s=(timeout_s or 60.0) + 10)
        self._require_alive(result)
        stdout, stdout_truncated = truncate_output(_decode(result.stdout))
        stderr, stderr_truncated = truncate_output(_decode(result.stderr))
        return CommandResult(
            exit_code=result.returncode,
            stdout=stdout,
            stderr=stderr,
            timed_out=result.returncode == 137,
            truncated=stdout_truncated or stderr_truncated,
        )

    def path_kind(self, path: str) -> PathKind:
        target = self._resolve(path)
        result = self._run(
            [*self.interpreter, 'if [ -d "$ACCRETE_PATH" ]; then echo directory; '
                                'elif [ -f "$ACCRETE_PATH" ]; then echo file; else echo missing; fi'],
            env={"ACCRETE_PATH": str(target)},
            timeout_s=30,
        )
        self._require_alive(result)
        if result.returncode != 0:
            raise OSError(_decode(result.stderr).strip() or "无法读取路径类型")
        kind = _decode(result.stdout).strip()
        return kind if kind in {"file", "directory", "missing"} else "missing"

    def read_file(self, path: str) -> str:
        target = self._resolve(path)
        result = self._run(["cat", "--", str(target)], timeout_s=30)
        self._require_alive(result)
        if result.returncode != 0:
            raise FileNotFoundError(_decode(result.stderr).strip() or str(target))
        return _decode(result.stdout)

    def write_file(self, path: str, content: str) -> int:
        target = self._resolve(path)
        data = content.encode("utf-8")
        result = self._run(
            [*self.interpreter, 'mkdir -p "$(dirname "$ACCRETE_PATH")" && cat > "$ACCRETE_PATH"'],
            env={"ACCRETE_PATH": str(target)},
            input_data=data,
            timeout_s=30,
        )
        self._require_alive(result)
        if result.returncode != 0:
            raise OSError(_decode(result.stderr).strip() or f"无法写入 {target}")
        return len(data)

    def list_dir(self, path: str) -> list[DirectoryEntry]:
        target = self._resolve(path)
        script = (
            'if [ ! -d "$ACCRETE_PATH" ]; then exit 2; fi; '
            'for entry in "$ACCRETE_PATH"/* "$ACCRETE_PATH"/.[!.]* "$ACCRETE_PATH"/..?*; do '
            '[ -e "$entry" ] || continue; '
            'if [ -d "$entry" ]; then printf "directory\\t%s\\n" "${entry##*/}"; '
            'else printf "file\\t%s\\n" "${entry##*/}"; fi; done'
        )
        result = self._run(
            [*self.interpreter, script],
            env={"ACCRETE_PATH": str(target)},
            timeout_s=30,
        )
        self._require_alive(result)
        if result.returncode != 0:
            raise FileNotFoundError(_decode(result.stderr).strip() or str(target))
        return [
            DirectoryEntry(path=f"{path.rstrip('/')}/{name}".lstrip("./"), kind=kind)
            for line in _decode(result.stdout).splitlines()
            if "\t" in line
            for kind, name in [line.split("\t", 1)]
            if kind in {"file", "directory"}
        ]

    def close(self) -> None:
        """删除本次运行创建的容器；重复调用无副作用。"""
        if self._closed:
            return
        self._closed = True
        subprocess.run(
            ["docker", "rm", "-f", self.container_name],
            capture_output=True,
            check=False,
        )

    def copy_to_container(self, source: Path, destination: str) -> None:
        """把宿主文件复制到容器绝对路径。"""
        if not source.is_file():
            raise FileNotFoundError(source)
        target = PurePosixPath(destination)
        if not target.is_absolute():
            raise ValueError("destination 必须是容器内绝对路径")
        self._copy(str(source), f"{self.container_id}:{target}")

    def copy_from_container(self, source: str, destination: Path) -> None:
        """把容器文件复制到已有的宿主目录。"""
        target = PurePosixPath(source)
        if not target.is_absolute():
            raise ValueError("source 必须是容器内绝对路径")
        if not destination.parent.is_dir():
            raise FileNotFoundError(destination.parent)
        self._copy(f"{self.container_id}:{target}", str(destination))

    def _resolve(self, path: str) -> PurePosixPath:
        raw = PurePosixPath(path)
        if raw.is_absolute():
            raise WorkspacePathError(f"路径必须相对 workspace: {path}")
        target = self.workspace_root.joinpath(raw)
        normalized = PurePosixPath(posixpath.normpath(str(target)))
        if normalized != self.workspace_root and self.workspace_root not in normalized.parents:
            raise WorkspacePathError(f"路径越出 workspace: {path}")
        return normalized

    def _run(
        self,
        command: list[str],
        *,
        cwd: PurePosixPath | None = None,
        env: dict[str, str] | None = None,
        input_data: bytes | None = None,
        timeout_s: float,
    ) -> subprocess.CompletedProcess[bytes]:
        args = ["docker", "exec"]
        if input_data is not None:
            args.append("-i")
        if cwd is not None:
            args.extend(["-w", str(cwd)])
        for key, value in (env or {}).items():
            args.extend(["-e", f"{key}={value}"])
        try:
            return subprocess.run(
                [*args, self.container_id, *command],
                input=input_data,
                capture_output=True,
                timeout=timeout_s,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise EnvironmentFailure(f"无法执行 docker 命令: {error}") from error

    def _copy(self, source: str, destination: str) -> None:
        """执行 docker cp，并把客户端错误归类为环境失败。"""
        try:
            result = subprocess.run(
                ["docker", "cp", source, destination],
                capture_output=True,
                timeout=60,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise EnvironmentFailure(f"无法执行 docker cp: {error}") from error
        if result.returncode != 0:
            raise EnvironmentFailure(f"docker cp 失败: {_decode(result.stderr).strip()}")

    def _require_alive(self, result: subprocess.CompletedProcess) -> None:
        """非零 docker exec 后探活，区分任务失败和容器死亡。"""
        if result.returncode == 0:
            return
        try:
            probe = subprocess.run(
                ["docker", "inspect", "-f", "{{.State.Running}}", self.container_id],
                capture_output=True,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise EnvironmentFailure(f"容器探活失败: {error}") from error
        if probe.returncode != 0 or _decode(probe.stdout).strip() != "true":
            raise EnvironmentFailure(f"容器 {self.container_name} 已不在运行")


def _decode(data: bytes | None) -> str:
    """按 UTF-8 解码容器返回的文本。"""
    return (data or b"").decode("utf-8", errors="replace")
