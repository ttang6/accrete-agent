# context.py
from __future__ import annotations

import os
import platform
from datetime import datetime
from pathlib import Path
from typing import Protocol

from .state import RunState
from .types import Message


class ContextBuilder(Protocol):
    def build(self, state: RunState) -> list[Message]: ...


def _is_git_repo(cwd: str) -> bool:
    """cwd 下存在 .git（目录或 worktree 指针文件）即视为 git 仓库。"""
    return (Path(cwd) / ".git").exists()


class DefaultContextBuilder:
    """默认：system（系统提示词 + 环境块）+ 历史 messages。压缩/lesson 用 hook 注入。"""

    def __init__(self, system_prompt: str | None = None) -> None:
        self.system_prompt = system_prompt

    def build(self, state: RunState) -> list[Message]:
        parts = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.append(self.build_env_block(state.workdir))
        out: list[Message] = [Message(role="system", content="\n\n".join(parts))]
        out.extend(state.messages)
        return out

    @staticmethod
    def build_env_block(cwd: str) -> str:
        """生成环境信息块：当前时间、时区、工作目录、平台、shell、git 状态。"""
        now = datetime.now().astimezone()
        return f"""# Environment
        Current date and time: {now.strftime("%A, %Y-%m-%d %H:%M %Z")}
        Timezone: {now.tzinfo}
        Working directory: {cwd}
        Platform: {platform.system().lower()}
        Shell: {os.environ.get("SHELL") or os.environ.get("COMSPEC") or "unknown"}
        Git repository: {"yes" if _is_git_repo(cwd) else "no"}
        """
