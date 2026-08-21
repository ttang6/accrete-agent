"""可恢复会话的完整消息日志与元数据。"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infra.core.types import FunctionCall, Message, ToolCall


def _root() -> Path:
    return Path("artifacts")


def resolve_artifacts_root(
    root: Path | None = None,
    namespace: str = "general",
) -> Path:
    """返回某类运行专用的 artifact 根目录。

    Args:
        namespace: ``artifacts/`` 下的相对分区路径，例如 ``general`` 或
            ``analysis/experiment/stage``。不允许离开 artifact 根目录。
    """
    namespace_path = Path(namespace)
    if (
        not namespace.strip()
        or namespace_path.is_absolute()
        or namespace_path.drive
        or ".." in namespace_path.parts
    ):
        raise ValueError(f"artifact namespace 必须是安全的相对路径: {namespace!r}")
    return (root or _root()) / namespace_path


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _day() -> str:
    """返回 session 首次创建日期，作为 artifacts 的稳定分区。"""
    return datetime.now(timezone.utc).date().isoformat()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _message_from_openai(raw: dict[str, Any]) -> Message:
    """把会话日志中的 OpenAI 兼容消息恢复为 Core 消息。"""
    tool_calls = raw.get("tool_calls")
    calls = None
    if tool_calls is not None:
        calls = [
            ToolCall(
                id=str(call["id"]),
                type=call.get("type", "function"),
                function=FunctionCall(
                    name=str(call["function"]["name"]),
                    arguments=str(call["function"].get("arguments", "{}")),
                ),
            )
            for call in tool_calls
        ]
    return Message(
        role=raw["role"],
        content=raw.get("content"),
        reasoning_content=raw.get("reasoning_content"),
        tool_calls=calls,
        tool_call_id=raw.get("tool_call_id"),
        name=raw.get("name"),
    )


def _load_messages(path: Path) -> list[Message]:
    """读取会话消息；只容忍文件末尾的崩溃残行。"""
    if not path.exists():
        return []
    messages: list[Message] = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        try:
            entry = json.loads(line)
            if entry.get("type") != "message":
                raise ValueError("未知会话事件")
            messages.append(_message_from_openai(entry["message"]))
        except (KeyError, TypeError, json.JSONDecodeError, ValueError) as error:
            if index == len(lines) - 1:
                break
            raise ValueError(f"会话日志第 {index + 1} 行损坏") from error
    return _clean_history(messages)


def _clean_history(messages: list[Message]) -> list[Message]:
    """只保留工具调用与结果完整配对的历史，丢弃崩溃留下的尾部半回合。"""
    clean: list[Message] = []
    pending_assistant: Message | None = None
    pending_results: dict[str, Message] = {}

    for message in messages:
        if pending_assistant is None:
            if message.role == "assistant" and message.tool_calls:
                pending_assistant = message
                pending_results = {}
            elif message.role != "tool":
                clean.append(message)
            continue

        expected = {call.id for call in pending_assistant.tool_calls or []}
        if message.role == "tool" and message.tool_call_id in expected:
            pending_results[message.tool_call_id] = message
            if expected <= pending_results.keys():
                clean.append(pending_assistant)
                clean.extend(pending_results[call.id] for call in pending_assistant.tool_calls or [])
                pending_assistant = None
                pending_results = {}
            continue

        # 一个新消息出现在未完成工具回合之后，说明前一回合不能恢复；丢掉它再继续。
        pending_assistant = None
        pending_results = {}
        if message.role != "tool":
            clean.append(message)
    return clean


class Session:
    """一个持久会话；完整消息日志用于恢复，meta/index 仅用于展示与检索。"""

    def __init__(self, session_id: str, root: Path, directory: Path, meta: dict[str, Any]) -> None:
        self.session_id = session_id
        self.root = root
        self._directory = directory
        self.meta = meta

    @property
    def directory(self) -> Path:
        """返回该会话专属目录。"""
        return self._directory

    @property
    def history_path(self) -> Path:
        """返回保存完整模型可见消息的 JSONL 路径。"""
        return self.directory / "history.jsonl"

    def append_message(self, message: Message) -> None:
        """追加一条模型可见消息，并立即刷新以缩小崩溃丢失窗口。"""
        self.directory.mkdir(parents=True, exist_ok=True)
        entry = {"type": "message", "message": message.to_openai()}
        with self.history_path.open("a", encoding="utf-8") as history:
            history.write(json.dumps(entry, ensure_ascii=False) + "\n")
            history.flush()

    def restore_history(self) -> list[Message]:
        """恢复已完成回合的干净历史，不返回半截工具调用。"""
        return _load_messages(self.history_path)

    def update(self, **kwargs: Any) -> None:
        """更新可重建的会话摘要。"""
        self.meta.update(kwargs)
        self.meta["updated_at"] = _now()
        _write_json(self.directory / "meta.json", self.meta)
        self._touch_index()

    def close(self) -> None:
        """会话按条打开并刷新日志，当前没有常驻句柄需要关闭。"""

    def _touch_index(self) -> None:
        index_path = self.root / "index.json"
        entries = _read_json(index_path, [])
        entry = {
            "session_id": self.session_id,
            "updated_at": self.meta["updated_at"],
            "status": self.meta.get("status", "active"),
            "title": self.meta.get("title", ""),
            "first_user_message": self.meta.get("first_user_message", ""),
            "directory": self.directory.relative_to(self.root).as_posix(),
        }
        entries = [item for item in entries if item.get("session_id") != self.session_id]
        entries.insert(0, entry)
        _write_json(index_path, entries)


def create_session(
    *,
    task: str = "",
    first_user_message: str = "",
    root: Path | None = None,
) -> Session:
    """创建空会话；第一条用户消息由实际运行开始时写入。"""
    root = root or _root()
    session_id = "ses_" + uuid.uuid4().hex[:12]
    now = _now()
    directory = root / _day() / session_id
    session = Session(
        session_id,
        root,
        directory,
        {
            "session_id": session_id,
            "created_at": now,
            "updated_at": now,
            "status": "active",
            "title": "",
            "first_user_message": (first_user_message or task)[:120],
        },
    )
    session.update()
    return session


def resume_session(session_id: str, *, root: Path | None = None) -> Session:
    """重新打开已有会话，并将其状态标为 active。"""
    root = root or _root()
    entries = _read_json(root / "index.json", [])
    entry = next((item for item in entries if item.get("session_id") == session_id), None)
    if entry is None or not isinstance(entry.get("directory"), str):
        raise FileNotFoundError(session_id)
    directory = root / entry["directory"]
    if not directory.exists():
        raise FileNotFoundError(session_id)
    meta = _read_json(directory / "meta.json", {"session_id": session_id})
    session = Session(session_id, root, directory, meta)
    session.update(status="active")
    return session


def list_sessions(root: Path | None = None, limit: int = 50) -> list[dict[str, Any]]:
    """按最近更新时间返回会话摘要。"""
    return _read_json((root or _root()) / "index.json", [])[:limit]
