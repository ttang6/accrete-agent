"""Pi-style append-only trajectory. Evidence for analysis/evolution."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, TextIO

from .types import Cost, ErrorClass, Message


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time()*1000)%1000:03d}Z"


def _sid() -> str:
    return uuid.uuid4().hex[:8]


def args_hash(arguments: str | dict) -> str:
    if isinstance(arguments, dict):
        payload = json.dumps(arguments, sort_keys=True, ensure_ascii=False)
    else:
        try:
            payload = json.dumps(json.loads(arguments), sort_keys=True, ensure_ascii=False)
        except Exception:
            payload = str(arguments)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class Trajectory(Protocol):
    def write(self, entry: dict[str, Any]) -> str:
        """Append entry; returns id."""

    def close(self) -> None: ...


class JsonlTrajectory:
    def __init__(self, path: str | Path, *, run_id: str | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or uuid.uuid4().hex
        self._fp: TextIO = self.path.open("a", encoding="utf-8")
        self._last_id: str | None = None

    def write(self, entry: dict[str, Any]) -> str:
        if entry.get("type") != "session":
            entry.setdefault("id", _sid())
            entry.setdefault("ts", _ts())
            if "parent_id" not in entry:
                entry["parent_id"] = self._last_id
            self._last_id = entry["id"]
        line = json.dumps(entry, ensure_ascii=False, default=str)
        self._fp.write(line + "\n")
        self._fp.flush()
        return entry.get("id") or self.run_id

    def close(self) -> None:
        self._fp.close()


def write_session_header(
    traj: JsonlTrajectory,
    *,
    cwd: str,
    task: str,
    agent_name: str = "accrete",
    agent_version: str = "0.1",
    source_kind: str = "self",
    extra: dict | None = None,
) -> None:
    traj.write({
        "type": "session",
        "schema": "accrete.traj.v1",
        "version": 1,
        "run_id": traj.run_id,
        "ts": _ts(),
        "cwd": cwd,
        "task": task,
        "agent": {"name": agent_name, "version": agent_version},
        "source": {"kind": source_kind},
        **(extra or {}),
    })


def write_message(
    traj: JsonlTrajectory,
    message: Message,
    *,
    turn: int | None = None,
    provider: str | None = None,
    model: str | None = None,
    usage: dict | None = None,
    cost: Cost | None = None,
    stop_reason: str | None = None,
    latency_ms: float | None = None,
    thinking: str | None = None,
) -> str:
    entry: dict[str, Any] = {"type": "message", "message": message.to_openai()}
    # 思考文本单独一个字段，不并进 message：message 是要能原样回灌给端点的，
    # 混进去会改变下一轮请求的形状。
    if thinking:
        entry["thinking"] = thinking
    if turn is not None:
        entry["turn"] = turn
    if provider:
        entry["provider"] = provider
    if model:
        entry["model"] = model
    if usage:
        entry["usage"] = usage
    if cost is not None:
        entry["cost"] = cost.to_dict()
    if stop_reason:
        entry["stop_reason"] = stop_reason
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    return traj.write(entry)


def write_tool_exec(
    traj: JsonlTrajectory,
    *,
    tool_call_id: str,
    tool_name: str,
    arguments: str | dict,
    status: str,
    turn: int | None = None,
    error_class: ErrorClass | None = None,
    exit_code: int | None = None,
    latency_ms: float | None = None,
    output: str | None = None,
    truncated: bool = False,
) -> str:
    entry: dict[str, Any] = {
        "type": "tool_exec",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "args_hash": args_hash(arguments),
        "status": status,  # ok | error
    }
    if turn is not None:
        entry["turn"] = turn
    if error_class:
        entry["error_class"] = error_class
    if exit_code is not None:
        entry["exit_code"] = exit_code
    if latency_ms is not None:
        entry["latency_ms"] = latency_ms
    if output is not None:
        entry["output_bytes"] = len(output.encode("utf-8", errors="replace"))
        entry["output_hash"] = content_hash(output)
        entry["truncated"] = truncated
    return traj.write(entry)


def write_tool_state(
    traj: Trajectory,
    *,
    tool_name: str,
    enabled: bool,
    reason: str,
    turn: int | None = None,
) -> str:
    """记录工具在一次运行中被启用或禁用的原因。"""
    entry: dict[str, Any] = {
        "type": "tool_state",
        "tool_name": tool_name,
        "enabled": enabled,
        "reason": reason,
    }
    if turn is not None:
        entry["turn"] = turn
    return traj.write(entry)
