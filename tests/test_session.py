"""会话历史的持久化、恢复和崩溃尾部修剪。"""

import json
from pathlib import Path

import pytest

from infra.core.types import FunctionCall, Message, ToolCall
from infra.runtime.session import create_session, resolve_artifacts_root, resume_session


def test_artifact_namespace_stays_below_its_root(tmp_path: Path):
    """运行分区各自拥有独立的 artifact 根目录。"""
    assert resolve_artifacts_root(tmp_path, "analysis/experiment") == tmp_path / "analysis" / "experiment"

    with pytest.raises(ValueError):
        resolve_artifacts_root(tmp_path, "../outside")


def test_session_uses_date_partition_and_indexed_directory(tmp_path: Path):
    session = create_session(task="任务", root=tmp_path)

    assert session.directory.parent.parent == tmp_path
    assert session.directory.name == session.session_id
    assert session.history_path == session.directory / "history.jsonl"
    index = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
    assert index[0]["directory"] == session.directory.relative_to(tmp_path).as_posix()

    resumed = resume_session(session.session_id, root=tmp_path)
    assert resumed.directory == session.directory


def _tool_call(call_id: str) -> Message:
    return Message(
        "assistant",
        "开始处理",
        tool_calls=[ToolCall(call_id, function=FunctionCall("bash", '{"command":"ls"}'))],
    )


def test_resume_restores_complete_tool_turn(tmp_path: Path):
    session = create_session(task="任务", root=tmp_path)
    session.append_message(Message("user", "任务"))
    session.append_message(_tool_call("call-1"))
    session.append_message(Message("tool", "文件列表", tool_call_id="call-1"))

    resumed = resume_session(session.session_id, root=tmp_path)

    history = resumed.restore_history()
    assert [message.role for message in history] == ["user", "assistant", "tool"]
    assert history[1].tool_calls[0].id == "call-1"
    assert history[2].tool_call_id == "call-1"


def test_resume_discards_incomplete_tail_tool_turn(tmp_path: Path):
    session = create_session(task="任务", root=tmp_path)
    session.append_message(Message("user", "任务"))
    session.append_message(_tool_call("call-1"))

    resumed = resume_session(session.session_id, root=tmp_path)

    assert [message.role for message in resumed.restore_history()] == ["user"]


def test_corrupt_final_line_is_tolerated_but_middle_line_is_not(tmp_path: Path):
    session = create_session(task="任务", root=tmp_path)
    session.append_message(Message("user", "任务"))
    with session.history_path.open("a", encoding="utf-8") as history:
        history.write('{"type":"mess')
    assert [message.role for message in session.restore_history()] == ["user"]

    with session.history_path.open("a", encoding="utf-8") as history:
        history.write("\n")
        history.write(json := '{"type":"message","message":{"role":"assistant","content":"后续"}}\n')
    with pytest.raises(ValueError, match="第 2 行损坏"):
        session.restore_history()
