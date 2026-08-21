"""最小 coding agent 工具与本机环境的契约测试。"""

from pathlib import Path

from infra.runtime.environments.local import LocalEnvironment
from infra.runtime.tools import BashTool, EditTool, ReadTool, WriteTool


def test_file_tools_share_one_environment(tmp_path: Path):
    environment = LocalEnvironment(tmp_path)
    write = WriteTool(environment)
    read = ReadTool(environment)
    edit = EditTool(environment)

    assert not write.execute({"path": "src/example.py", "content": "one\ntwo\n"}).is_error
    displayed = read.execute({"path": "src/example.py", "offset": 1, "limit": 1})
    assert "2|two" in displayed.content

    changed = edit.execute({"path": "src/example.py", "old_string": "two", "new_string": "three"})
    assert not changed.is_error
    assert environment.read_file("src/example.py") == "one\nthree\n"


def test_tools_reject_workspace_escape(tmp_path: Path):
    environment = LocalEnvironment(tmp_path)

    result = WriteTool(environment).execute({"path": "../outside.txt", "content": "no"})

    assert result.is_error
    assert result.error_type == "permission"


def test_read_lists_directory_with_path_kind(tmp_path: Path):
    environment = LocalEnvironment(tmp_path)
    environment.write_file("a.txt", "a")
    environment.write_file("nested/b.txt", "b")

    result = ReadTool(environment).execute({"path": "."})

    assert not result.is_error
    assert "dir nested" in result.content
    assert "file a.txt" in result.content


def test_edit_refuses_ambiguous_replacement(tmp_path: Path):
    environment = LocalEnvironment(tmp_path)
    environment.write_file("repeat.txt", "same\nsame\n")

    result = EditTool(environment).execute({"path": "repeat.txt", "old_string": "same", "new_string": "new"})

    assert result.is_error
    assert environment.read_file("repeat.txt") == "same\nsame\n"


def test_bash_preserves_nonzero_exit_as_tool_error(tmp_path: Path):
    result = BashTool(LocalEnvironment(tmp_path)).execute({"command": "cmd /c exit 7"})

    assert result.is_error
    assert result.error_type == "exec_error"
    assert result.attributes["exit_code"] == 7


def test_builtin_tools_declare_contract_permission_groups(tmp_path: Path):
    """内置工具的权限分类应与 Manifest Gate 契约一致。"""
    environment = LocalEnvironment(tmp_path)

    assert ReadTool(environment).permission_group == "read_only"
    assert {WriteTool(environment).permission_group, EditTool(environment).permission_group,
            BashTool(environment).permission_group} == {"mutating"}
