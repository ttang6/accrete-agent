"""GrepTool — 在工作目录下按正则搜索文件内容。

和 GlobTool 同为 strict baseline：`pattern` + `path` 两个必填 string、
additionalProperties:false，schema 完全静态，strict_mode 默认开着。它真有用
（让 LLM 直接定位代码/文本里的命中行，不靠对话来回贴），不是纯 probe。

边界与安全：只读、不写；path 必须解析到 cwd 之下，越界（含 .. 穿越）直接拒绝；
按扩展名跳过常见二进制；命中行截断防超长；总命中数设上限防把上下文灌满。
"""

from __future__ import annotations

import re
from pathlib import Path

from accrete.tool.base import BaseTool

# 命中上限 + 单行长度上限：搜索结果只是定位线索，不该把整份文件搬进上下文。
_MAX_MATCHES = 100
_MAX_LINE_CHARS = 300
# 跳过的二进制 / 体积型扩展名——grep 文本搜索对它们无意义且易污染输出。
_SKIP_SUFFIXES = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".gz", ".tar",
    ".whl", ".so", ".dll", ".pyc", ".bin", ".exe", ".sqlite", ".db",
})


class GrepTool(BaseTool):
    """在 cwd 下用正则搜索文件内容，返回 `相对路径:行号: 行` 命中。"""

    strict_mode = True

    @property
    def name(self) -> str:
        return "grep"

    @property
    def description(self) -> str:
        return (
            "在工作目录下按正则搜索文件内容，返回 `相对路径:行号: 命中行`。"
            "path 给文件搜单个文件，给目录则递归搜其下文本文件。最多 100 条命中。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Python 正则表达式，如 `def \\w+` 或 `TODO`",
                },
                "path": {
                    "type": "string",
                    "description": "搜索范围：相对 cwd 的文件或目录，搜全工作区传 `.`",
                },
            },
            "required": ["pattern", "path"],
            "additionalProperties": False,
        }

    def _execute(self, *, pattern: str, path: str, **_kwargs) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as e:
            return f"[grep] 非法正则：{e}"

        root = Path.cwd().resolve()
        target = (root / path).resolve()
        try:
            target.relative_to(root)
        except ValueError:
            return f"[grep] 路径越界（必须在工作目录下）：{path}"
        if not target.exists():
            return f"[grep] 路径不存在：{path}"

        files = [target] if target.is_file() else [
            p for p in target.rglob("*")
            if p.is_file() and p.suffix.lower() not in _SKIP_SUFFIXES
        ]

        hits: list[str] = []
        for f in files:
            try:
                with f.open("r", encoding="utf-8", errors="ignore") as fh:
                    for lineno, line in enumerate(fh, 1):
                        if regex.search(line):
                            rel = f.resolve().relative_to(root).as_posix()
                            hits.append(f"{rel}:{lineno}: {line.rstrip()[:_MAX_LINE_CHARS]}")
                            if len(hits) >= _MAX_MATCHES:
                                hits.append(f"...（命中已达上限 {_MAX_MATCHES}，停止）")
                                return "\n".join(hits)
            except OSError:
                continue
        if not hits:
            return f"[grep] 无命中：{pattern}"
        return "\n".join(hits)
