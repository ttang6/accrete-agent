"""GlobTool — 按 glob 模式列出工作目录下的文件路径。

它和 GrepTool 一起接替原 CalcTool 当 strict baseline：schema 完全静态（单个
必填 string，additionalProperties:false），天生满足 OpenAI strict 的全部约束，
于是 strict_mode 可以默认开着。区别于 SkillExecTool 的 args 是 free-form object、
strict 表达不了——这一对照正是"按能力而非厂商决定 strict"的活样本。

与 calc 不同，glob 是真有用的工具（让 LLM 自己定位文件，省去靠对话猜路径），
不是纯 probe。路径安全靠"结果必须落在 cwd 之下"兜底，模式里的 .. 穿越会被滤掉。
"""

from __future__ import annotations

from pathlib import Path

from accrete.tool.base import BaseTool

# 单次返回的路径上限，防一个 `**/*` 把整棵树灌进上下文。
_MAX_RESULTS = 200


class GlobTool(BaseTool):
    """按 glob 模式（支持 `**` 递归）列出 cwd 下匹配的文件。"""

    strict_mode = True

    @property
    def name(self) -> str:
        return "glob"

    @property
    def description(self) -> str:
        return (
            "按 glob 模式列出当前工作目录下的文件路径（支持 ** 递归，如 "
            "`src/**/*.py`）。返回相对路径，按字母序，最多 200 条。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "glob 模式，如 `*.md` 或 `src/**/*.py`",
                },
            },
            "required": ["pattern"],
            "additionalProperties": False,
        }

    def _execute(self, *, pattern: str, **_kwargs) -> str:
        root = Path.cwd().resolve()
        matches: list[str] = []
        for p in root.glob(pattern):
            if not p.is_file():
                continue
            try:
                rel = p.resolve().relative_to(root)
            except ValueError:
                # glob 模式里的 .. 把结果带出了 cwd —— 越界，丢弃
                continue
            matches.append(rel.as_posix())
        if not matches:
            return f"[glob] 无匹配：{pattern}"
        matches.sort()
        head = matches[:_MAX_RESULTS]
        out = "\n".join(head)
        if len(matches) > _MAX_RESULTS:
            out += f"\n...（共 {len(matches)} 条，已截断到前 {_MAX_RESULTS}）"
        return out
