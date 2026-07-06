"""Skill dataclass：单个 skill 的数据结构。

融合 HelloAgents 生产版和 archive 两种风格：
  - 从 HelloAgents 继承：dir 属性 + scripts/examples/references 子目录自动扫描
  - 从 archive 继承：allowed_tools + disable_model_invocation（accrete 特色，
    skill 控制 ToolRegistry 白名单 + 是否暴露给 LLM 主动调用）

字段语义：
  - `name` / `description` — 来自 frontmatter 必需字段
  - `body` — SKILL.md 去掉 frontmatter 后的正文
  - `allowed_tools` — None 表示不裁剪；非 None 列表表示只允许这些 tool
  - `disable_model_invocation` — 默认 False；True 时不让 skill 本身作为可被 LLM
    自主选择的"能力"暴露（仅命令/路由触发）
  - `path` — SKILL.md 文件路径
  - `dir` — skill 子目录（含 scripts/ 等资源）
  - `scope` — 骨架预留字段。元数据标签，当前不驱动行为分支（reflexion 归属
    用）。后续可升级驱动 user_overrides / shared skill 等运行时行为。
    默认 = skill name（loader 自动填充）
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class Skill:
    name: str
    description: str
    body: str
    path: Path
    dir: Path
    allowed_tools: Optional[list[str]] = None
    disable_model_invocation: bool = False
    scope: Optional[str] = None

    @property
    def scripts(self) -> list[Path]:
        """skill 目录下 scripts/ 里的所有文件（递归）。"""
        scripts_dir = self.dir / "scripts"
        if not scripts_dir.exists():
            return []
        return [f for f in scripts_dir.rglob("*") if f.is_file()]

    @property
    def examples(self) -> list[Path]:
        examples_dir = self.dir / "examples"
        if not examples_dir.exists():
            return []
        return [f for f in examples_dir.rglob("*") if f.is_file()]

    @property
    def references(self) -> list[Path]:
        references_dir = self.dir / "references"
        if not references_dir.exists():
            return []
        return [f for f in references_dir.rglob("*") if f.is_file()]
