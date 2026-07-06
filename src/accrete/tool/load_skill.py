"""LoadSkillTool — HelloAgents 风格的渐进披露 skill 触发器。

设计对齐 HelloAgents 生产版 `skill_tool.py`：
  - Tool 的 **description** 动态列出所有可用 skill + "何时使用"触发规则
  - LLM 从 tool schema 就能看到 skill 目录（不依赖 system_prompt 广告）
  - 用户自然语言请求 → LLM 匹配 skill description → 调 `load_skill(skill=...)`
  - 返回 SKILL.md body + scripts/references 资源清单，包在 `<skill-loaded>` XML 标签里
  - LLM 接下去按 body 指引调 `skill_exec` / `fetch` / 其他工具

与 `/skill <name>` 手动切换的分工：
  - `/skill` = 持久 pin：body 进**每一轮** system_prompt，直到 `/skill none`
  - `load_skill` = 就地注入：body 进**这一轮** conversation tool_result，随 history 滚动失效

两者并存。长会话锁定某 skill 用 /skill；单次任务用 load_skill。
"""

from typing import Optional

from accrete.skills.loader import Skill, SkillLoader
from accrete.tool.base import BaseTool


class LoadSkillTool(BaseTool):
    """按需加载 skill 的元数据 + body + 资源清单。"""

    def __init__(self, skill_loader: SkillLoader):
        self._loader = skill_loader

    @property
    def name(self) -> str:
        return "load_skill"

    @property
    def description(self) -> str:
        """动态 description：每次 schema 生成时重算 skill 列表。

        LLM 从 OpenAI function schema 里直接看到"可用技能：..."，不需要
        BASE_IDENTITY / system_prompt 广告 skill 存在。这是 HelloAgents
        "渐进披露"的核心机制。
        """
        descriptions = self._loader.get_descriptions().strip() or "（暂无可用技能）"
        return (
            "加载专业 skill 以完成领域任务。返回 skill 的完整说明 + 自带 scripts/"
            "references 资源清单。\n\n"
            "可用技能：\n"
            f"{descriptions}\n\n"
            "何时使用：\n"
            "- 用户任务明确匹配某个技能描述时，立即加载\n"
            "- 开始领域特定工作之前（如生成日报 / 分析特定文件）\n"
            "- 需要调用 skill 自带 scripts 时（先加载获取 script 清单 + 调用格式）\n\n"
            "加载后严格按返回的技能说明执行，不要凭印象调用。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "要加载的技能名称（从上文'可用技能'列表中选）",
                },
            },
            "required": ["skill"],
        }

    def validate(self, **kwargs) -> Optional[str]:
        if not (kwargs.get("skill") or "").strip():
            return "skill 参数不能为空"
        return None

    def _execute(self, **kwargs) -> str:
        skill_name = kwargs["skill"].strip()
        skill = self._loader.get_skill(skill_name)
        if skill is None:
            available = ", ".join(self._loader.list_skills()) or "（无）"
            return (
                f"[load_skill 错误] 技能 '{skill_name}' 不存在。"
                f"可用技能：{available}"
            )

        # render() 走合并 overrides + reflexion 注入
        body = self._loader.render(skill_name)
        resources = self._resources_hint(skill)

        # 包在 XML 标签里，给 LLM 一个清晰的"技能边界"信号
        return (
            f'<skill-loaded name="{skill.name}">\n'
            f"{body}"
            f"{resources}\n"
            f"</skill-loaded>\n\n"
            f"✅ 已加载技能：{skill.name}\n"
            f"📝 描述：{skill.description}\n\n"
            f"请严格按上述说明执行任务。"
        )

    @staticmethod
    def _resources_hint(skill: Skill) -> str:
        """列出 skill 自带的可执行脚本、参数定义和参考资源。"""
        hints = []
        scripts_dir = skill.dir / "scripts"
        if scripts_dir.exists():
            executable_scripts = sorted(
                f.name for f in scripts_dir.glob("*.py")
                if f.is_file() and not f.name.startswith("_")
            )
            schema_files = sorted(
                f.name for f in scripts_dir.glob("*.schema.json")
                if f.is_file() and not f.name.startswith("_")
            )
            if executable_scripts:
                shown = ", ".join(executable_scripts[:8])
                extra = f" 等 {len(executable_scripts)} 个" if len(executable_scripts) > 8 else ""
                hints.append(f"- 脚本：{shown}{extra}")
            if schema_files:
                shown = ", ".join(schema_files[:8])
                extra = f" 等 {len(schema_files)} 个" if len(schema_files) > 8 else ""
                hints.append(f"- 参数定义：{shown}{extra}")

        for folder, label in (("references", "参考文档"), ("examples", "示例")):
            folder_path = skill.dir / folder
            if not folder_path.exists():
                continue
            files = sorted(
                f.name for f in folder_path.glob("*")
                if f.is_file() and not f.name.startswith("_")
            )
            if not files:
                continue
            shown = ", ".join(files[:8])
            extra = f" 等 {len(files)} 个" if len(files) > 8 else ""
            hints.append(f"- {label}：{shown}{extra}")
        if not hints:
            return ""
        return "\n\n**可用资源**：\n" + "\n".join(hints)
