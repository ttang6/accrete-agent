"""DescribeScriptTool — Two-hop progressive disclosure 的 describe 半跳。

设计动机：
  skill_exec 作为 hyper-tool 在 JSON Schema 层无法表达 `{script → 参数 shape}` 的
  discriminated union。脚本级参数约定只能靠 SKILL.md 自由文本，LLM 的 adherence
  比 schema 约束弱一个数量级。

  本 tool 让 LLM 在调 skill_exec 前**按需**拉取结构化 schema，把"被动给的
  文本"转为"主动拉的结构"。in-context 主动信息的 adherence 据业界观察比被动信息
  高 30-40%。

  Schema 来源：`skills/<skill>/scripts/<script>.schema.json`（侧文件方案）
  和 python 代码解耦——skill 作者改 schema 不用动代码，也便于未来把 script
  动态注册为独立 BaseTool 时复用这些 schema.json。

安全边界：
  路径校验和 skill_exec 同款（resolve + relative_to），防穿越、防越界。
  不执行任何代码，纯读 + JSON 解析 + 格式化。
"""

import json
from pathlib import Path
from typing import Any, Optional

from nanoagent.tool.base import BaseTool


class DescribeScriptTool(BaseTool):
    """按需披露 skill 下 script 的结构化参数 schema 和用法示例。"""

    def __init__(self, skills_dir: Path):
        self._skills_dir = Path(skills_dir).resolve()

    @property
    def name(self) -> str:
        return "describe_script"

    @property
    def description(self) -> str:
        return (
            "获取指定 skill 下某 script 的结构化参数 schema 和用法示例。"
            "对于 args 必填或参数复杂的 script（例如 dup_check 这类 action 分派型），"
            "调 skill_exec 之前先调本工具拿 schema，避免参数遗漏或猜错。"
            "对于 args 全可选的 script（如 fetch_hf / fetch_rss / fetch_github），"
            "若你已从 SKILL.md 或过往会话中确认参数形态，也可直接 skill_exec。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "skill": {
                    "type": "string",
                    "description": "skill 名（如 'ai-digest'）",
                },
                "script": {
                    "type": "string",
                    "description": "scripts/ 下的脚本名，不含 .py 或 .schema.json 后缀（如 'dup_check'）",
                },
            },
            "required": ["skill", "script"],
        }

    def validate(self, **kwargs) -> Optional[str]:
        if not (kwargs.get("skill") or "").strip():
            return "skill 参数不能为空"
        if not (kwargs.get("script") or "").strip():
            return "script 参数不能为空"
        return None

    def _execute(self, **kwargs) -> str:
        skill = kwargs["skill"].strip()
        script = kwargs["script"].strip()

        schema_path = self._resolve_schema_path(skill, script)
        if schema_path is None:
            return self._fallback(skill, script)

        try:
            raw = schema_path.read_text(encoding="utf-8")
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            return f"[describe_script 错误] schema.json 格式损坏: {e}"
        except OSError as e:
            return f"[describe_script 错误] 读取 schema 失败: {e}"

        if not isinstance(data, dict):
            return "[describe_script 错误] schema.json 顶层必须是 object"

        return self._format(data)

    def _resolve_schema_path(self, skill: str, script: str) -> Optional[Path]:
        """安全解析：必须落在 skills/<skill>/scripts/ 实体目录下的 .schema.json。"""
        expected_root = (self._skills_dir / skill / "scripts").resolve()
        candidate = (expected_root / f"{script}.schema.json").resolve()
        try:
            candidate.relative_to(expected_root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        return candidate

    def _fallback(self, skill: str, script: str) -> str:
        """schema.json 不存在时，给 LLM 提示可用 schema 列表，便于自我纠错。"""
        scripts_dir = self._skills_dir / skill / "scripts"
        if not scripts_dir.is_dir():
            return (
                f"[describe_script] skill '{skill}' 不存在或无 scripts/ 目录。"
                f"先调 load_skill 确认 skill 名。"
            )

        available = sorted(
            p.name[: -len(".schema.json")]
            for p in scripts_dir.glob("*.schema.json")
            if p.is_file()
        )
        msg = f"[describe_script] script '{script}' 没有对应的 schema.json。"
        if available:
            msg += f"\n该 skill 已声明 schema 的 script：{', '.join(available)}"
        else:
            msg += "\n该 skill 当前没有任何 script 声明 schema。"
        msg += (
            "\n对于未声明 schema 的 script，可参考 SKILL.md 的调用示例，"
            "或直接尝试 skill_exec（若 args 全可选即可 work）。"
        )
        return msg

    def _format(self, data: dict) -> str:
        """schema dict → markdown 供 LLM 阅读。

        输出顺序是 **example-first**：
          1. **正确调用形态** —— 完整 skill_exec wrapper 调用 JSON（含 args 包装）
          2. **常见错误** —— flat hoisting（把 inner 字段提到 skill_exec 顶层）
          3. **inner args schema** —— 真正的 JSON Schema 约束
          4. **额外说明（notes）**

        理由：实测 gpt-5.4-mini 系统性犯"漏 args wrapper"——LLM 看到 inner schema
        的 `properties: {magic_phrase: ...}` 误以为 magic_phrase 在 skill_exec
        顶层。把"正确 wrapper 形态"放最前 + 紧跟"常见错误"反例对照，能让模型一眼
        确定调用边界，再回头看 inner schema 时不再 hoist。

        schema.json 约定字段：
          - skill (str) / name (str) / description (str)
          - parameters (dict, JSON Schema)
          - examples (list of {title, args})
          - notes (list of 额外说明)
        """
        lines: list[str] = []

        skill = data.get("skill", "?")
        name = data.get("name", "(unnamed)")
        desc = data.get("description", "")

        lines.append(f"# script: {name}  (skill: {skill})")
        if desc:
            lines.append("")
            lines.append(desc)

        examples = data.get("examples") or []
        valid_examples = [ex for ex in examples if isinstance(ex, dict)] if isinstance(examples, list) else []

        # 1. Correct call shape —— 用 examples[0]（约定：第 1 个 example 即最小合法
        # 调用）展示完整 skill_exec wrapper JSON。如果 schema 没有 examples，
        # 用 schema.required 推断一个最小合法 args 示例 stub。
        first_args = (
            valid_examples[0].get("args", {})
            if valid_examples
            else _infer_minimal_args(data.get("parameters") or {})
        )
        lines.append("")
        lines.append("## ✅ 正确调用形态（必须用 args 包装 inner 字段）")
        lines.append("")
        lines.append("```json")
        lines.append(
            json.dumps(
                {"skill": skill, "script": name, "args": first_args},
                ensure_ascii=False,
                indent=2,
            )
        )
        lines.append("```")

        # 2. Common mistake —— flat hoisting 反例
        if isinstance(first_args, dict) and first_args:
            flat: dict[str, Any] = {"skill": skill, "script": name}
            flat.update(first_args)
            lines.append("")
            lines.append("## ❌ 常见错误（不要把 inner 字段提到 skill_exec 顶层）")
            lines.append("")
            lines.append("```json")
            lines.append(json.dumps(flat, ensure_ascii=False, indent=2))
            lines.append("```")
            lines.append("")
            lines.append(
                "上面是错误形态——RepairGate 会拦下并要求重发。inner 字段必须放进 args。"
            )

        # 3. Inner args schema
        params = data.get("parameters")
        if params:
            lines.append("")
            lines.append("## inner args schema (JSON Schema)")
            lines.append("```json")
            lines.append(json.dumps(params, ensure_ascii=False, indent=2))
            lines.append("```")

        # 额外的 examples（除了已经用作正确形态的第 1 个）展示
        extra_examples = valid_examples[1:]
        if extra_examples:
            lines.append("")
            lines.append("## 更多调用示例")
            for ex in extra_examples:
                title = ex.get("title", "")
                args = ex.get("args", {})
                if title:
                    lines.append("")
                    lines.append(f"**{title}**:")
                lines.append("```")
                lines.append(
                    f'skill_exec(skill="{skill}", script="{name}", '
                    f"args={json.dumps(args, ensure_ascii=False)})"
                )
                lines.append("```")

        notes = data.get("notes") or []
        if isinstance(notes, list) and notes:
            lines.append("")
            lines.append("## 注意事项")
            for note in notes:
                lines.append(f"- {note}")

        return "\n".join(lines)


def _infer_minimal_args(parameters: dict) -> dict:
    """schema.required 推断最小合法 args 示例 stub（schema 无 examples 时用）。

    每个 required 字段填一个 type-specific placeholder：string→"<填值>"，
    number/integer→0，boolean→false，array→[]，object→{}，其他→null。
    LLM 看到这个 stub 知道字段名和类型轮廓，结合 schema 自己填真值。
    """
    if not isinstance(parameters, dict):
        return {}
    required = parameters.get("required") or []
    properties = parameters.get("properties") or {}
    if not isinstance(required, list) or not isinstance(properties, dict):
        return {}
    placeholders: dict[str, Any] = {
        "string": "<填值>",
        "number": 0,
        "integer": 0,
        "boolean": False,
        "array": [],
        "object": {},
    }
    out: dict[str, Any] = {}
    for field_name in required:
        prop = properties.get(field_name)
        if not isinstance(prop, dict):
            out[field_name] = None
            continue
        # const 优先于 type：schema 有 const 表示字面值，最有用
        if "const" in prop:
            out[field_name] = prop["const"]
            continue
        ptype = prop.get("type")
        out[field_name] = placeholders.get(ptype, None)
    return out
