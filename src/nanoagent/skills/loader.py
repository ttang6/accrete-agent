"""SkillLoader：扫描 skills/ 目录加载所有 SKILL.md。

融合两个框架的风格：
  - HelloAgents 生产版：两级缓存（启动只扫 metadata，body 按需 lazy load）+
    get_descriptions() 给 system prompt 的 bullet list + reload 热重载 + 子目录
    自动索引（scripts / examples / references）
  - archive nanoagent：allowed-tools + disable-model-invocation frontmatter 字段 +
    render() 方法支持 body 的 `{placeholder}` 占位渲染 + extra_blocks 前置块注入

扩展能力：
  - `scope` frontmatter 字段（骨架预留，默认 = skill dir name）
  - `user_overrides.json` 自动合并到 render() 的 format_map overrides

skills/ 目录约定：
    skills/
    ├ <name>/
    │  ├ SKILL.md            (必需，YAML frontmatter + Markdown body)
    │  ├ user_overrides.json (可选；key-value 映射，填充 body {placeholder})
    │  ├ scripts/            (可选)
    │  ├ examples/           (可选)
    │  └ references/         (可选)
    └ SKILL_TEMPLATE.md      (模板参考，不是 skill)

frontmatter schema：
    ---
    name: <skill name>                      # 必需
    description: <一句话何时用>              # 必需
    scope: <标签>                            # 可选，仅元数据，默认 = dir name
    allowed-tools:                           # 可选，裁剪 ToolRegistry 白名单
      - tool_a
    disable-model-invocation: true           # 可选，默认 false
    license: MIT                             # 可选，信息字段
    ---

    <markdown body>
"""

import json
import re
from pathlib import Path
from typing import Optional

import yaml

from nanoagent.core.logger import get_logger
from nanoagent.skills.base import Skill

_logger = get_logger("skills.loader")

# frontmatter 匹配：开头 `---\n...\n---\n`
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


# 匹配 `{identifier}`（纯 Python identifier 形式）。不匹配 `{"key": value}` 这类 JSON。
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


def _safe_substitute(template: str, overrides: dict) -> str:
    """只替换白名单 identifier 占位符，避开 str.format_map 对 JSON-like `{"key": v}` 的误判。

    - `{hf_max}` / `{arxiv_id}` 这类纯 identifier → 若 overrides 里有对应 key 则替换，否则保留原样
    - `{"max_results": 10}` 这类带引号 / 空格的 → 不匹配正则，原样保留
    - 空 overrides → 直接返回 template（无替换需求跳过）
    """
    if not overrides:
        return template

    def _replace(m: re.Match) -> str:
        key = m.group(1)
        if key in overrides:
            return str(overrides[key])
        return m.group(0)

    return _PLACEHOLDER_RE.sub(_replace, template)


class SkillLoader:
    """skill 加载器。

    两级缓存：
      - `_metadata_cache`: 构造时扫一遍，仅解析 frontmatter（约 100 tokens/skill）
      - `_skills_cache`: get_skill(name) 时才 lazy 读全文

    用法：
        loader = SkillLoader(Path("skills"))
        print(loader.get_descriptions())
        skill = loader.get_skill("ai-digest")
        rendered = loader.render("ai-digest",
            extra_blocks={"用户画像": "..."},
            output_count=8,
        )
    """

    def __init__(
        self,
        skills_dir: Path,
    ):
        self._skills_dir = Path(skills_dir)
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: dict[str, dict] = {}
        self._skills_cache: dict[str, Skill] = {}
        self._scan_metadata()

    # ============================================================
    # 扫描 + 加载
    # ============================================================

    def _scan_metadata(self) -> None:
        """启动时扫描所有 skill 子目录，仅解析 frontmatter 填 metadata_cache。"""
        if not self._skills_dir.exists():
            _logger.warning(f"skills 目录不存在: {self._skills_dir}")
            return

        for sub in sorted(self._skills_dir.iterdir()):
            if not sub.is_dir():
                continue
            skill_md = sub / "SKILL.md"
            if not skill_md.exists():
                continue

            meta = self._parse_frontmatter_only(skill_md)
            if not meta:
                _logger.warning(f"跳过无效 frontmatter: {skill_md}")
                continue

            name = meta.get("name") or sub.name
            self._metadata_cache[name] = {
                "name": name,
                "description": meta.get("description", ""),
                "allowed_tools": meta.get("allowed-tools"),
                "disable_model_invocation": bool(meta.get("disable-model-invocation", False)),
                "scope": meta.get("scope") or name,  # 默认 = skill name
                "path": skill_md,
                "dir": sub,
            }

    @staticmethod
    def _parse_frontmatter_only(path: Path) -> Optional[dict]:
        """仅解析 frontmatter，不读 body。解析失败返回 None。"""
        try:
            content = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None

        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            return None

        if not isinstance(meta, dict):
            return None

        # 必需字段校验
        if "name" not in meta or "description" not in meta:
            return None

        return meta

    def get_skill(self, name: str) -> Optional[Skill]:
        """按需 lazy 加载完整 skill（含 body）。"""
        if name in self._skills_cache:
            return self._skills_cache[name]

        meta = self._metadata_cache.get(name)
        if meta is None:
            return None

        skill_md: Path = meta["path"]
        try:
            content = skill_md.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            _logger.warning(f"读 {skill_md} 失败: {e}")
            return None

        # 切 frontmatter + body
        match = _FRONTMATTER_RE.match(content)
        if not match:
            return None
        body = content[match.end():].strip()

        skill = Skill(
            name=meta["name"],
            description=meta["description"],
            body=body,
            path=skill_md,
            dir=meta["dir"],
            allowed_tools=list(meta["allowed_tools"]) if meta["allowed_tools"] else None,
            disable_model_invocation=meta["disable_model_invocation"],
            scope=meta["scope"],
        )
        self._skills_cache[name] = skill
        return skill

    # ============================================================
    # 查询
    # ============================================================

    def list_skills(self) -> list[str]:
        return list(self._metadata_cache.keys())

    def get_descriptions(self) -> str:
        """格式化的 bullet list，适合塞进 system prompt。"""
        if not self._metadata_cache:
            return "（暂无可用 skill）"
        return "\n".join(
            f"- {name}: {meta['description']}"
            for name, meta in self._metadata_cache.items()
        )

    # ============================================================
    # user_overrides.json 加载
    # ============================================================

    @staticmethod
    def _load_user_overrides(skill_dir: Path) -> dict:
        """若 skill 目录下有 user_overrides.json，返回其内容作为 overrides dict。

        格式错 / 非 dict / 读失败 → warning 并返回空 dict，不中断 skill 加载。
        """
        override_file = skill_dir / "user_overrides.json"
        if not override_file.exists():
            return {}
        try:
            data = json.loads(override_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            _logger.warning(f"user_overrides.json 读失败 {override_file}: {e}")
            return {}
        if not isinstance(data, dict):
            _logger.warning(f"user_overrides.json 格式不符（非 dict）: {override_file}")
            return {}
        return data

    # ============================================================
    # 渲染（archive 特色 + 扩展）
    # ============================================================

    def render(
        self,
        name: str,
        extra_blocks: Optional[dict[str, str]] = None,
        **overrides,
    ) -> str:
        """渲染 skill body。

        合成顺序：
          1. 加载 skill.dir/user_overrides.json（若存在），作为底层 overrides
          2. 调用方 **overrides 叠加（优先级高于文件）
          3. body.format_map 填占位
          4. 命名块（extra_blocks）按序前置于 body

        Args:
            name: skill 名字
            extra_blocks: 形如 {"用户画像": "..."}，渲染为 `# 用户画像\n\n...\n\n` 前缀
            **overrides: body 里 {placeholder} 的替换值；覆盖 user_overrides.json
                里的同名字段；缺失字段原样保留

        Raises:
            KeyError: skill 不存在
        """
        skill = self.get_skill(name)
        if skill is None:
            raise KeyError(f"skill not found: {name}")

        # 1. 合并 overrides：文件作底层，调用方作覆盖
        file_overrides = self._load_user_overrides(skill.dir)
        merged_overrides = dict(file_overrides)
        merged_overrides.update(overrides)

        rendered_body = _safe_substitute(skill.body, merged_overrides)

        # 组装 combined_blocks：用户 extra_blocks 顺序保留
        combined_blocks: dict[str, str] = {}
        if extra_blocks:
            combined_blocks.update(extra_blocks)

        if not combined_blocks:
            return rendered_body

        prefix = "".join(
            f"# {title}\n\n{content}\n\n" for title, content in combined_blocks.items()
        )
        return prefix + rendered_body

    # ============================================================
    # 维护
    # ============================================================

    def reload(self) -> None:
        """清空缓存并重新扫描。开发期热重载用。"""
        self._metadata_cache.clear()
        self._skills_cache.clear()
        self._scan_metadata()
