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
  - `reflexions_store` 注入：render() 查最近 N 条 reflexion，前置为 `# 历史教训` 块

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
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import yaml

from nanoagent.core.logger import get_logger
from nanoagent.skills.base import Skill
from nanoagent.skills.contract import SkillContract, load_skill_contract

if TYPE_CHECKING:
    from nanoagent.evolution.reflexion import ReflexionStore
    from nanoagent.evolution.skill_preference_store import SkillPreferenceStore

_logger = get_logger("skills.loader")

# frontmatter 匹配：开头 `---\n...\n---\n`
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)


# 匹配 `{identifier}`（纯 Python identifier 形式）。不匹配 `{"key": value}` 这类 JSON。
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


# Generic Preference Application Contract。
# 跟 NL summary 配套出现：summary 描述"用户喜欢什么"，contract 描述"这个 skill
# 允许偏好影响什么"——把 preference memory 接到 skill action space 的 affordance 层。
#
# 核心 wording：偏好内容 soft / 应用协议在合法选择空间内 hard（受限硬化）。
_PREFERENCE_CONTRACT_GENERIC = """## 偏好应用协议（受限硬化）

上面的偏好内容是软记忆，**不得**覆盖当前用户请求、tool schema、required action 或 coverage gate。

但当多个合法执行选择都可以时，**必须**用偏好来决定排序、数量、关键词。本 skill 中，偏好可以影响：

1. 搜索 query / topic 关键词
2. 信息源分配
3. 候选筛选与排序
4. 输出风格与段落侧重

**规则**：
- 如果偏好提到 topic / concept / evaluation criterion，相关 search/query/ranking 步骤必须把它包含进去。
- 如果当前任务无法应用偏好（无相关候选 / 无相关自由度），正常继续，不得编造证据。
"""


# preference 注入模式：
#   off              不注入任何 preference 块（ablation 实验"无偏好"对照）
#   summary_only     只注入 NL summary 块
#   summary_contract 注入 NL summary + Generic Contract + skill-local rules（默认）
_PREFERENCE_MODE_ENV = "NANOAGENT_PREFERENCE_MODE"
_PREFERENCE_MODES = ("off", "summary_only", "summary_contract")
_PREFERENCE_MODE_DEFAULT = "summary_contract"


def _current_preference_mode() -> str:
    raw = os.getenv(_PREFERENCE_MODE_ENV, _PREFERENCE_MODE_DEFAULT).strip().lower()
    return raw if raw in _PREFERENCE_MODES else _PREFERENCE_MODE_DEFAULT


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
        # 或者挂 reflexions_store 让 render 自动注入历史教训
        loader = SkillLoader(Path("skills"), reflexions_store=refl_store)

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
        reflexions_store: Optional["ReflexionStore"] = None,
        reflexion_n: int = 5,
        preference_store: Optional["SkillPreferenceStore"] = None,
    ):
        self._skills_dir = Path(skills_dir)
        self._skills_dir.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: dict[str, dict] = {}
        self._skills_cache: dict[str, Skill] = {}
        # SkillContract 在首次 get_contract 时 lazy 加载（启动开销仅扫 metadata）。
        # value=None 表示已查过且文件不存在，不重复尝试加载。
        self._contract_cache: dict[str, Optional[SkillContract]] = {}
        self._reflexions_store = reflexions_store
        self._reflexion_n = reflexion_n
        # 自动推断的 skill 偏好后置注入到 body 之后
        # （区别于 reflexion 的前置——前置 = 硬约束 / 后置 = 软指导）
        self._preference_store = preference_store
        self._scan_metadata()

    @property
    def reflexions_store(self) -> Optional["ReflexionStore"]:
        """暴露内部 reflexions store 给 Harness 共享同一实例（HITL feedback 通道）。

        None 表示装配时未挂载——Harness 应据此降级，告诉用户 feedback 未启用。
        """
        return self._reflexions_store

    @property
    def preference_store(self) -> Optional["SkillPreferenceStore"]:
        """暴露 distilled preference store 给 Harness（/profile skill-prefs 子命令用）。"""
        return self._preference_store

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

    def get_contract(self, name: str) -> Optional[SkillContract]:
        """按需 lazy 加载 skill 的 runtime contract（skill.yaml）。

        无 skill.yaml 文件 → 返回 None（向后兼容：未声明 contract 的 skill
        行为不变）。文件解析失败 → 也返回 None（contract 模块内部 log warning）。
        """
        if name in self._contract_cache:
            return self._contract_cache[name]
        meta = self._metadata_cache.get(name)
        if meta is None:
            self._contract_cache[name] = None
            return None
        contract = load_skill_contract(meta["dir"])
        self._contract_cache[name] = contract
        return contract

    # ============================================================
    # user_overrides.json 加载
    # ============================================================

    def _load_skill_local_preference_rules(self, name: str) -> str:
        """加载 skills/<name>/preference_application.md（如存在）。

        skill-local if-then 规则是关键——把 generic Contract
        翻译成本 skill 具体的 query/filter/rank/summary 自由度上的应用规则。
        无文件时返空串，不当作错误（不是所有 skill 都需要）。
        """
        meta = self._metadata_cache.get(name)
        if meta is None:
            return ""
        path = meta["dir"] / "preference_application.md"
        if not path.exists():
            return ""
        try:
            return path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            _logger.warning(f"preference_application.md 读失败 {path}: {e}")
            return ""

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
          4. 若 reflexions_store 挂载 → 查 `render_for_skill(name)` 拿历史教训块，
             前置于 extra_blocks 之前注入
          5. 命名块按序前置于 body

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

        # 2. 查 reflexions（若挂载）
        reflexion_text = ""
        if self._reflexions_store is not None:
            try:
                reflexion_text = self._reflexions_store.render_for_skill(
                    name, n=self._reflexion_n
                )
            except Exception as e:
                _logger.warning(f"render_for_skill({name}) 失败: {e}")

        # 3. 组装 combined_blocks：历史教训前置，用户 extra_blocks 顺序保留
        combined_blocks: dict[str, str] = {}
        if reflexion_text:
            combined_blocks["历史教训"] = reflexion_text
        if extra_blocks:
            combined_blocks.update(extra_blocks)

        # 4. 后置 preference 三层注入
        # 设计意图：
        #   feedback / 历史教训前置 → 抢 attention，硬约束
        #   preference 后置三层 → 软指导 + 受限硬化的 affordance contract
        #     a. NL summary 块（"用户喜欢什么"，soft）
        #     b. Generic Contract 块（"该 skill 允许偏好影响哪些自由度"，bounded-hard）
        #     c. Skill-local rules 块（"具体到本 skill 的 if-then 应用规则"，可选文件）
        # 注入模式由 NANOAGENT_PREFERENCE_MODE env 控制（off / summary_only / summary_contract）
        suffix = ""
        mode = _current_preference_mode()
        if mode != "off" and self._preference_store is not None:
            try:
                pref_block = self._preference_store.render_for_skill(name)
                if pref_block:
                    parts: list[str] = [pref_block]
                    if mode == "summary_contract":
                        parts.append(_PREFERENCE_CONTRACT_GENERIC.rstrip())
                        sl_rules = self._load_skill_local_preference_rules(name)
                        if sl_rules:
                            parts.append(sl_rules)
                    suffix = "\n\n" + "\n\n".join(parts)
            except Exception as e:
                _logger.warning(f"preference render_for_skill({name}) 失败: {e}")

        if not combined_blocks:
            return rendered_body + suffix

        prefix = "".join(
            f"# {title}\n\n{content}\n\n" for title, content in combined_blocks.items()
        )
        return prefix + rendered_body + suffix

    # ============================================================
    # 维护
    # ============================================================

    def reload(self) -> None:
        """清空缓存并重新扫描。开发期热重载用。"""
        self._metadata_cache.clear()
        self._skills_cache.clear()
        self._contract_cache.clear()
        self._scan_metadata()
