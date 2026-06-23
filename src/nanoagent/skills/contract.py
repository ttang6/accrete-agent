"""SkillContract —— skill 运行期契约的声明层（v3 收口后只剩 coverage 一类）。

历史上这里还托管过 action_contracts（RequiredActionGate）/ evaluation triggers
（旧 DigestEvaluator）/ archive markers 三类声明。v3 阶段三-2b 把验收链路改成
确定性的 **publish 发布流程**后，这三类整条退役：
- action_contracts → publish 流程自动写去重历史，不再靠 lexical_hints 猜用户意图；
- evaluation triggers → critic 子 agent（非阻断），harness 直接 dispatch；
- archive markers → 发布即落盘，archive 是 publish 的副作用、不再靠 answer 命中字串。

留下的唯一声明是 **coverage manifest**：硬覆盖类别的 (tool, skill, script) +
threshold + count regex。它仍是 skill-specific 数据从 framework 里分离的载体——
framework 只持运行规则，类别定义来自 skill 自带 yaml。

向后兼容：未提供 skill.yaml（含当前的 ai-digest——其 coverage 已下沉到 SKILL.md
body 的软方法论）的 skill 行为不变：SkillLoader.get_contract 返回 None，
coverage 检查跳过（CoverageChecker 无 specs → missing 永远空）。

Schema 形态：
```yaml
name: <skill name>

coverage:
  categories:
    - name: paper
      threshold: 3
      tool: skill_exec
      skill: ai-digest
      script: fetch_hf
      count_pattern: "(\\d+)\\s*篇"
      suggestion: fetch_hf      # 可选
```
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from nanoagent.core.logger import get_logger

_logger = get_logger("skills.contract")


# ============================================================
# Coverage Contract（硬覆盖类别声明化）
# ============================================================


@dataclass(frozen=True)
class CoverageCategorySpec:
    """单条 coverage category 的完整规范。

    把 ai-digest 业务从 framework 里分离：之前 `runtime/turn_context.py` 顶部
    `_CATEGORY_MAP` / `_DEFAULT_THRESHOLDS` / `_COUNT_RE` 三个模块级常量硬编码，
    违反 framework / skill 边界。manifest 化后 framework 只持运行规则，
    skill-specific 数据来自 yaml。

    字段：
    - name: coverage 维度的标签（例 "paper"），CoverageChecker.counts 用作 key
    - threshold: 达标计数门槛
    - tool / skill / script: (tool_name, skill, script) 三元组用于 _resolve_category
      命中判定。skill / script 为 None 时不做相应过滤
    - count_pattern: 编译后的 regex，第 1 group 必须是数字字符串。CoverageChecker
      在 tool 输出首个 markdown header 上 search
    - suggestion: build_hint 缺类别时的修复建议（默认用 script 名）
    """
    name: str
    threshold: int
    tool: str
    count_pattern: re.Pattern
    skill: Optional[str] = None
    script: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass(frozen=True)
class CoverageManifest:
    """单 skill 声明的 coverage 类别集合。无 manifest 时 SkillContract.coverage_manifest 为 None。"""
    categories: List[CoverageCategorySpec] = field(default_factory=list)


# ============================================================
# Skill Contract 顶层
# ============================================================


@dataclass
class SkillContract:
    """单 skill 的运行期契约集合。当前只剩 coverage_manifest 一类——
    未声明 coverage 段的 skill 该字段为 None，无任何覆盖约束。"""
    name: str
    coverage_manifest: Optional[CoverageManifest] = None


# ============================================================
# 加载
# ============================================================


def load_skill_contract(skill_dir: Path) -> Optional[SkillContract]:
    """读 `<skill_dir>/skill.yaml`，解析为 SkillContract（只取 coverage 段）。

    fail-open：文件不存在 / 解析失败 / schema 不符 → log warning 返回 None，
    skill 仍可正常运行（只是不享受 coverage 检查）。
    """
    path = skill_dir / "skill.yaml"
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
    except (OSError, yaml.YAMLError) as e:
        _logger.warning(f"skill.yaml 读取/解析失败 {path}: {e}")
        return None

    if not isinstance(data, dict):
        _logger.warning(f"skill.yaml 顶层必须是 mapping: {path}")
        return None

    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        _logger.warning(f"skill.yaml 缺 name 字段: {path}")
        return None

    return SkillContract(
        name=name,
        coverage_manifest=_parse_coverage_manifest(data.get("coverage")),
    )


def _parse_coverage_manifest(raw: Any) -> Optional[CoverageManifest]:
    """解析 coverage 段。无段 / 格式不符 → None（fail-open）。

    Schema：
        coverage:
          categories:
            - name: paper
              threshold: 3
              tool: skill_exec
              skill: ai-digest
              script: fetch_hf
              count_pattern: "(\\d+)\\s*篇"
              suggestion: fetch_hf      # 可选，build_hint 显示的修复建议
    """
    if not isinstance(raw, dict):
        return None
    cats_raw = raw.get("categories")
    if not isinstance(cats_raw, list):
        return None
    out: List[CoverageCategorySpec] = []
    for c in cats_raw:
        if not isinstance(c, dict):
            continue
        name = c.get("name")
        threshold = c.get("threshold")
        tool = c.get("tool")
        pattern_str = c.get("count_pattern")
        if not (
            isinstance(name, str) and name.strip()
            and isinstance(threshold, int) and threshold > 0
            and isinstance(tool, str) and tool.strip()
            and isinstance(pattern_str, str) and pattern_str.strip()
        ):
            _logger.warning(f"coverage category 字段不完整，跳过: {c}")
            continue
        try:
            pat = re.compile(pattern_str)
        except re.error as e:
            _logger.warning(f"coverage count_pattern 编译失败 {pattern_str}: {e}")
            continue
        out.append(
            CoverageCategorySpec(
                name=name,
                threshold=threshold,
                tool=tool,
                count_pattern=pat,
                skill=c.get("skill"),
                script=c.get("script"),
                suggestion=c.get("suggestion"),
            )
        )
    if not out:
        return None
    return CoverageManifest(categories=out)


# ============================================================
# Multi-skill aggregation（union 合并所有 skill 的 coverage manifest）
# ============================================================


def aggregate_coverage_specs(
    contracts: List[Optional[SkillContract]],
) -> List[CoverageCategorySpec]:
    """把多个 SkillContract 的 coverage_manifest union 合并为单一 spec 列表。

    同一进程内所有 skill 的 coverage 类别共同生效。

    冲突约定：同 (tool, skill, script) 三元组若在多 manifest 出现，**最后加载者胜**
    （SkillLoader.list_skills 返回顺序决定）。罕见场景，写注释明说足以。

    contracts 元素可为 None（skill 未声明 manifest），自动跳过。空输入返空列表。
    """
    by_key: Dict[
        tuple, CoverageCategorySpec
    ] = {}
    for contract in contracts:
        if contract is None or contract.coverage_manifest is None:
            continue
        for spec in contract.coverage_manifest.categories:
            by_key[(spec.tool, spec.skill, spec.script)] = spec
    return list(by_key.values())
