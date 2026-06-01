"""SkillContract —— skill 运行期契约的声明层。

不要把 skill 的 runtime contract 散落在 prompt 和 harness 里。RepairGate 解了
"调用必须合法"，但留下两个泄漏点：

1. **Action contract 缺失**：SKILL.md 用 imperative 语句（"输出日报后调 mark"）
   依赖 LLM 自觉，不是协议级约束。"用户说记下来 / agent 口头承认但没真调 mark"
   就是这种泄漏。
2. **Evaluation contract 泄漏**：`harness.py:_DIGEST_MARKERS` 硬编码 ai-digest
   专属字符串，违反 channel-agnostic / skill-as-plugin 原则。

本模块把两者抽象成 skill 自带的 yaml 声明（`skills/<name>/skill.yaml`），
harness 只负责 dispatch 不再持有 skill-specific 知识。

向后兼容：未提供 skill.yaml 的 skill（含旧的 ai-digest）行为完全不变 ——
SkillLoader.get_contract 返回 None，所有 contract-based 检查跳过。

不在范围（推迟）：
- 完整 evaluator plugin lifecycle（每个 skill 自带 evaluator.py）—— 当前只
  一个 reference skill，过设计
- LLM-driven user_intent 分类 —— v0 只支持 lexical_hints 子串匹配
- 配置化全部阈值 —— 当前阈值在 main.py 顶部常量已可改

Schema 形态：
```yaml
name: ai-digest

action_contracts:
  - id: <unique_id>
    trigger:
      lexical_hints: [str, ...]      # 任一子串命中即创建 obligation
    obligation:                      # 必须满足的 tool call
      tool: skill_exec
      skill: ai-digest
      script: dup_check
      args:                          # subset match：obligation args 是必需子集
        action: mark
    on_missing:
      mode: repair_once              # finish 前未满足 → inject hint 一次
      message: <自然语言修复提示>

evaluation:
  triggers:
    - on_answer_contains_any: [str, ...]   # answer 含任一子串 → 触发 evaluator
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
# Action Contract（必备工具调用约束）
# ============================================================


@dataclass(frozen=True)
class ActionTrigger:
    """什么用户输入触发该 obligation。v0 只支持 lexical_hints。"""
    lexical_hints: List[str] = field(default_factory=list)

    def matches(self, user_text: str) -> bool:
        if not self.lexical_hints:
            return False
        return any(h in user_text for h in self.lexical_hints)


@dataclass(frozen=True)
class ActionObligation:
    """该 obligation 需要 LLM 完成的具体 tool call。

    args_match 语义：obligation 的 args 是 satisfied 的**必需子集**。
    实际 tool call args 可以多于 obligation.args，但 obligation.args 里的
    每个 (key, value) 都必须出现且相等。
    """
    tool: str
    skill: Optional[str] = None
    script: Optional[str] = None
    args_match: Dict[str, Any] = field(default_factory=dict)

    def matches_call(
        self, tool: str, kwargs: Dict[str, Any], output_is_failure: bool
    ) -> bool:
        """判一次成功 tool call 是否满足本 obligation。失败的 call 不算。"""
        if output_is_failure:
            return False
        if tool != self.tool:
            return False
        if self.skill is not None and kwargs.get("skill") != self.skill:
            return False
        if self.script is not None and kwargs.get("script") != self.script:
            return False
        # args_match 的每个 key/value 都必须出现在 kwargs.args 里
        if self.args_match:
            call_args = kwargs.get("args") or {}
            if not isinstance(call_args, dict):
                return False
            for k, v in self.args_match.items():
                if call_args.get(k) != v:
                    return False
        return True


@dataclass(frozen=True)
class ActionContract:
    """单条 action contract：when → obligation → satisfied_by → on_missing。"""
    id: str
    trigger: ActionTrigger
    obligation: ActionObligation
    on_missing_mode: str = "repair_once"  # v0 只支持 repair_once
    on_missing_message: str = ""


# ============================================================
# Evaluation Contract（evaluator trigger 声明化）
# ============================================================


@dataclass(frozen=True)
class EvaluationTrigger:
    """什么时候触发 evaluator。v0 只支持 on_answer_contains_any（替代当前
    harness._DIGEST_MARKERS 子串匹配的最小变更）。"""
    on_answer_contains_any: List[str] = field(default_factory=list)

    def is_triggered(self, answer: str) -> bool:
        if not self.on_answer_contains_any:
            return False
        return any(s in answer for s in self.on_answer_contains_any)


# ============================================================
# Coverage Contract（硬覆盖类别声明化）
# ============================================================


@dataclass(frozen=True)
class CoverageCategorySpec:
    """单条 coverage category 的完整规范。

    之前 `runtime/turn_context.py` 顶部 `_CATEGORY_MAP` /
    `_DEFAULT_THRESHOLDS` / `_COUNT_RE` 三个模块级常量硬编码 ai-digest 业务，
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
# Archive Contract（归档 markers 声明化）
# ============================================================


@dataclass(frozen=True)
class ArchiveManifest:
    """单 skill 的归档触发声明。

    之前 `runtime/harness.py` 的 `_DIGEST_ARCHIVE_MARKERS` 元组硬编码 6 个
    ai-digest 字符串，违反 framework / skill 边界。manifest 化后 harness 只 dispatch。

    与 evaluation.triggers 高度相似但**不能合并**：archive 包含 evaluator 不需要的
    扩展字段（如 "## 论文速览" vs evaluation 的 "## 论文"），且语义不同——evaluator
    判 quality，archive 判"是否值得落盘留底"。

    字段：
    - enabled: 关闭归档但保留声明（调试 / 测试场景）
    - output_dir: 相对 data_dir 的子路径名；None 时 caller fallback 默认目录
    - answer_contains_any: 任一子串命中即触发
    """
    enabled: bool = True
    output_dir: Optional[str] = None
    answer_contains_any: List[str] = field(default_factory=list)

    def is_triggered(self, answer: str) -> bool:
        """answer 是否命中任一归档 marker。enabled=False 时永不触发。"""
        if not self.enabled:
            return False
        if not self.answer_contains_any:
            return False
        return any(s in answer for s in self.answer_contains_any)


# ============================================================
# Skill Contract 顶层
# ============================================================


@dataclass
class SkillContract:
    """单 skill 的运行期契约集合。所有字段 optional：未声明则该 skill 无相关约束。"""
    name: str
    action_contracts: List[ActionContract] = field(default_factory=list)
    evaluation_triggers: List[EvaluationTrigger] = field(default_factory=list)
    coverage_manifest: Optional[CoverageManifest] = None
    archive_manifest: Optional[ArchiveManifest] = None

    def matches_action_triggers(self, user_text: str) -> List[ActionContract]:
        """user_text 命中哪些 action contract → 返回对应的 contract 列表。"""
        return [c for c in self.action_contracts if c.trigger.matches(user_text)]

    def is_evaluation_triggered(self, answer: str) -> bool:
        """answer 是否命中任一 evaluation trigger。"""
        return any(t.is_triggered(answer) for t in self.evaluation_triggers)

    def is_archive_triggered(self, answer: str) -> bool:
        """answer 是否命中归档 manifest（manifest=None / disabled → False）。"""
        if self.archive_manifest is None:
            return False
        return self.archive_manifest.is_triggered(answer)

    def archive_output_dir(self) -> Optional[str]:
        """归档目标目录（相对 data_dir 的子路径名）。无 manifest / 未指定 → None。"""
        if self.archive_manifest is None:
            return None
        return self.archive_manifest.output_dir


# ============================================================
# 加载
# ============================================================


def load_skill_contract(skill_dir: Path) -> Optional[SkillContract]:
    """读 `<skill_dir>/skill.yaml`，解析为 SkillContract。

    fail-open：文件不存在 / 解析失败 / schema 不符 → log warning 返回 None，
    skill 仍可正常运行（只是不享受 contract-based 保护）。
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
        action_contracts=_parse_action_contracts(data.get("action_contracts") or []),
        evaluation_triggers=_parse_eval_triggers(data.get("evaluation") or {}),
        coverage_manifest=_parse_coverage_manifest(data.get("coverage")),
        archive_manifest=_parse_archive_manifest(data.get("archive")),
    )


def _parse_action_contracts(raw_list: Any) -> List[ActionContract]:
    if not isinstance(raw_list, list):
        return []
    out: List[ActionContract] = []
    for raw in raw_list:
        if not isinstance(raw, dict):
            continue
        cid = raw.get("id")
        trigger_raw = raw.get("trigger") or {}
        obl_raw = raw.get("obligation") or {}
        on_missing = raw.get("on_missing") or {}
        if not isinstance(cid, str) or not isinstance(obl_raw, dict):
            continue
        tool = obl_raw.get("tool")
        if not isinstance(tool, str):
            continue
        out.append(
            ActionContract(
                id=cid,
                trigger=ActionTrigger(
                    lexical_hints=list(trigger_raw.get("lexical_hints") or []),
                ),
                obligation=ActionObligation(
                    tool=tool,
                    skill=obl_raw.get("skill"),
                    script=obl_raw.get("script"),
                    args_match=dict(obl_raw.get("args") or {}),
                ),
                on_missing_mode=on_missing.get("mode") or "repair_once",
                on_missing_message=on_missing.get("message") or "",
            )
        )
    return out


def _parse_eval_triggers(raw: Any) -> List[EvaluationTrigger]:
    if not isinstance(raw, dict):
        return []
    triggers_raw = raw.get("triggers") or []
    if not isinstance(triggers_raw, list):
        return []
    out: List[EvaluationTrigger] = []
    for t in triggers_raw:
        if not isinstance(t, dict):
            continue
        contains = t.get("on_answer_contains_any")
        if isinstance(contains, list):
            out.append(
                EvaluationTrigger(on_answer_contains_any=[str(s) for s in contains])
            )
    return out


def _parse_coverage_manifest(raw: Any) -> Optional[CoverageManifest]:
    """解析 coverage 段。无段 / 格式不符 → None（fail-open，runtime 会用 default
    fallback specs 保 ai-digest 旧行为）。

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


def _parse_archive_manifest(raw: Any) -> Optional[ArchiveManifest]:
    """解析 archive 段。无段 → None。

    Schema：
        archive:
          enabled: true                          # 默认 true，可显式关
          output_dir: digests                    # 相对 data_dir 子路径
          answer_contains_any:
            - "# AI 日报"
            - "## 论文速览"
    """
    if not isinstance(raw, dict):
        return None
    contains = raw.get("answer_contains_any")
    if not isinstance(contains, list):
        return None
    enabled = raw.get("enabled")
    if enabled is None:
        enabled_bool = True
    else:
        enabled_bool = bool(enabled)
    output_dir = raw.get("output_dir")
    if output_dir is not None and not isinstance(output_dir, str):
        output_dir = None
    return ArchiveManifest(
        enabled=enabled_bool,
        output_dir=output_dir,
        answer_contains_any=[str(s) for s in contains],
    )


# ============================================================
# Multi-skill aggregation（union 合并所有 skill 的 coverage manifest）
# ============================================================


def aggregate_coverage_specs(
    contracts: List[Optional[SkillContract]],
) -> List[CoverageCategorySpec]:
    """把多个 SkillContract 的 coverage_manifest union 合并为单一 spec 列表。

    与 `_is_evaluator_triggered_by_any_skill` 的"any-skill"心智一致——同一进程内
    所有 skill 的 coverage 类别共同生效。

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
