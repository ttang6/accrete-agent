"""ToolCallRepairGate —— skill_exec subprocess 前的 jsonschema 校验层。

设计动机：
  Lesson 注入是经验层（"以前这种 args 错过"），但 LLM 看到自然语言文本不一定能
  转成合法 tool call。主流框架（LangGraph wrap_tool_call / Microsoft Agent Framework
  middleware）的共识是：把"建议"变成"调用前校验 + 修复约束"。

  真跑暴露过：某 PROMOTED lesson 已召回命中，但 agent 仍以**完全空 args** 调
  dup_check 三次 → stop_condition repeated_failure，日报没拿到。RepairGate 把这种
  "协议级错误"在 subprocess 前拦下来。

返回契约：
  - schema 不存在 / 无法读 → fail-open，返回 None（让 SkillExecTool 正常 subprocess）
  - schema 校验通过 → 返回 None
  - schema 校验失败 → 返回 ValidationOutcome（含 repair_text 字符串）
    SkillExecTool 把 repair_text 直接当 tool_result 返回。
    repair_text 以固定标记 `[gate-invalid-call]`（MARKER_GATE_INVALID_CALL）开头：
      1. MainLoop 检测前缀 → emit ACTION_TOOL_CALL_REPAIR_REQUIRED trace 事件
      2. FailureMemory `_FAILURE_SIGNATURES` 含此前缀 → 走 lesson 召回 / 重复失败
         / 3rd 次 stop_condition 的现有保护链
      3. OutcomeTracker 看到 lesson_used + 同 tool 后续 repair_required
         → 判 ineffective_application（不污染 helped/hurt 计数）

依赖：
  jsonschema（项目 requirements；transitive via mcp，pyproject 已显式声明）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError, best_match

from nanoagent.runtime.context_sources import MARKER_GATE_INVALID_CALL
from nanoagent.tool.base import ValidationOutcome


# repair 输出固定前缀 = gate 命名空间的 invalid-call marker。字面量单一来源在
# context_sources（B∩C 共读），此处仅留向后兼容别名（同 tool_failure 的
# `_FAILURE_SIGNATURES` 别名手法）；MainLoop / FailureMemory / OutcomeTracker 靠它识别。
REPAIR_REQUIRED_PREFIX: str = MARKER_GATE_INVALID_CALL


# ============================================================
# Schema 加载（与 describe_script 共用）
# ============================================================


def resolve_skill_script_schema_path(
    skills_dir: Path, skill: str, script: str
) -> Optional[Path]:
    """解析 skills/<skill>/scripts/<script>.schema.json 路径，防穿越。

    返回 None：路径越界 / 文件不存在。
    """
    expected_root = (skills_dir / skill / "scripts").resolve()
    candidate = (expected_root / f"{script}.schema.json").resolve()
    try:
        candidate.relative_to(expected_root)
    except ValueError:
        return None
    if not candidate.is_file():
        return None
    return candidate


def load_skill_script_schema(
    skills_dir: Path, skill: str, script: str
) -> Optional[dict]:
    """fail-open 读 schema dict。任何异常 → None（让上层走原路径）。

    SkillExecTool 用这个：schema 没有就跳过校验，照常 subprocess。
    DescribeScriptTool 不能用这个 —— 它要区分"路径不存在" vs "JSON 损坏"
    给不同 fallback 提示，所以仍走自己的逻辑。
    """
    path = resolve_skill_script_schema_path(skills_dir, skill, script)
    if path is None:
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


# ============================================================
# 校验 + 修复消息
# ============================================================


# ValidationOutcome 已挪到中性位置 `nanoagent.tool.base`（check_args 富钩子的返回类型，
# 不再绑 skill 校验）。这里只构造它。结构化 repair_example 沿飞轮 trace → episode →
# lesson.example 一路携带完整 dict，下次召回 LessonRetriever 据它渲染 [learned-example] 块。


def _extract_parameters(schema: dict) -> dict:
    """从 schema.json 顶层取 `parameters` 子对象（即真正的 JSON Schema）。

    schema.json 约定外层有 skill/name/description/parameters/examples/notes。
    校验只看 parameters。若顶层就是 JSON Schema（无 parameters wrapper），
    也兼容直接返回顶层。
    """
    params = schema.get("parameters")
    if isinstance(params, dict):
        return params
    return schema


def _collect_required_fields(schema: dict, args: dict) -> list[str]:
    """对比 schema.required 和 args.keys()，列出缺失字段（路径用点号）。

    只看顶层 required —— enum/嵌套 required 由 jsonschema validator 抓。
    """
    req = schema.get("required") or []
    if not isinstance(req, list):
        return []
    return [f"args.{k}" for k in req if k not in args]


def _collect_allowed_values(schema: dict) -> dict[str, list[Any]]:
    """从 schema.properties 抽 enum 字段 → 候选值 dict。

    只看顶层 properties —— 嵌套 enum 太啰嗦，让 LLM 看顶层即可。
    """
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return {}
    out: dict[str, list[Any]] = {}
    for k, v in props.items():
        if isinstance(v, dict) and isinstance(v.get("enum"), list):
            out[k] = list(v["enum"])
    return out


def _format_example_args(schema: dict) -> Optional[dict]:
    """schema.json 约定的 examples[0].args 当作"最小合法调用模板"。"""
    examples = schema.get("examples")
    if not isinstance(examples, list) or not examples:
        return None
    first = examples[0]
    if not isinstance(first, dict):
        return None
    args = first.get("args")
    if isinstance(args, dict):
        return args
    return None


def _format_repair_text(
    skill: str,
    script: str,
    error_message: str,
    required_fields: list[str],
    allowed_values: dict[str, list[Any]],
    example_args: Optional[dict],
) -> str:
    """给 LLM 的多行 repair observation。结构化但仍是文本（function calling
    返回值是 string）。固定以 REPAIR_REQUIRED_PREFIX 开头，下游靠它识别。
    """
    lines: list[str] = [
        f"{REPAIR_REQUIRED_PREFIX} skill_exec({skill}/{script}) 调用在 subprocess 前被 schema 校验拦下。",
        f"validation_error: {error_message}",
    ]
    if required_fields:
        lines.append(f"required_fields: {json.dumps(required_fields, ensure_ascii=False)}")
    if allowed_values:
        lines.append(
            "allowed_values: "
            + json.dumps(allowed_values, ensure_ascii=False)
        )
    if example_args is not None:
        # 单行 canonical JSON，下游 main_loop._after_tool_repair_gate 与 validation_error /
        # required_fields 同样按行前缀解析（取代旧的多行 example_call 块 + DOTALL 正则
        # 往返）。结构化对象单一来源即此处，渲染与解析读同一份 dict。
        lines.append(
            "repair_example: "
            + json.dumps(
                {"skill": skill, "script": script, "args": example_args},
                ensure_ascii=False,
            )
        )
    lines.append(
        "next_call_must: 严格按 schema 重新生成 args。如不确定参数 shape，"
        "先调 describe_script 拉完整 schema 再调 skill_exec。"
    )
    lines.append(
        "do_not_repeat: 不要再以同样的 args 重发本调用 —— "
        "这是协议级错误，不是 transient，重试同 args 会被 stop_condition 终止。"
    )
    return "\n".join(lines)


def validate_skill_args(
    schema: dict, skill: str, script: str, args: dict
) -> Optional[ValidationOutcome]:
    """对 args 做 jsonschema 校验。

    返回 None = 合法（subprocess 正常进行）。
    返回 ValidationOutcome = 非法（SkillExecTool 应直接返回 outcome.repair_text）。

    fail-safe：
    - schema 内部格式错（不可校验） → 返回 None（不阻塞合法调用）
    - args 不是 dict → 返回 None（让 SkillExecTool 自己报 args 错）
    """
    if not isinstance(args, dict):
        return None
    parameters = _extract_parameters(schema)
    try:
        validator = Draft202012Validator(parameters)
        # iter_errors 也可能抛 UnknownType 等（schema 内 type 非法）→ 一并 fail-safe
        errors = list(validator.iter_errors(args))
    except Exception:
        return None

    if not errors:
        return None

    # best_match 给最具诊断价值的那条
    primary: ValidationError = best_match(errors)  # type: ignore[assignment]
    error_message = primary.message if primary is not None else "schema validation failed"
    required_fields = _collect_required_fields(parameters, args)
    allowed_values = _collect_allowed_values(parameters)
    example_args = _format_example_args(schema)
    repair_text = _format_repair_text(
        skill=skill,
        script=script,
        error_message=error_message,
        required_fields=required_fields,
        allowed_values=allowed_values,
        example_args=example_args,
    )
    repair_example = (
        {"skill": skill, "script": script, "args": example_args}
        if example_args is not None
        else None
    )
    return ValidationOutcome(
        error_message=error_message,
        required_fields=required_fields,
        allowed_values=allowed_values,
        repair_text=repair_text,
        repair_example=repair_example,
    )
