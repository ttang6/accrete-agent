"""模板式经验生产 —— 键重构（刀3·3f）后只剩 `build_schema_lesson` 一个纯函数。

**为什么塌成一个函数**:组件存在资格 = 它抽象的多样性还在不在。旧
`LessonGenerator` 的身份来自六类模板分发;键重构把多样性收走后（transient→重试 /
same_args→熔断 / 质量三类→离线 / runtime→unknown 弃权），只剩 schema_mismatch
一条有本地 oracle（校验器给的 required_fields）的模板生产者——类壳、
`_build_format_context`、`_safe_format` 等"六类通用"的装备一并撤。

**经验生产按 oracle 分级门控**:模板生产者仅 schema 过门（本函数);grounded
Reflector 以"观测到恢复"为门覆盖其余（`offline_mint`);无门的类诚实弃权走离线
（`lesson_ingestor` 的显式 abstain 分支）。回判对所有生产者统一。不是飞轮缩水,
是它第一次说实话。

`lesson_id` = sha1(failure_class + op + failure_reason)[:16] —— **原因层语义键**:
同一种失败原因（同 failure_class + tool + 原因签名）在不同 trace、不同参数下共享同一
lesson_id,便于跨 trace 累加 evidence。args_hash 不参与键,只作 evidence 实例证据。

保留的公共 helper（离线 mint / retriever 共用,故不随 LessonGenerator 一起塌）:
- `_failure_reason` —— ingestor 键审计 + offline_mint._build_lesson
- `_deterministic_lesson_id` —— offline_mint._build_lesson
- `_condition_hint` —— retriever.format_hint 召回时现渲染 [适用条件]
- `_now_iso` —— offline_mint / 其他 backend 时间戳
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from typing import Optional

from accrete.evolution.runtime_memory.schema import (
    FailureEvent,
    LessonEvidence,
    LessonStats,
    LessonTrigger,
    RuntimeEpisode,
    RuntimeLesson,
)


# ============================================================
# 唯一模板生产者:schema_mismatch（oracle = 本地 schema 校验的 required_fields）
# ============================================================


def build_schema_lesson(
    episode: RuntimeEpisode, fe: FailureEvent
) -> RuntimeLesson:
    """schema_mismatch 失败 → CANDIDATE lesson。纯函数,不调 LLM。

    只服务 schema_mismatch:这是唯一有本地 oracle（校验器已给 required_fields /
    结构化 repair_example）可确定性模板化的类。调用方（LessonIngestor）保证只对
    schema_mismatch 分类调本函数,其余类走显式 abstain。
    """
    failure_reason = _failure_reason(fe, "schema_mismatch")
    tool = fe.op or "(no tool)"
    advice = (
        f"调用 {tool} 前先用 describe_script 确认参数 schema"
        f"（历史失败特征：{failure_reason or '-'}）"
    )

    trigger = LessonTrigger(
        failure_class="schema_mismatch",
        op=fe.op,
        failure_count_gte=1,
        scope=f"agent:{episode.agent_name}",
        failure_reason=failure_reason,  # 确定性提取,存为可读字段
    )

    # 结构化 example 从 extras 流入 content.example（首条 evidence 即 canonical）;
    # 来源 = RepairGate 的 gate_repair_example（extras["repair_example"] 是 plumbing 键）。
    example = (fe.extras or {}).get("repair_example")
    if not isinstance(example, dict):
        example = None

    evidence = LessonEvidence(
        source_episode_ids=[episode.episode_id],
        sample_trace_path=episode.trace_path,
        sample_failure_iteration=fe.iteration,
        sample_args_hash=fe.args_hash,
        sample_error_message=fe.error_message[:300],
    )

    now = _now_iso()
    lesson_id = _deterministic_lesson_id(
        failure_class="schema_mismatch",
        op=fe.op,
        failure_reason=failure_reason,
    )

    return RuntimeLesson(
        lesson_id=lesson_id,
        advice=advice,
        trigger=trigger,
        evidence=evidence,
        source_type="template",
        stats=LessonStats(),
        created_at=now,
        updated_at=now,
        example=example,
    )


# ============================================================
# 公共 helper（离线 mint / retriever 共用）
# ============================================================


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# repair_text 里 "required_fields: [...]" 行的兜底解析（旧 trace 没有专用
# trace 事件字段时从 error_message 截断文本里抢救）
_REQUIRED_FIELDS_RE = re.compile(r"required_fields:\s*(\[[^\]]*\])")


def _failure_reason(fe: FailureEvent, failure_class: str) -> str:
    """失败的原因签名——lesson 去重键的原因层成分。确定性提取、不调 LLM。

    取不到结构化要点时退化为空串（同 (failure_class, tool) 聚成一条）。
    各 failure_class 的来源:
    - schema_mismatch：validator 的缺失字段列表
      （extras["required_fields"] 优先;旧 trace 从 error_message 文本兜底解析）
    - semantic_failure：首条 semantic_failures 的 failure_type
    - soft_quality_issue：evaluator 的 recommended_action
    - transient / unknown 等：空串（修复策略类内通用,与具体参数无关）
    """
    extras = fe.extras or {}
    if failure_class == "schema_mismatch":
        fields = extras.get("required_fields")
        if isinstance(fields, list) and fields:
            return "missing:" + ",".join(sorted(str(f) for f in fields))
        m = _REQUIRED_FIELDS_RE.search(fe.error_message or "")
        if m:
            try:
                parsed = json.loads(m.group(1))
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list) and parsed:
                return "missing:" + ",".join(sorted(str(f) for f in parsed))
        return ""
    if failure_class == "semantic_failure":
        sf = extras.get("semantic_failures") or []
        first = sf[0] if isinstance(sf, list) and sf and isinstance(sf[0], dict) else {}
        return str(first.get("failure_type") or "")
    if failure_class == "soft_quality_issue":
        return str(extras.get("recommended_action") or "")
    return ""


def _condition_hint(
    op: Optional[str], failure_class: str, failure_reason: str
) -> str:
    """一句话适用条件,随 lesson 注入,声明这条经验的边界。模板生成。"""
    target = op or "任意工具"
    hint = f"适用于 {target} 的 {failure_class} 类失败"
    if failure_reason:
        hint += f"（特征 {failure_reason}）"
    return hint


def _deterministic_lesson_id(
    *, failure_class: str, op: str, failure_reason: str
) -> str:
    """原因层语义键:同一种失败原因跨 trace、跨参数共享同一 ID。
    episode_id / args_hash 不参与——cross-trace 去重靠 backend.extend_lesson_evidence,
    实例细节留在 evidence。"""
    payload = f"{failure_class}|{op}|{failure_reason}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
