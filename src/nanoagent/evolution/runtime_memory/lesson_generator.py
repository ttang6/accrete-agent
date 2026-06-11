"""LessonGenerator — RuntimeEpisode + FailureEvent + error_class → RuntimeLesson。

模板式：每 error_class 一个 trigger / recommendation / memory_text 模板。
不调 LLM，纯字符串格式化 + 确定性 lesson_id。

`lesson_id` = sha1(error_class + tool_key + cause_sig)[:16] —— **原因层语义键**：
同一种失败原因（同 error_class + 同 tool + 同原因签名）在不同 trace、
**不同参数**下共享同一 lesson_id，便于跨 trace 累加 evidence（caller 用 backend
`extend_lesson_evidence` 处理 LessonAlreadyExists）。args_hash 不参与键，
只作为 evidence 字段保留。

历史变更：
- 原本把 episode_id 也拼进 ID（一 trace 一 lesson）→ 持久化层积累大量近似重复；
- 后改 args_hash 语义键 → 同因不同参仍裂成多条，outcome 统计被摊薄、升级慢；
- 现为 cause_sig 原因签名键（`_cause_signature`，确定性提取、不调 LLM）。

后续会用 LLM 替换或扩展 memory_text 模板；trigger / recommendation
schema 不变——这是故意把模板写死的边界。
"""

from __future__ import annotations

import datetime
import hashlib
import json
import re
from typing import Any, Dict, Final, Optional

from nanoagent.evolution.runtime_memory.schema import (
    FailureEvent,
    LessonEvidence,
    LessonStatus,
    LessonStats,
    LessonTrigger,
    RuntimeEpisode,
    RuntimeLesson,
)


# ============================================================
# 模板表
# ============================================================

# 每个 error_class 对应一组：
#   tool_name_in_trigger: bool   是否绑工具（coverage/soft 不绑）
#   failure_count_gte:    int    repeated 类设 2，其他 1
#   recommendation:       str    短句注入 system prompt（中文）
#   memory_text:          str    自然语言长描述（英文，给 mem0.add 用）
_TEMPLATES: Final[Dict[str, Dict[str, Any]]] = {
    "schema_mismatch": {
        "tool_name_in_trigger": True,
        "failure_count_gte": 1,
        "recommendation": (
            "调用 {tool} 前先用 describe_script 确认参数 schema（历史失败特征：{cause}）"
        ),
        "memory_text": (
            "In episode {eid}, tool {tool} rejected args with "
            "schema_mismatch (signature={cause}): {err}. Re-validate input "
            "schema (e.g. via describe_script) before next call."
        ),
    },
    "repeated_same_args_failure": {
        "tool_name_in_trigger": True,
        "failure_count_gte": 2,
        "recommendation": (
            "{tool} 同类调用已重复失败 ≥2 次（特征 {cause}），必须改写 args 或换工具路径"
        ),
        "memory_text": (
            "Tool {tool} repeatedly fails (signature={cause}, ≥2 times) in "
            "episode {eid}: {err}. Rewrite args, switch tool, or call "
            "describe_script before retrying identical input."
        ),
    },
    "coverage_gap": {
        "tool_name_in_trigger": False,
        "failure_count_gte": 1,
        "recommendation": (
            "输出未覆盖必填字段 {missing}，先补齐再 finalize"
        ),
        "memory_text": (
            "Coverage check failed in episode {eid} at iter {it}: missing "
            "{missing}. Address coverage gap before finalizing."
        ),
    },
    "soft_quality_issue": {
        "tool_name_in_trigger": False,
        "failure_count_gte": 1,
        "recommendation": (
            "Evaluator 判定低质量（{recommended_action}），按反馈修订：{reason}"
        ),
        "memory_text": (
            "Evaluator returned non-finalize at iter {it} of episode {eid}: "
            "recommended_action={recommended_action}, missing={missing}, "
            "soft_issues={soft_issues}, reason={reason}. Revise per critique."
        ),
    },
    "tool_runtime_error": {
        "tool_name_in_trigger": True,
        "failure_count_gte": 1,
        "recommendation": (
            "{tool} 运行时异常（{err_type}），下次调用前考虑兜底/重试策略"
        ),
        "memory_text": (
            "Tool {tool} raised {err_type} in episode {eid}: {err}. Add "
            "fallback handling or retry-with-backoff."
        ),
    },
    # 通用 semantic_failure 模板：skill script 输出结构化 semantic_failures 列表
    # 标记软失败（质量/语义问题）时，用其首条 failure_type / message 形成定向
    # hint。framework 不识别 type 的具体取值，只透传——任何 skill 输出此 schema 都受益。
    "semantic_failure": {
        "tool_name_in_trigger": True,
        "failure_count_gte": 1,
        "recommendation": (
            "{tool} 报告了 {sf_count} 条 semantic failure（首条 {sf_primary_type}）："
            "{sf_primary_message}。按 failure_type 修复后重调；勿直接 finalize"
        ),
        "memory_text": (
            "Tool {tool} reported {sf_count} semantic_failures in episode {eid} "
            "(first type={sf_primary_type}): {sf_primary_message}. Address each "
            "failure_type before next call; the script's output JSON contains the "
            "full list."
        ),
    },
}


# ============================================================
# Generator
# ============================================================


class LessonGenerator:
    """RuntimeEpisode + FailureEvent + error_class → RuntimeLesson(CANDIDATE)。"""

    def __init__(self, ttl_days: int = 14):
        self._ttl_days = ttl_days

    def generate(
        self,
        episode: RuntimeEpisode,
        fe: FailureEvent,
        error_class: str,
    ) -> RuntimeLesson:
        if error_class not in _TEMPLATES:
            raise ValueError(f"unknown error_class: {error_class!r}")

        tpl = _TEMPLATES[error_class]
        tool_for_trigger = fe.tool_key if tpl["tool_name_in_trigger"] else None
        cause_sig = _cause_signature(fe, error_class)

        trigger = LessonTrigger(
            error_class=error_class,
            tool_name=tool_for_trigger,
            failure_count_gte=tpl["failure_count_gte"],
            scope=f"agent:{episode.agent_name}",
            task_type=None,  # 暂置 None；router 接入后再填
            condition_hint=_condition_hint(tool_for_trigger, error_class, cause_sig),
        )

        # 结构化 repair_example 从 extras 流入 evidence + 同时
        # 作为 lesson 首次 suggested_action（首条 evidence 即 canonical）；
        # extend_lesson_evidence 后续 ingest 时 LessonIngestor 决定是否更新。
        repair_example = (fe.extras or {}).get("repair_example")
        if not isinstance(repair_example, dict):
            repair_example = None

        evidence = LessonEvidence(
            source_episode_ids=[episode.episode_id],
            sample_trace_path=episode.trace_path,
            sample_failure_iteration=fe.iteration,
            sample_args_hash=fe.args_hash,
            sample_error_message=fe.error_message[:300],
            repair_example=repair_example,
        )

        fmt_ctx = _build_format_context(episode, fe)
        fmt_ctx["cause"] = cause_sig or "-"
        recommendation = _safe_format(tpl["recommendation"], fmt_ctx)
        memory_text = _safe_format(tpl["memory_text"], fmt_ctx)

        now = _now_iso()
        expires_on = (
            datetime.date.today() + datetime.timedelta(days=self._ttl_days)
        ).isoformat()

        lesson_id = _deterministic_lesson_id(
            error_class=error_class,
            tool_key=fe.tool_key,
            cause_sig=cause_sig,
        )

        tags = [error_class, episode.agent_name]
        if cause_sig:
            tags.append(f"sig:{cause_sig}")

        return RuntimeLesson(
            lesson_id=lesson_id,
            memory_text=memory_text,
            recommendation=recommendation,
            trigger=trigger,
            evidence=evidence,
            stats=LessonStats(),
            status=LessonStatus.CANDIDATE,
            created_at=now,
            updated_at=now,
            expires_on=expires_on,
            ttl_days=self._ttl_days,
            tags=tags,
            suggested_action=repair_example,
        )


# ============================================================
# helpers
# ============================================================


def _build_format_context(
    episode: RuntimeEpisode, fe: FailureEvent
) -> Dict[str, Any]:
    """供模板 `{...}` 占位符消费的上下文。所有字段都默认值兜底，避免 KeyError。"""
    extras = fe.extras or {}
    sf_list = extras.get("semantic_failures") or []
    sf_primary = sf_list[0] if isinstance(sf_list, list) and sf_list and isinstance(sf_list[0], dict) else {}
    return {
        "tool": fe.tool_key or "(no tool)",
        "ah": fe.args_hash or "-",
        "eid": episode.episode_id,
        "it": fe.iteration,
        "err": fe.error_message[:120].replace("\n", " "),
        "err_type": fe.error_type,
        "err_summary": fe.error_message[:60].replace("\n", " "),
        "missing": extras.get("missing") or [],
        "recommended_action": extras.get("recommended_action") or "-",
        "soft_issues": extras.get("soft_issues") or [],
        "reason": (extras.get("reason") or fe.error_message[:60]).replace("\n", " "),
        # semantic_failure 模板用：count + 首条 type/message（截断防 lesson 膨胀）
        "sf_count": len(sf_list) if isinstance(sf_list, list) else 0,
        "sf_primary_type": sf_primary.get("failure_type") or "unknown",
        "sf_primary_message": (sf_primary.get("message") or "")[:120].replace("\n", " "),
    }


def _safe_format(template: str, ctx: Dict[str, Any]) -> str:
    """模板格式化：缺字段不抛，留占位以便调试。"""
    try:
        return template.format(**ctx)
    except (KeyError, IndexError) as e:
        return template + f"  [format-error: {e}]"


def _now_iso() -> str:
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


# repair_text 里 "required_fields: [...]" 行的兜底解析（旧 trace 没有专用
# trace 事件字段时从 error_message 截断文本里抢救）
_REQUIRED_FIELDS_RE = re.compile(r"required_fields:\s*(\[[^\]]*\])")


def _cause_signature(fe: FailureEvent, error_class: str) -> str:
    """失败的原因签名——lesson 去重键的原因层成分。确定性提取、不调 LLM。

    取不到结构化要点时退化为空串（同 (error_class, tool) 聚成一条）。
    各 error_class 的来源：
    - schema_mismatch / repeated_same_args_failure：validator 的缺失字段列表
      （extras["required_fields"] 优先；旧 trace 从 error_message 文本兜底解析）
    - semantic_failure：首条 semantic_failures 的 failure_type
    - coverage_gap：缺失类目列表
    - soft_quality_issue：evaluator 的 recommended_action
    - tool_runtime_error 等：空串（修复策略类内通用，与具体参数无关）
    """
    extras = fe.extras or {}
    if error_class in ("schema_mismatch", "repeated_same_args_failure"):
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
    if error_class == "semantic_failure":
        sf = extras.get("semantic_failures") or []
        first = sf[0] if isinstance(sf, list) and sf and isinstance(sf[0], dict) else {}
        return str(first.get("failure_type") or "")
    if error_class == "coverage_gap":
        missing = extras.get("missing") or []
        if isinstance(missing, list) and missing:
            return "missing:" + ",".join(sorted(str(x) for x in missing))
        return ""
    if error_class == "soft_quality_issue":
        return str(extras.get("recommended_action") or "")
    return ""


def _condition_hint(
    tool_key: Optional[str], error_class: str, cause_sig: str
) -> str:
    """一句话适用条件，随 lesson 注入，声明这条经验的边界。模板生成。"""
    target = tool_key or "任意工具"
    hint = f"适用于 {target} 的 {error_class} 类失败"
    if cause_sig:
        hint += f"（特征 {cause_sig}）"
    return hint


def _deterministic_lesson_id(
    *, error_class: str, tool_key: str, cause_sig: str
) -> str:
    """原因层语义键：同一种失败原因跨 trace、跨参数共享同一 ID。
    episode_id / args_hash 不参与——cross-trace 去重靠 backend.extend_lesson_evidence，
    实例细节留在 evidence。"""
    payload = f"{error_class}|{tool_key}|{cause_sig}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
