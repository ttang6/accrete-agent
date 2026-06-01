"""LessonGenerator — RuntimeEpisode + FailureEvent + error_class → RuntimeLesson。

模板式：每 error_class 一个 trigger / recommendation / memory_text 模板。
不调 LLM，纯字符串格式化 + 确定性 lesson_id。

`lesson_id` = sha1(error_class + tool_key + args_hash)[:16] —— **语义键**：
同一种失败模式（同 error_class + 同 tool + 同参数 hash）在不同 trace 里
共享同一 lesson_id，便于跨 trace 累加 evidence（caller 用 backend
`extend_lesson_evidence` 处理 LessonAlreadyExists）。

历史变更：原本把 episode_id 也拼进 ID（一 trace 一 lesson），这会让持久化层
积累大量近似重复 lesson；改为语义键 + evidence 累加。

后续会用 LLM 替换或扩展 memory_text 模板；trigger / recommendation
schema 不变——这是故意把模板写死的边界。
"""

from __future__ import annotations

import datetime
import hashlib
from typing import Any, Dict, Final

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
            "调用 {tool} 前先用 describe_script 确认参数 schema：{err}"
        ),
        "memory_text": (
            "In episode {eid}, tool {tool} rejected args (hash={ah}) with "
            "schema_mismatch: {err}. Re-validate input schema (e.g. via "
            "describe_script) before next call."
        ),
    },
    "repeated_same_args_failure": {
        "tool_name_in_trigger": True,
        "failure_count_gte": 2,
        "recommendation": (
            "{tool} 用同样参数已失败 ≥2 次（{err_summary}），必须改写 args 或换工具路径"
        ),
        "memory_text": (
            "Tool {tool} repeatedly fails for args_hash={ah} (≥2 times) in "
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

        trigger = LessonTrigger(
            error_class=error_class,
            tool_name=tool_for_trigger,
            failure_count_gte=tpl["failure_count_gte"],
            scope=f"agent:{episode.agent_name}",
            task_type=None,  # 暂置 None；router 接入后再填
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
        recommendation = _safe_format(tpl["recommendation"], fmt_ctx)
        memory_text = _safe_format(tpl["memory_text"], fmt_ctx)

        now = _now_iso()
        expires_on = (
            datetime.date.today() + datetime.timedelta(days=self._ttl_days)
        ).isoformat()

        lesson_id = _deterministic_lesson_id(
            error_class=error_class,
            tool_key=fe.tool_key,
            args_hash=fe.args_hash,
        )

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
            tags=[error_class, episode.agent_name],
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


def _deterministic_lesson_id(
    *, error_class: str, tool_key: str, args_hash: str
) -> str:
    """语义键：同一种失败模式跨 trace 共享同一 ID。
    episode_id 不再参与——cross-trace 去重靠 backend.extend_lesson_evidence。"""
    payload = f"{error_class}|{tool_key}|{args_hash}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
