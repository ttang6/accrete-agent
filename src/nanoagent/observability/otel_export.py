"""OTel GenAI 语义约定导出 —— 把一次 run 的 trace 事件导成行业标准 OpenTelemetry span。

分层原则(见 docs 讨论):`trace_schema` 是**内部领域模型**(驱动飞轮/eval,带 failure_class /
lesson_used 等飞轮语义);OTel GenAI 是**对外互操作/可观测层**。二者不混——这里只做导出:
RunTracer.save() 时把已收集的事件**重放成真 OTel span**(经真 SDK + 配置的 exporter),
任何 OTel 后端(Jaeger / CozeLoop / LoongSuite)可直接吃。飞轮/eval 消费者继续读内部 jsonl。

诚实边界:span 在 run 收尾时按**事件里记录的真实时间戳**构造(start_time/end_time 显式设),
而非解码期实时上下文传播——对每次 agent run(短)足够,且不必把 OTel 上下文穿过整个 loop。

守卫:`opentelemetry` 未装 或 `NANOAGENT_OTEL` 未设 → 导出静默 no-op,不影响主流程。
激活:`uv pip install -e ".[otel]"` + `NANOAGENT_OTEL=1`(+ 可选 exporter 配置,见 configure)。
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, List, Optional

# OTel GenAI 语义约定属性键(稳定字符串,不依赖 semconv 包的 _incubating 常量)。
GEN_AI_OPERATION_NAME = "gen_ai.operation.name"
GEN_AI_SYSTEM = "gen_ai.system"
GEN_AI_AGENT_NAME = "gen_ai.agent.name"
GEN_AI_REQUEST_MODEL = "gen_ai.request.model"
GEN_AI_USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
GEN_AI_USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
GEN_AI_TOOL_NAME = "gen_ai.tool.name"


def _otel_available() -> bool:
    try:
        import opentelemetry.trace  # noqa: F401
        return True
    except ImportError:
        return False


def is_enabled() -> bool:
    """导出是否该跑：`NANOAGENT_OTEL` 已设 且 opentelemetry 可导入。"""
    return bool(os.getenv("NANOAGENT_OTEL")) and _otel_available()


def _ns(iso: Optional[str]) -> Optional[int]:
    """ISO 时间戳 → epoch 纳秒。解析失败 → None。"""
    if not isinstance(iso, str) or not iso:
        return None
    try:
        return int(datetime.fromisoformat(iso).timestamp() * 1_000_000_000)
    except (ValueError, OverflowError):
        return None


def _start_from_end(end_ns: Optional[int], ev: dict) -> Optional[int]:
    """用 end_ns - duration_ms 反推 span 起点(工具/无 start 事件用)。"""
    if end_ns is None:
        return None
    dur = ev.get("duration_ms")
    if isinstance(dur, (int, float)):
        return end_ns - int(dur * 1_000_000)
    return end_ns


def _last_end_ns(steps: List[dict], fallback: int) -> int:
    latest = fallback
    for ev in steps:
        n = _ns(ev.get("timestamp")) or _ns(ev.get("ended_at"))
        if n is not None and n > latest:
            latest = n
    return latest


def _set(span, key: str, value: Any) -> None:
    if value is not None:
        try:
            span.set_attribute(key, value)
        except Exception:
            pass


def emit_run(
    agent_name: str,
    user_input: str,
    start_epoch: float,
    steps: List[dict],
    token_summary: Optional[dict] = None,
    *,
    tracer_provider=None,
    model: Optional[str] = None,
    system: Optional[str] = None,
) -> None:
    """把一次 run 的事件重放成 OTel GenAI span(根 invoke_agent + 子 chat / execute_tool)。

    fail-open：任何异常吞掉(可观测导出不该影响主流程)。tracer_provider=None 时用全局
    provider(需 configure 或宿主已配)；测试注入 InMemory provider。
    """
    if not _otel_available():
        return
    try:
        from opentelemetry import trace as ot
        from opentelemetry.trace import SpanKind, set_span_in_context

        provider = tracer_provider or ot.get_tracer_provider()
        tracer = provider.get_tracer("nanoagent.otel_export")

        start_ns = int(start_epoch * 1_000_000_000)
        end_ns = _last_end_ns(steps, start_ns)

        root = tracer.start_span(
            f"invoke_agent {agent_name}", start_time=start_ns, kind=SpanKind.INTERNAL
        )
        _set(root, GEN_AI_OPERATION_NAME, "invoke_agent")
        _set(root, GEN_AI_AGENT_NAME, agent_name)
        _set(root, GEN_AI_SYSTEM, system)
        _set(root, GEN_AI_REQUEST_MODEL, model)
        if isinstance(token_summary, dict):
            _set(root, GEN_AI_USAGE_INPUT_TOKENS, token_summary.get("in"))
            _set(root, GEN_AI_USAGE_OUTPUT_TOKENS, token_summary.get("out"))

        ctx = set_span_in_context(root)
        try:
            pending_llm_start: List[Optional[int]] = []
            for ev in steps:
                action = ev.get("action") or ev.get("type")
                if action == "llm_call_start":
                    pending_llm_start.append(_ns(ev.get("timestamp")))
                elif action == "llm_call_end":
                    e = _ns(ev.get("timestamp"))
                    s = pending_llm_start.pop() if pending_llm_start else _start_from_end(e, ev)
                    span = tracer.start_span(
                        "chat", context=ctx, start_time=s or e, kind=SpanKind.CLIENT
                    )
                    _set(span, GEN_AI_OPERATION_NAME, "chat")
                    _set(span, GEN_AI_REQUEST_MODEL, model)
                    span.end(end_time=e or s)
                elif action in ("tool_call_end", "tool_op"):
                    e = _ns(ev.get("timestamp"))
                    s = _start_from_end(e, ev)
                    tool = ev.get("tool") or "?"
                    span = tracer.start_span(
                        f"execute_tool {tool}", context=ctx,
                        start_time=s or e, kind=SpanKind.INTERNAL,
                    )
                    _set(span, GEN_AI_OPERATION_NAME, "execute_tool")
                    _set(span, GEN_AI_TOOL_NAME, tool)
                    _maybe_mark_error(span, ev.get("output"))
                    span.end(end_time=e or s)
        finally:
            root.end(end_time=end_ns)
    except Exception:
        return


def _maybe_mark_error(span, output) -> None:
    """工具输出是失败签名 → span 置 ERROR 状态。"""
    if not isinstance(output, str):
        return
    try:
        from nanoagent.runtime.tool_failure import is_tool_failure
        if is_tool_failure(output):
            from opentelemetry.trace import Status, StatusCode
            span.set_status(Status(StatusCode.ERROR))
    except Exception:
        pass


def configure(exporter: str = "console", endpoint: Optional[str] = None):
    """一次性配置全局 TracerProvider + exporter(宿主启动时调；测试不用,直接注入 provider)。

    exporter: "console"(打到 stdout,零外部依赖) | "otlp"(发 OTLP/HTTP,需 endpoint)。
    返回配好的 provider;opentelemetry 未装 → 返回 None。
    """
    if not _otel_available():
        return None
    from opentelemetry import trace as ot
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource

    provider = TracerProvider(resource=Resource.create({"service.name": "nanoagent"}))
    if exporter == "otlp":
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        exp = OTLPSpanExporter(endpoint=endpoint) if endpoint else OTLPSpanExporter()
    else:
        exp = ConsoleSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exp))
    ot.set_tracer_provider(provider)
    return provider
