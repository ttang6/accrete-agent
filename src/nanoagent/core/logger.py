"""统一日志系统。

提供两种输出：
  - 控制台：人类可读的纯文本（带颜色级别 + 时间戳）
  - 文件：JSON Lines 格式，每行一条可解析的记录，写入 data/logs/

还提供 RunTracer，用于记录 Agent 单次 run 的完整轨迹。

用法：
    from nanoagent.core.logger import get_logger, RunTracer

    logger = get_logger("react_agent")
    logger.info("开始执行", extra={"step": 1, "tool": "search"})

    with RunTracer("react_agent", user_input="北京天气") as tracer:
        tracer.step(action="search", input="北京天气", output="晴天 25°C")
        tracer.step(action="finish", output="北京今天晴天")
    # 轨迹自动保存到 data/logs/traces/react_agent_20260414_153000.jsonl
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


# ============================================================
# Formatters
# ============================================================

class TextFormatter(logging.Formatter):
    """控制台用的纯文本格式，带时间戳和级别。"""

    LEVEL_TAGS = {
        logging.DEBUG: "DEBUG",
        logging.INFO: "INFO",
        logging.WARNING: " WARN",
        logging.ERROR: "ERROR",
    }

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        tag = self.LEVEL_TAGS.get(record.levelno, record.levelname)
        name = record.name.replace("nanoagent.", "")
        return f"[{ts}] [{tag}] [{name}] {record.getMessage()}"


class JSONFormatter(logging.Formatter):
    """文件用的 JSON Lines 格式，每行一条可解析记录。"""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # 把 extra 字段合并进来（排除 logging 内置字段）
        _BUILTIN = logging.LogRecord("", 0, "", 0, "", (), None).__dict__.keys()
        for k, v in record.__dict__.items():
            if k not in _BUILTIN and k not in entry:
                try:
                    json.dumps(v)  # 确保可序列化
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)
        return json.dumps(entry, ensure_ascii=False)


# ============================================================
# Logger 工厂
# ============================================================

_initialized = False


def _ensure_handlers() -> None:
    """首次调用时配置根 logger 的 handlers。"""
    global _initialized
    if _initialized:
        return
    _initialized = True

    root = logging.getLogger("nanoagent")
    root.setLevel(logging.DEBUG)
    root.propagate = False

    # 控制台 handler：INFO 及以上
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(TextFormatter())
    root.addHandler(console)

    # 文件 handler：DEBUG 及以上，JSON Lines
    from nanoagent.core.paths import data_dir
    log_dir = data_dir("logs")
    log_file = log_dir / f"{datetime.now().strftime('%Y%m%d')}.jsonl"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(JSONFormatter())
    root.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """获取框架日志器。

    Args:
        name: 模块名，如 "react_agent"、"memory"、"rag"。
              实际创建的 logger 名为 "nanoagent.{name}"。
    """
    _ensure_handlers()
    return logging.getLogger(f"nanoagent.{name}")


# ============================================================
# RunTracer — Agent 运行轨迹记录
# ============================================================

class RunTracer:
    """记录单次 Agent run 的完整轨迹。

    默认走 stream 模式：每一步立即写入文件并 flush，进程被杀也能保留现场。
    这是调试 Agent 卡住问题的关键——batch 模式只在 save() 调用时写入，
    卡在某一步时轨迹文件根本不存在。

    用法：
        with RunTracer("react_agent", user_input="北京天气") as tracer:
            tracer.step(action="search", input="北京天气", output="晴天")
            tracer.step(action="finish", output="最终答案")
    """

    def __init__(self, agent_name: str, user_input: str = "", stream: bool = True):
        self.agent_name = agent_name
        self.user_input = user_input
        self.steps: list[dict[str, Any]] = []
        self._start_time = time.time()
        self._step_start: Optional[float] = None
        self._stream = stream
        self._stream_file = None
        self._trace_path: Optional[Path] = None
        self._token_summary: Optional[dict] = None

        if stream:
            self._open_stream()

    def _open_stream(self) -> None:
        """在构造时就创建轨迹文件，写入 header 但 total_steps 先占位。"""
        from nanoagent.core.paths import data_dir
        today = datetime.now().strftime("%Y-%m-%d")
        trace_dir = data_dir("logs") / "traces" / today
        trace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        self._trace_path = trace_dir / f"{self.agent_name}_{ts}.jsonl"

        header = {
            "type": "run_header",
            "agent": self.agent_name,
            "user_input": self.user_input,
            "total_steps": "...",           # 运行中占位，save() 时不会重写
            "total_duration_ms": "...",
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(),
        }
        self._stream_file = open(self._trace_path, "w", encoding="utf-8")
        self._stream_file.write(json.dumps(header, ensure_ascii=False) + "\n")
        self._stream_file.flush()

    def step(self, **fields) -> None:
        """记录一个步骤。常用字段：action, input, output, tool, error。"""
        now = time.time()
        entry = {
            "step": len(self.steps) + 1,
            "timestamp": datetime.now().isoformat(),
            "duration_ms": round((now - (self._step_start or self._start_time)) * 1000),
            **fields,
        }
        self.steps.append(entry)
        self._step_start = now

        # 流式写入：立即落盘，卡住也能看见
        if self._stream_file:
            try:
                self._stream_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                self._stream_file.flush()
            except Exception:
                pass  # 日志失败不影响主流程

    def set_token_summary(self, data: dict) -> None:
        """注入 token 用量数据，save() 时写入 run_summary。"""
        self._token_summary = data

    def step_error(self, exc: BaseException, **extra) -> None:
        """记录一个带 evolution 预留字段的错误步骤。

        骨架位：Tracer 捕到异常时调 classify_error 填 4 个字段
        （is_error / error_class / is_transient / error_message）。
        **下游消费者为后续扩展预留**（ReflexionGate + consolidator），
        当前只写不读。

        Args:
            exc: 捕获到的异常
            **extra: 额外字段（如 context_tags / action / tool 等）
        """
        from nanoagent.core.error_classifier import classify_error
        error_class, is_transient = classify_error(exc)
        self.step(
            is_error=True,
            error_class=error_class,
            is_transient=is_transient,
            error_message=str(exc),
            error_type=type(exc).__name__,
            **extra,
        )

    def append_post_save(self, **fields) -> None:
        """save() 后追加事件到 trace JSONL（如 OutcomeTracker 的 outcome_update）。

        为什么需要：trace 在 user-turn 结束时 save() 关闭文件，但 OutcomeTracker
        在 save 之后才扫 trace 判 lesson outcome。lesson_helped/hurt/ineffective
        信号必须写回 trace 否则 grader / eval harness 永远看不见——self-evolution
        是否真有效就无法量化。

        语义：
        - step 单调递增（继续 self.steps 序列），同 lesson 维度可去重
        - 不进 self.steps 内存（避免影响 in-process 后续调用），仅按 len 计数
        - 文件不存在 / 路径丢失 → silent skip，不抛
        - 写失败 → silent skip（同 step()），不阻塞主流程
        """
        if self._trace_path is None or not self._trace_path.exists():
            return
        entry = {
            "step": len(self.steps) + 1,
            "timestamp": datetime.now().isoformat(),
            "action_phase": "post_save",
            **fields,
        }
        # 增长 self.steps 让多次调用 step 单调递增；不进 stream_file（已关闭）
        self.steps.append(entry)
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def save(self) -> Optional[Path]:
        """关闭轨迹文件。stream 模式下文件已在运行中写完。"""
        if not self.steps and not self._stream_file:
            return None

        total_ms = round((time.time() - self._start_time) * 1000)

        if self._stream_file:
            # 在文件末尾追加一个 summary 条目，便于事后定位完成状态
            try:
                summary = {
                    "type": "run_summary",
                    "total_steps": len(self.steps),
                    "total_duration_ms": total_ms,
                    "ended_at": datetime.now().isoformat(),
                }
                if self._token_summary:
                    summary["tokens"] = self._token_summary
                self._stream_file.write(json.dumps(summary, ensure_ascii=False) + "\n")
                self._stream_file.flush()
                self._stream_file.close()
            except Exception:
                pass
            logger = get_logger(self.agent_name)
            logger.info(f"轨迹已保存: {self._trace_path} ({len(self.steps)} 步, {total_ms}ms)")
            return self._trace_path

        # 非流式模式：兼容老行为，运行完一次性写
        from nanoagent.core.paths import data_dir
        today = datetime.now().strftime("%Y-%m-%d")
        trace_dir = data_dir("logs") / "traces" / today
        trace_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%H%M%S")
        trace_file = trace_dir / f"{self.agent_name}_{ts}.jsonl"

        header = {
            "type": "run_header",
            "agent": self.agent_name,
            "user_input": self.user_input,
            "total_steps": len(self.steps),
            "total_duration_ms": total_ms,
            "started_at": datetime.fromtimestamp(self._start_time).isoformat(),
        }
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(header, ensure_ascii=False) + "\n")
            for s in self.steps:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

        logger = get_logger(self.agent_name)
        logger.info(f"轨迹已保存: {trace_file} ({len(self.steps)} 步, {total_ms}ms)")
        return trace_file

    def __enter__(self) -> "RunTracer":
        return self

    def __exit__(self, *args) -> None:
        self.save()
