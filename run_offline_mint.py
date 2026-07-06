"""离线飞轮 mint 批跑入口（调度 / 手动，**非 inline 热路径**）。

扫最近若干 trace，跑 ReflectorMintRunner：
- grounded（在轨恢复验证门通过：同 op 失败后重执行成功）→ Reflector 产
  suggested_action → CANDIDATE lesson 入 backend（之后走 PromotionGate → 召回消费）。
- 信号缺失（无观测恢复）→ 视 EMIT_SOFT（默认关）写软旁路 jsonl（钉 CANDIDATE，
  不进 backend、不被召回）。

跑：.\\.venv\\Scripts\\python.exe run_offline_mint.py
改行为改顶部常量（不走 argparse / env，env 仅 API key / OPENAI_MODEL_ID）。
"""

import os
import sys
from pathlib import Path
from typing import List

from accrete.evolution.runtime_memory.sqlite_backend import (
    DEFAULT_DB_PATH as LESSONS_DB_PATH,
    SqliteMemoryBackend,
)

# ============================================================
# 运行参数（改这里就改行为）
# ============================================================

# Reflector 模型：诊断 + 打包已观测恢复（比主 agent 发明修复容易）。空 provider = openai。
REFLECTOR_MODEL: str = os.getenv("OPENAI_MODEL_ID", "gpt-5.4-mini")
REFLECTOR_PROVIDER: str = ""  # 可选 dashscope / zhipu / deepseek / ...

TRACES_DIR: Path = Path("data/logs/traces")
RECENT_TRACE_LIMIT: int = 50  # 跑最近 N 条 trace（按 mtime 倒序）

# 信号缺失那半默认关：开了才产软 suggest 写旁路 jsonl（永不进 backend / 召回）。
EMIT_SOFT: bool = False
SOFT_SINK_PATH: Path = Path("data/runtime/lessons/soft_suggests.jsonl")


def _recent_traces(traces_dir: Path, limit: int) -> List[Path]:
    """按 mtime 倒序取最近 limit 条 trace jsonl（data/logs/traces/<date>/*.jsonl）。"""
    if not traces_dir.exists():
        return []
    paths = sorted(
        traces_dir.glob("*/*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return paths[:limit]


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    from accrete.core.llm_client import LLMClient
    from accrete.evolution.runtime_memory.offline_mint import (
        ReflectorMintRunner,
        SoftSuggestSink,
    )
    from accrete.evolution.runtime_memory.reflector import Reflector

    traces = _recent_traces(TRACES_DIR, RECENT_TRACE_LIMIT)
    if not traces:
        print(f"[offline_mint] {TRACES_DIR} 下无 trace，无可 mint。")
        return 1
    print(f"[offline_mint] model={REFLECTOR_PROVIDER or 'openai'}/{REFLECTOR_MODEL} "
          f"traces={len(traces)} db={LESSONS_DB_PATH} emit_soft={EMIT_SOFT}")

    llm = LLMClient(
        model=REFLECTOR_MODEL,
        provider=REFLECTOR_PROVIDER or None,
        instance_name="reflector",
    )
    reflector = Reflector(llm)
    backend = SqliteMemoryBackend(db_path=LESSONS_DB_PATH)
    try:
        sink = SoftSuggestSink(SOFT_SINK_PATH) if EMIT_SOFT else None
        runner = ReflectorMintRunner(
            backend, reflector, soft_sink=sink, emit_soft=EMIT_SOFT,
        )
        report = runner.process_trace_paths(traces)
    finally:
        backend.close()

    print(
        f"[offline_mint] 完成：grounded 新建 {report.grounded_minted} / "
        f"extend {report.grounded_extended}；软旁路 {report.soft_emitted}；"
        f"跳过 {report.skipped}；弃权 {report.abstained}。({llm.usage})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
