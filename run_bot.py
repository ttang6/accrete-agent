"""accrete v2 Telegram Bot 入口 — channel 薄壳。

复用 main.py 的装配 builder，每个白名单 chat 一个独立 Harness（避免 _current_skill /
_tracer 跨 chat 串扰），共享 SessionStore / SkillLoader / UserFacts / 各 lesson tracker
（这些都是 stateless 或线程安全；SQLite 写入由 sqlite3 默认序列化）。

运行：
    1. .env 设 TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID（逗号分隔白名单）
    2. uv pip install -e ".[telegram]"
    3. .\\.venv\\Scripts\\python.exe run_bot.py

改参数就改顶部常量；和 main.py 风格一致，不走 argparse。
"""

import asyncio
import os
import sys
from typing import Optional

from dotenv import load_dotenv

import main as cli_main
from accrete.core.llm_client import LLMClient
from accrete.memory.user_facts import UserFacts
from accrete.runtime.harness import Harness
from accrete.runtime.session import SessionStore
from accrete.runtime.telegram_channel import (
    TelegramChannel,
    make_bootstrap_session_key,
    make_session_key_prefix,
)
from accrete.skills.loader import SkillLoader

# ============================================================
# 运行参数
# ============================================================

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"  # 逗号分隔白名单
TELEGRAM_DIGEST_AT_ENV = "TELEGRAM_DIGEST_AT"  # "HH:MM" 开启进程内每日定时发布；不设则关


# ============================================================
# 装配
# ============================================================

def _parse_chat_ids(raw: Optional[str]) -> set[str]:
    if not raw:
        return set()
    return {x.strip() for x in raw.split(",") if x.strip()}


def build_harness_factory(
    *,
    store: SessionStore,
    loader: SkillLoader,
    user_facts: UserFacts,
    critic_llm,
    lesson_retriever,
    outcome_tracker,
    lesson_ingestor,
    global_memory=None,
    global_memory_distiller=None,
):
    """返回 (chat_id) -> Harness 的 closure。

    每次调用为该 chat 新建 MainLoop（避免 _tracer 跨 chat 串）+ Harness。
    下游 store / loader / user_facts / lesson 各 tracker 共享。
    """
    def factory(chat_id: str) -> Harness:
        loop = cli_main.build_loop(loader, lesson_retriever=lesson_retriever)
        bootstrap_key = make_bootstrap_session_key(chat_id)
        prefix = make_session_key_prefix(chat_id)
        return Harness(
            loop=loop,
            store=store,
            loader=loader,
            user_facts=user_facts,
            session_key=bootstrap_key,
            base_identity=cli_main.BASE_IDENTITY,
            critic_llm=critic_llm,
            critic_max_revise=cli_main.EVALUATOR_MAX_RETRIES,
            outcome_tracker=outcome_tracker,
            lesson_ingestor=lesson_ingestor,
            session_key_prefix=prefix,
            global_memory=global_memory,
            global_memory_distiller=global_memory_distiller,
            context_window_tokens=cli_main.CONTEXT_MAX_TOKENS,
            auto_distill_overflow_ratio=cli_main.AUTO_DISTILL_OVERFLOW_RATIO,
            auto_distill_switch_ratio=cli_main.AUTO_DISTILL_SWITCH_RATIO,
        )
    return factory


# ============================================================
# 入口
# ============================================================

async def run_bot() -> int:
    load_dotenv()

    token = os.getenv(TELEGRAM_BOT_TOKEN_ENV)
    if not token:
        print(f"[bot] 缺 {TELEGRAM_BOT_TOKEN_ENV}（请在 .env 或环境变量中设置），退出")
        return 1
    allowed = _parse_chat_ids(os.getenv(TELEGRAM_CHAT_ID_ENV))
    if not allowed:
        print(
            f"[bot] {TELEGRAM_CHAT_ID_ENV} 为空 → fail-fast 拒启动。"
            "请设逗号分隔的白名单 chat_id"
        )
        return 1

    # 共享下游服务（一次装配，多 chat 共用）
    loader = SkillLoader(cli_main.SKILLS_DIR)
    # Telegram channel 在单 worker ThreadPoolExecutor 跑 handle，但装配仍在
    # main thread；sqlite check_same_thread=False 让 connection 跨"装配 thread →
    # 单 worker thread"使用。多 worker 并发写不安全，靠 channel 单 worker 串行护住。
    lesson_retriever, outcome_tracker, lesson_ingestor = \
        cli_main.build_runtime_memory(sqlite_check_same_thread=False)
    store = SessionStore(persist_dir=cli_main.SESSION_DIR)
    user_facts = UserFacts(cli_main.USER_FACTS_PATH)
    # 全局自动记忆（Dream-lite）：多 chat 共用一份（同 user_facts 的单用户假设）。
    global_memory = cli_main.GlobalMemory(cli_main.AUTO_MEMORY_PATH)
    global_memory_distiller = (
        cli_main.GlobalMemoryDistiller(
            global_memory, cli_main.SubAgentRunner(llm_builder=cli_main.build_subagent_llm)
        )
        if os.getenv("DASHSCOPE_API_KEY") else None
    )

    critic_llm = None
    if os.getenv("DASHSCOPE_API_KEY"):
        critic_llm = LLMClient(
            model=cli_main.EVALUATOR_MODEL,
            provider=cli_main.EVALUATOR_PROVIDER,
            instance_name="critic",
            timeout=cli_main.EVALUATOR_TIMEOUT,
        )
        print(
            f"[Critic] enabled: {cli_main.EVALUATOR_PROVIDER}/{cli_main.EVALUATOR_MODEL}"
        )

    factory = build_harness_factory(
        store=store,
        loader=loader,
        user_facts=user_facts,
        critic_llm=critic_llm,
        lesson_retriever=lesson_retriever,
        outcome_tracker=outcome_tracker,
        lesson_ingestor=lesson_ingestor,
        global_memory=global_memory,
        global_memory_distiller=global_memory_distiller,
    )

    channel = TelegramChannel(
        token=token,
        allowed_chat_ids=allowed,
        harness_factory=factory,
        daily_publish_at=os.getenv(TELEGRAM_DIGEST_AT_ENV),
    )
    await channel.start()
    return 0


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
    if hasattr(sys.stdin, "reconfigure"):
        sys.stdin.reconfigure(encoding="utf-8", errors="replace")

    try:
        return asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\n[bot] 收到 Ctrl+C，退出。")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
