"""nanoagent v2 Telegram Bot 入口 — channel 薄壳。

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
from nanoagent.core.llm_client import LLMClient
from nanoagent.evolution.reflexion import ReflexionStore
from nanoagent.evolution.skill_preference_audit import SkillPreferenceAuditWriter
from nanoagent.evolution.skill_preference_store import SkillPreferenceStore
from nanoagent.memory.user_facts import UserFacts
from nanoagent.runtime.evaluator import DigestEvaluator
from nanoagent.runtime.harness import Harness
from nanoagent.runtime.session import SessionStore
from nanoagent.runtime.telegram_channel import (
    TelegramChannel,
    make_bootstrap_session_key,
    make_session_key_prefix,
)
from nanoagent.skills.loader import SkillLoader

# ============================================================
# 运行参数
# ============================================================

TELEGRAM_BOT_TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
TELEGRAM_CHAT_ID_ENV = "TELEGRAM_CHAT_ID"  # 逗号分隔白名单


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
    evaluator: Optional[DigestEvaluator],
    lesson_retriever,
    outcome_tracker,
    lesson_ingestor,
    promotion_gate,
    promotion_audit_callback=None,
    distill_pipeline=None,
    preference_store=None,
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
            evaluator=evaluator,
            evaluator_max_retries=cli_main.EVALUATOR_MAX_RETRIES,
            outcome_tracker=outcome_tracker,
            lesson_ingestor=lesson_ingestor,
            promotion_gate=promotion_gate,
            promotion_audit_callback=promotion_audit_callback,
            session_key_prefix=prefix,
            reflexions_store=loader.reflexions_store,
            distill_pipeline=distill_pipeline,
            preference_store=preference_store,
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
    reflexions = ReflexionStore(cli_main.REFLEXIONS_DIR)
    preference_store = SkillPreferenceStore(cli_main.SKILL_PREFERENCES_PATH)
    preference_audit = SkillPreferenceAuditWriter(cli_main.SKILL_PREFERENCES_AUDIT_PATH)
    loader = SkillLoader(
        cli_main.SKILLS_DIR,
        reflexions_store=reflexions,
        preference_store=preference_store,
    )
    # Telegram channel 在单 worker ThreadPoolExecutor 跑 handle，但装配仍在
    # main thread；sqlite check_same_thread=False 让 connection 跨"装配 thread →
    # 单 worker thread"使用。多 worker 并发写不安全，靠 channel 单 worker 串行护住。
    lesson_retriever, outcome_tracker, lesson_ingestor, promotion_gate = \
        cli_main.build_runtime_memory(sqlite_check_same_thread=False)
    store = SessionStore(persist_dir=cli_main.SESSION_DIR)
    user_facts = UserFacts(cli_main.USER_FACTS_PATH)

    evaluator: Optional[DigestEvaluator] = None
    if os.getenv("DASHSCOPE_API_KEY"):
        eval_llm = LLMClient(
            model=cli_main.EVALUATOR_MODEL,
            provider=cli_main.EVALUATOR_PROVIDER,
            instance_name="evaluator",
            timeout=cli_main.EVALUATOR_TIMEOUT,
        )
        evaluator = DigestEvaluator(llm=eval_llm)
        print(
            f"[Evaluator] enabled: {cli_main.EVALUATOR_PROVIDER}/{cli_main.EVALUATOR_MODEL}"
        )

    promotion_audit_callback = (
        cli_main.JsonlAuditWriter(cli_main.PROMOTION_AUDIT_LOG_PATH)
        if promotion_gate is not None and cli_main.PROMOTION_AUDIT_LOG_PATH
        else None
    )

    distill_pipeline = cli_main.build_distill_pipeline(
        preference_store, preference_audit
    )
    if distill_pipeline is not None:
        print(
            f"[DistillPipeline] enabled: {cli_main.DISTILLER_PROVIDER}/"
            f"{cli_main.DISTILLER_MODEL}"
        )

    factory = build_harness_factory(
        store=store,
        loader=loader,
        user_facts=user_facts,
        evaluator=evaluator,
        lesson_retriever=lesson_retriever,
        outcome_tracker=outcome_tracker,
        lesson_ingestor=lesson_ingestor,
        promotion_gate=promotion_gate,
        promotion_audit_callback=promotion_audit_callback,
        distill_pipeline=distill_pipeline,
        preference_store=preference_store,
    )

    channel = TelegramChannel(
        token=token,
        allowed_chat_ids=allowed,
        harness_factory=factory,
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
