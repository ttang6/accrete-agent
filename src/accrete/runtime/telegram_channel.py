"""Telegram channel 薄壳 — 把 chat 文本路由到 Harness。

定位（channel 薄壳，调 harness.handle）：
- 不持任何对话状态——状态全在 Harness / SessionStore
- 每个白名单 chat_id 对应一个独立 Harness 实例（多 chat 不互相污染 _current_skill）
- session_key 形如 `tg:<chat_id>:<uuid8>`，首句自动开新（复用 Harness._fresh_session_pending）
- 砍掉 v1 业务功能（feedback button / HITL / aifeedback / detect / profile_clear UI）
  ——这些都是产品层逻辑，当前不需要

核心机制：
- 双 HTTPXRequest 连接池：长轮询不卡发送
- typing indicator 循环（4 秒刷新，Telegram 5 秒过期）
- Markdown 失败自动降级纯文本（BadRequest 'can't parse'）
- 长消息按 `\\n## ` section header 优先切块（日报场景天然分段）
- RetryAfter / TimedOut 指数退避

故意不做：
- 主动推送（scheduler 范畴的后续扩展）
- inline keyboard / callback button（产品层反馈，现在没机制消费）
- 流式编辑消息（MainLoop 同步，无流式）
"""

from __future__ import annotations

import asyncio
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional

from telegram import ReactionTypeEmoji, Update
from telegram.error import BadRequest, RetryAfter, TimedOut
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from accrete.runtime.harness import Harness

_logger = logging.getLogger("accrete.telegram_channel")

_MAX_MESSAGE_CHARS = 4000
_TYPING_INTERVAL = 4
_SEND_MAX_RETRIES = 3
_SEND_RETRY_BASE = 0.5

# 定时发布走和 /digest 完全相同的确定性入口（harness 已实现），不另起一套发布逻辑
_DIGEST_COMMAND = "/digest"

# 同 Harness 的命名约定：channel 前缀 + chat_id + uuid8 三段
_TG_PREFIX = "tg:"


def _parse_hhmm(value: Optional[str]) -> Optional[tuple[int, int]]:
    """"HH:MM" → (h, m)；None/空 → None（关闭定时）。格式错即 fail-fast。"""
    if not value:
        return None
    try:
        hh, mm = value.strip().split(":")
        h, m = int(hh), int(mm)
    except (ValueError, AttributeError):
        raise ValueError(f"daily_publish_at 格式应为 HH:MM，收到: {value!r}")
    if not (0 <= h < 24 and 0 <= m < 60):
        raise ValueError(f"daily_publish_at 超出范围: {value!r}")
    return (h, m)

# `(chat_id) -> Harness`：channel 不知道下游装配（loop / store / loader / ...），
# run_bot.py 注入 closure。lazy 创建：第一条消息时才 build。
HarnessFactory = Callable[[str], Harness]


class TelegramChannel:
    """每个白名单 chat 一个 Harness 实例的薄壳路由。"""

    def __init__(
        self,
        token: str,
        allowed_chat_ids: set[str],
        harness_factory: HarnessFactory,
        daily_publish_at: Optional[str] = None,
    ):
        if not token:
            raise ValueError("TelegramChannel: token 不能为空")
        if not allowed_chat_ids:
            raise ValueError(
                "TelegramChannel: allowed_chat_ids 为空 → fail-fast 拒启动。"
                "请设 TELEGRAM_CHAT_ID 环境变量（逗号分隔）"
            )
        self._token = token
        self._allowed = {str(cid).strip() for cid in allowed_chat_ids if str(cid).strip()}
        self._factory = harness_factory
        self._harnesses: dict[str, Harness] = {}
        self._app: Optional[Application] = None
        self._typing_tasks: dict[int, asyncio.Task] = {}
        self._running = False
        # 单 worker 线程池：harness.handle 全部跑在同一 thread，匹配
        # SqliteMemoryBackend 的 sqlite3 默认 check_same_thread=True 约束。
        # 单 owner 场景下多 chat 串行处理影响极小，换来 OutcomeTracker /
        # LessonIngestor 的 SQLite 句柄不跨线程出错。
        self._handler_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="accrete-handler"
        )
        # 进程内每日定时发布（in-process scheduler）。daily_publish_at="HH:MM" 开启，
        # None 关闭（向后兼容，不传就没有定时）。设计取舍：计算"下一个 HH:MM 的未来
        # 时刻"再 sleep —— 天然 at-most-once-per-day，重启不重发，无需额外落盘记录。
        # 跑在 bot 自己的 event loop 里，复用交互路径同一套 harness / 单 worker 线程池 /
        # 发送机制（不另起 PTB Application、不靠外部 OS cron）。
        self._publish_hm = _parse_hhmm(daily_publish_at)
        self._scheduler_task: Optional[asyncio.Task] = None

    # ------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------

    async def start(self) -> None:
        """启动 bot 并阻塞运行直到 stop() 或 KeyboardInterrupt。"""
        api_request = HTTPXRequest(connection_pool_size=16, read_timeout=30, connect_timeout=30)
        poll_request = HTTPXRequest(connection_pool_size=4, read_timeout=30, connect_timeout=30)

        self._app = (
            Application.builder()
            .token(self._token)
            .request(api_request)
            .get_updates_request(poll_request)
            .build()
        )

        # /start /help 单独走 CommandHandler 给 telegram-标准体验；
        # 其他 / 命令（/skill /skills /sessions /resume /new /profile）走 MessageHandler
        # 透传给 harness——harness 已支持完整 slash 路由
        self._app.add_handler(CommandHandler(["start", "help"], self._on_help))
        self._app.add_handler(MessageHandler(filters.TEXT, self._on_message))
        self._app.add_error_handler(self._on_error)

        await self._app.initialize()
        await self._app.start()
        info = await self._app.bot.get_me()
        _logger.info(
            f"[tg] 连接成功 @{info.username} (id={info.id}); "
            f"白名单 chat_id={','.join(sorted(self._allowed))}"
        )

        await self._app.updater.start_polling(allowed_updates=["message"])
        self._running = True
        if self._publish_hm is not None:
            self._scheduler_task = asyncio.create_task(self._scheduler_loop())
            _logger.info(
                f"[tg] 进程内定时发布已启用 @ "
                f"{self._publish_hm[0]:02d}:{self._publish_hm[1]:02d}"
            )
        try:
            while self._running:
                await asyncio.sleep(1)
        finally:
            await self._shutdown()

    async def stop(self) -> None:
        self._running = False

    async def _shutdown(self) -> None:
        if self._scheduler_task is not None:
            self._scheduler_task.cancel()
        for task in list(self._typing_tasks.values()):
            task.cancel()
        self._typing_tasks.clear()
        if self._app is not None:
            _logger.info("[tg] 停止中...")
            try:
                await self._app.updater.stop()
                await self._app.stop()
                await self._app.shutdown()
            except Exception as e:
                _logger.warning(f"[tg] 关停异常: {e}")
        # 关 single-worker executor；wait=True 让在跑的 handle 自然结束，
        # 避免 SqliteBackend 写入半截
        self._handler_executor.shutdown(wait=True)

    # ------------------------------------------------------------
    # 进程内每日定时发布（in-process scheduler）
    # ------------------------------------------------------------

    def _seconds_until_next_publish(self) -> float:
        """距下一个 HH:MM 未来时刻的秒数。已过今天的点 → 算到明天。"""
        from datetime import datetime, timedelta

        h, m = self._publish_hm
        now = datetime.now()
        target = now.replace(hour=h, minute=m, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return (target - now).total_seconds()

    async def _scheduler_loop(self) -> None:
        """到点对每个白名单 chat 跑 /digest 并推送。计算未来时刻再 sleep —— 天然
        at-most-once-per-day；发布耗时结束后重算的下一个点必在次日，不会重发。"""
        while self._running:
            try:
                await asyncio.sleep(self._seconds_until_next_publish())
            except asyncio.CancelledError:
                return
            if not self._running:
                return
            await self._run_scheduled_publish()

    async def _run_scheduled_publish(self) -> None:
        """复用交互路径同一套机制：每 chat 的 harness 跑 /digest（单 worker 线程池，
        与用户消息串行、不跨线程碰 SQLite）→ _send_text 推送。单 chat 失败不影响其余。"""
        event_loop = asyncio.get_running_loop()
        for chat_id in sorted(self._allowed):
            try:
                harness = self._get_harness(chat_id)
                response = await event_loop.run_in_executor(
                    self._handler_executor, harness.handle, _DIGEST_COMMAND
                )
                await self._send_text(int(chat_id), response.content or "（空日报）")
                _logger.info(f"[tg] 定时日报已推送 chat={chat_id}")
            except Exception as e:  # 单 chat 失败不影响其余
                _logger.error(
                    f"[tg] 定时日报失败 chat={chat_id}: {type(e).__name__}: {e}"
                )

    # ------------------------------------------------------------
    # 路由
    # ------------------------------------------------------------

    def _is_allowed(self, chat_id: int) -> bool:
        return str(chat_id) in self._allowed

    def _get_harness(self, chat_id: str) -> Harness:
        """Lazy 创建：每个 chat 一个 Harness。session_key 用 bootstrap 占位
        （不会落盘），首条消息触发 _fresh_session_pending → 生成 tg:<chat>:<uuid8>。"""
        h = self._harnesses.get(chat_id)
        if h is None:
            h = self._factory(chat_id)
            self._harnesses[chat_id] = h
        return h

    async def _on_help(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.message:
            return
        chat_id = update.effective_chat.id
        if not self._is_allowed(chat_id):
            return
        await update.message.reply_text(
            "你好！我是 accrete。\n\n"
            "• 直接发消息提问\n"
            "• /skills 列出可用 skill；/skill <name> 切入；/skill none 退出\n"
            "• /sessions 列出历史会话；/resume <key> 切回；/new 开新会话\n"
            "• /profile 查看用户画像；/profile set <k> <v> 写入；/profile delete <k>\n"
            "• /help 看这条说明"
        )

    async def _on_message(self, update: Update, _: ContextTypes.DEFAULT_TYPE) -> None:
        if not update.effective_chat or not update.message:
            return
        chat_id = update.effective_chat.id
        message_id = update.message.message_id
        text = (update.message.text or "").strip()
        if not text:
            return
        if not self._is_allowed(chat_id):
            _logger.warning(f"[tg] 拒非白名单 chat_id={chat_id}: {text[:40]}")
            return

        _logger.info(f"[tg] 收到 chat={chat_id}: {text[:60]}")
        await self._add_reaction(chat_id, message_id, "👀")
        typing_task = asyncio.create_task(self._typing_loop(chat_id))
        self._typing_tasks[chat_id] = typing_task

        try:
            harness = self._get_harness(str(chat_id))
            # Harness.handle 是同步的（loop.run 内部同步），扔单 worker 线程池跑——
            # 既不阻塞 asyncio 主循环，又保证 SqliteMemoryBackend 的句柄始终在同一
            # thread（SQLite 默认 check_same_thread=True；跨 thread 用会抛
            # ProgrammingError → 飞轮 backend 读写静默失效）
            event_loop = asyncio.get_running_loop()
            response = await event_loop.run_in_executor(
                self._handler_executor, harness.handle, text
            )
            reply = response.content or "（空回复）"
            if response.usage:
                # 尾部加 usage 行——和 CLI 体验一致
                reply = f"{reply}\n\n_({response.usage})_"
        except Exception as e:
            _logger.error(f"[tg] handler 异常 chat={chat_id}: {type(e).__name__}: {e}")
            reply = f"❌ 处理失败: {type(e).__name__}: {e}"
        finally:
            typing_task.cancel()
            self._typing_tasks.pop(chat_id, None)
            await self._remove_reaction(chat_id, message_id)

        try:
            await self._send_text(chat_id, reply)
        except Exception as e:
            _logger.error(f"[tg] 发送失败 chat={chat_id}: {e}")

    async def _on_error(self, _: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        err = context.error
        if isinstance(err, (TimedOut, RetryAfter)):
            _logger.warning(f"[tg] 网络问题: {err}")
        else:
            _logger.error(f"[tg] 未预期错误: {type(err).__name__}: {err}")

    # ------------------------------------------------------------
    # UX：reaction + typing
    # ------------------------------------------------------------

    async def _add_reaction(self, chat_id: int, message_id: int, emoji: str) -> None:
        try:
            await self._app.bot.set_message_reaction(
                chat_id=chat_id, message_id=message_id,
                reaction=[ReactionTypeEmoji(emoji=emoji)],
            )
        except Exception as e:
            _logger.debug(f"[tg] reaction 失败（已忽略）: {e}")

    async def _remove_reaction(self, chat_id: int, message_id: int) -> None:
        try:
            await self._app.bot.set_message_reaction(
                chat_id=chat_id, message_id=message_id, reaction=[],
            )
        except Exception:
            pass

    async def _typing_loop(self, chat_id: int) -> None:
        try:
            while True:
                try:
                    await self._app.bot.send_chat_action(chat_id=chat_id, action="typing")
                except Exception:
                    pass
                await asyncio.sleep(_TYPING_INTERVAL)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------
    # 发送：分块 + Markdown 降级 + 退避
    # ------------------------------------------------------------

    async def _send_text(self, chat_id: int, text: str) -> None:
        for chunk in self._split(text, _MAX_MESSAGE_CHARS):
            await self._call_with_retry(self._send_one_chunk, chat_id, chunk)

    async def _send_one_chunk(self, chat_id: int, text: str) -> None:
        try:
            await self._app.bot.send_message(
                chat_id=chat_id,
                text=text,
                parse_mode="Markdown",
                disable_web_page_preview=True,
            )
            return
        except BadRequest as e:
            desc = str(e).lower()
            if "can't parse" in desc or "entity" in desc:
                _logger.warning(f"[tg] Markdown 解析失败，降级纯文本: {e}")
                await self._app.bot.send_message(
                    chat_id=chat_id, text=text,
                    disable_web_page_preview=True,
                )
                return
            raise

    async def _call_with_retry(self, fn, *args, **kwargs):
        for attempt in range(1, _SEND_MAX_RETRIES + 1):
            try:
                return await fn(*args, **kwargs)
            except RetryAfter as e:
                delay = float(e.retry_after)
                _logger.warning(
                    f"[tg] 限流，等 {delay}s (尝试 {attempt}/{_SEND_MAX_RETRIES})"
                )
                if attempt == _SEND_MAX_RETRIES:
                    raise
                await asyncio.sleep(delay)
            except TimedOut:
                delay = _SEND_RETRY_BASE * (2 ** (attempt - 1))
                _logger.warning(
                    f"[tg] 超时，退避 {delay}s (尝试 {attempt}/{_SEND_MAX_RETRIES})"
                )
                if attempt == _SEND_MAX_RETRIES:
                    raise
                await asyncio.sleep(delay)

    @staticmethod
    def _split(text: str, max_chars: int) -> list[str]:
        """按 `\\n## ` Markdown header 优先切；超长段再字数硬切兜底。

        日报场景下天然形成「概览 / 论文 / 项目 / 动态」多条独立消息，体验好。
        非日报纯回复无 ## 时整段走兜底逻辑。
        """
        sections = [s.strip() for s in re.split(r"\n(?=## )", text) if s.strip()]
        if not sections:
            return []
        chunks: list[str] = []
        for section in sections:
            if len(section) <= max_chars:
                chunks.append(section)
            else:
                chunks.extend(TelegramChannel._split_by_chars(section, max_chars))
        return chunks

    @staticmethod
    def _split_by_chars(text: str, max_chars: int) -> list[str]:
        if len(text) <= max_chars:
            return [text]
        chunks: list[str] = []
        remaining = text
        while len(remaining) > max_chars:
            window = remaining[:max_chars]
            split_at = window.rfind("\n\n")
            if split_at < max_chars // 2:
                split_at = window.rfind("\n")
            if split_at < max_chars // 2:
                split_at = max_chars
            chunk = remaining[:split_at].rstrip()
            if chunk:
                chunks.append(chunk)
            remaining = remaining[split_at:].lstrip()
        if remaining:
            chunks.append(remaining)
        return chunks


def make_bootstrap_session_key(chat_id: str) -> str:
    """Bootstrap session_key for harness construction.

    Harness._fresh_session_pending=True 会让首条消息立即调
    _create_and_switch_to_new_session()——bootstrap key 不写入 SessionStore。
    """
    return f"{_TG_PREFIX}{chat_id}:bootstrap"


def make_session_key_prefix(chat_id: str) -> str:
    """Per-chat prefix：harness 生成新 session 时形如 `tg:<chat_id>:<uuid8>`。"""
    return f"{_TG_PREFIX}{chat_id}:"
