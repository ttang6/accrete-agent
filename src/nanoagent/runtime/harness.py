"""Channel-agnostic user-turn orchestrator."""

import hashlib
import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from nanoagent.core import trace_schema as ts
from nanoagent.core.metrics import metrics
from nanoagent.core.paths import data_dir
from nanoagent.memory.user_facts import UserFacts
from nanoagent.runtime.critic import Critic
from nanoagent.runtime.main_loop import MainLoop
from nanoagent.runtime.publish import (
    check_provenance,
    extract_adopted_items,
    mark_published_items,
)
from nanoagent.runtime.session import SessionStore
from nanoagent.runtime.token_counter import TokenCounter
from nanoagent.skills.loader import SkillLoader

if TYPE_CHECKING:
    from nanoagent.evolution.runtime_memory.lesson_ingestor import LessonIngestor
    from nanoagent.memory.global_memory import GlobalMemory
    from nanoagent.memory.global_memory_distiller import GlobalMemoryDistiller
    from nanoagent.evolution.runtime_memory.outcome_tracker import OutcomeTracker
    from nanoagent.evolution.runtime_memory.promotion_gate import (
        AuditCallback,
        PromotionGate,
    )

_logger = logging.getLogger("nanoagent.harness")

# 不能作为 /<skill> 快捷命令使用。
_RESERVED_SLASH_CMDS = frozenset({"clear", "new", "sessions", "resume", "skill", "skills", "profile", "digest"})

# /digest 发布日报时跑的 skill 与合成任务（发布是确定性命令入口，不靠自然语言猜意图）。
_DIGEST_SKILL = "ai-digest"
_DIGEST_TASK = "生成今日 AI 日报。"

# 单条 user message 截多长存进 session.meta.title
_TITLE_MAX_CHARS = 50

# /new 生成的 session_key 默认前缀（CLI channel 用 `cli:`）。
# 其他 channel（如 Telegram）通过构造参数 `session_key_prefix` 覆盖，
# 例如 `tg:<chat_id>:`，最终 key 形如 `tg:<chat_id>:<uuid8>`。
_DEFAULT_SESSION_PREFIX = "cli:"

# Session 级上下文体量软告警阈值（token）。历史滚到此值以上，每个 session
# 首次越过时在回复末尾附一句"建议 /new"。只量+提醒，不自动截断历史
# （硬底线仍是 HistoryManager.max_messages 按条数）。0 = 关闭软告警。
_SESSION_HISTORY_WARN_TOKENS_DEFAULT = 60_000


@dataclass
class HarnessResponse:
    """Response returned to the channel adapter."""
    kind: Literal["system", "assistant"]
    content: str
    skill: Optional[str] = None
    usage: Optional[str] = None


def build_system_prompt(
    base_identity: str,
    skill_body: Optional[str],
    user_facts_block: Optional[str],
    auto_memory_block: Optional[str] = None,
) -> str:
    """Compose system prompt sections, leaving datetime for MainLoop."""
    parts = [base_identity]
    if skill_body:
        parts.append(skill_body)
    if user_facts_block:
        parts.append(user_facts_block)
    if auto_memory_block:
        parts.append(auto_memory_block)
    parts.append("当前时间：{current_datetime}")
    return "\n\n".join(parts)


# 全局自动记忆（Dream-lite）触发参数：撑爆 / 切换 阈值（占 context window 比例）。
# 每个触发点一个 per-session 布尔、各最多触发一次——挡住"同一 session 反复进出反复蒸"。
_AUTO_DISTILL_OVERFLOW_RATIO = 0.9
_AUTO_DISTILL_SWITCH_RATIO = 0.5
_CONTEXT_WINDOW_TOKENS_DEFAULT = 80_000


class Harness:
    """Stateful orchestrator for one session."""

    def __init__(
        self,
        loop: MainLoop,
        store: SessionStore,
        loader: SkillLoader,
        user_facts: UserFacts,
        *,
        session_key: str,
        base_identity: str,
        digests_dir: Optional[Path] = None,
        critic: Optional[Critic] = None,
        critic_max_revise: int = 1,
        outcome_tracker: Optional["OutcomeTracker"] = None,
        lesson_ingestor: Optional["LessonIngestor"] = None,
        promotion_gate: Optional["PromotionGate"] = None,
        promotion_audit_callback: Optional["AuditCallback"] = None,
        session_key_prefix: str = _DEFAULT_SESSION_PREFIX,
        session_history_warn_tokens: int = _SESSION_HISTORY_WARN_TOKENS_DEFAULT,
        global_memory: Optional["GlobalMemory"] = None,
        global_memory_distiller: Optional["GlobalMemoryDistiller"] = None,
        context_window_tokens: int = _CONTEXT_WINDOW_TOKENS_DEFAULT,
        auto_distill_overflow_ratio: float = _AUTO_DISTILL_OVERFLOW_RATIO,
        auto_distill_switch_ratio: float = _AUTO_DISTILL_SWITCH_RATIO,
    ):
        self._loop = loop
        self._store = store
        self._loader = loader
        self._user_facts = user_facts
        self._session_key = session_key
        self._base_identity = base_identity
        self._digests_dir = digests_dir
        # critic：发布流程里的质量评审子 agent（非阻断、最多触发一轮 revise）。
        self._critic = critic
        self._critic_max_revise = critic_max_revise
        self._outcome_tracker = outcome_tracker
        # trace 写完后扫 failure 自动产 candidate lesson 入 backend
        # None 时退化到旧行为（仅手动 backfill 能造 candidate）
        self._lesson_ingestor = lesson_ingestor
        # 在 OutcomeTracker / Ingestor 之后扫 backend 自动 promote/retire
        self._promotion_gate = promotion_gate
        # 每次 sweep 命中转移时调一次（fail-open 由 sweep 兜底），
        # 默认 None 行为与之前一致。装配层（main.py）通常注入 JsonlAuditWriter
        # 把决策追加到 data/runtime/lessons/promotion_audit.jsonl。
        self._promotion_audit_callback = promotion_audit_callback
        self._session_key_prefix = session_key_prefix
        # Session 级 context budget（最小版，对齐 nanobot：只量+提醒、不自动截断）。
        # Harness 自持一份 TokenCounter（谁用谁持），不伸手进 MainLoop 的内部字段。
        self._token_counter = TokenCounter()
        self._session_history_warn_tokens = session_history_warn_tokens
        # 全局自动记忆（Dream-lite）：副 LLM 从对话蒸全局画像 → 独立文件 → 注入 system prompt。
        # 物理独立于 UserFacts（手动钉）。distiller=None（未注入 / 无 API key）→ 只读不更新。
        self._global_memory = global_memory
        self._global_memory_distiller = global_memory_distiller
        self._context_window_tokens = context_window_tokens
        self._auto_distill_overflow_ratio = auto_distill_overflow_ratio
        self._auto_distill_switch_ratio = auto_distill_switch_ratio
        self._current_skill = self._load_current_skill()
        # 启动后第一条对话消息默认开新 session——避免无意识续上次。
        # /resume / /new / /clear 任一显式动作都会把它置 False。
        # 已知边界：/skill alpha 紧接首条消息会丢失 alpha（新 session 的 meta
        # 里没 skill）；要保留请用 /new + /skill + 消息显式三步。
        self._fresh_session_pending = True

    @property
    def current_skill(self) -> Optional[str]:
        return self._current_skill

    def list_sessions(self) -> list[dict]:
        """Proxy SessionStore.list_sessions() — for channel layer to render."""
        return self._store.list_sessions()

    # ============================================================
    # 主入口
    # ============================================================

    def handle(self, text: str) -> HarnessResponse:
        text = text.strip()
        if not text:
            return self._sys("（空输入已忽略）")

        # 仅当首 token 是已注册命令 / skill 时才走斜杠命令路径；否则 fall through
        # 到对话。避免把 `/etc/passwd ...`、`/unknownword` 等合法消息吃成"未知命令"。
        # 有意取舍：typo 命令静默当对话消息（对齐 Claude Code 宽松 pass-through）。
        if text.startswith("/"):
            known = _RESERVED_SLASH_CMDS | set(self._loader.list_skills())
            parts = text[1:].split(maxsplit=1)   # 先取列表
            first = parts[0] if parts else ""    # 防 "/  " strip 后只剩 "/" → IndexError
            if first and "/" not in first and first in known:
                return self._handle_slash(text)

        return self._handle_dialogue(text)

    # ============================================================
    # 斜杠命令路由
    # ============================================================

    def _handle_slash(self, text: str) -> HarnessResponse:
        # /clear 和 /new 同义：开新 UUID8 session，旧 session 文件保留
        # （和 Claude Code 心智模型一致，比 destructive clear 安全）
        if text == "/clear" or text == "/new":
            return self._open_new_session()

        if text == "/sessions":
            return self._list_sessions()

        if text.startswith("/sessions "):
            arg = text.split(None, 1)[1].strip()
            if arg == "all":
                return self._list_sessions(limit=None)
            if arg.isdigit() and int(arg) > 0:
                return self._list_sessions(limit=int(arg))
            return self._sys("用法：/sessions [all | <n>]；默认显示最近 10 个")

        if text == "/resume":
            return self._sys("用法：/resume <session_key>；可先用 /sessions 查看可选 key")

        if text.startswith("/resume "):
            target = text.split(None, 1)[1].strip()
            return self._resume_session(target)

        if text == "/skills":
            body = "可用 skills：\n" + self._loader.get_descriptions()
            body += f"\n当前：{self._current_skill or '（base 无 skill）'}"
            return self._sys(body)

        if text == "/skill":
            return self._sys("用法：/skill <name> 切入；/skill none 回 base；/skills 列表")

        if text.startswith("/skill "):
            name = text.split(None, 1)[1].strip()
            if name == "none":
                self._current_skill = None
                self._store.set_meta(self._session_key, current_skill=None)
                return self._sys("已退出 skill 模式，回到 base。")
            if name not in self._loader.list_skills():
                return self._sys(f"未知 skill: {name}。用 /skills 查看可用")
            self._current_skill = name
            self._store.set_meta(self._session_key, current_skill=name)
            return self._sys(f"已切入 skill: {name}")

        if text == "/digest":
            return self._handle_digest()

        if text == "/profile" or text.startswith("/profile "):
            return self._handle_profile(text)

        # /<name> shortcut for /skill <name>
        if len(text) > 1 and " " not in text:
            name = text[1:]
            if name in _RESERVED_SLASH_CMDS:
                return self._sys(f"未知命令: {text}")
            if name in self._loader.list_skills():
                self._current_skill = name
                self._store.set_meta(self._session_key, current_skill=name)
                return self._sys(f"已切入 skill: {name}")
            return self._sys(f"未知命令或 skill: {text}。用 /skills 查看可用")

        return self._sys(f"未知命令: {text}")

    # ============================================================
    # Session 切换（/clear /new /sessions /resume）
    # ============================================================

    def _create_and_switch_to_new_session(self) -> str:
        """生成新 UUID8 + 切换 + materialize meta + 把 fresh_session_pending 置 False。

        显式 `/new` `/clear` 和首条消息隐式触发都走这里，区别只在 caller 是否
        包裹一条用户可见的系统消息。"""
        new_key = f"{self._session_key_prefix}{secrets.token_hex(4)}"
        while self._store.has(new_key):
            new_key = f"{self._session_key_prefix}{secrets.token_hex(4)}"
        self._switch_session(new_key)
        self._store.set_meta(
            new_key,
            last_used_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self._fresh_session_pending = False
        return new_key

    def _open_new_session(self) -> HarnessResponse:
        """显式 /new / /clear：创建并切换，返回用户可见的系统消息。"""
        # 切走当前 session 前：够厚（≥switch 阈值）就把这段对话蒸进全局记忆。
        self._maybe_auto_distill("switch", self._auto_distill_switch_ratio)
        new_key = self._create_and_switch_to_new_session()
        return self._sys(f"已开新会话：{new_key}（用 /sessions 查看历史会话）")

    def _create_and_switch_preserving_current_skill(self) -> str:
        """跟 _create_and_switch_to_new_session 同，但**保留**当前 _current_skill 跨 fork。

        用于隐式 fork 路径（首次 dialogue 触发 _fresh_session_pending）：
        如果用户在 fresh 状态下显式 /skill X 设置了 skill，他**期望**这个 skill
        跟着 dialogue 一起转入新 session——不是被静默丢弃。

        显式 /new / /clear 路径仍走 _create_and_switch_to_new_session（不保留
        skill）——用户主动开新会话表达"清空"意图。

        修复了隐式 fork 路径下显式设置的 skill 被静默丢弃的已知边界。
        """
        skill_to_preserve = self._current_skill
        new_key = self._create_and_switch_to_new_session()
        if skill_to_preserve is not None:
            self._current_skill = skill_to_preserve
            self._store.set_meta(new_key, current_skill=skill_to_preserve)
        return new_key

    def _resume_session(self, target_key: str) -> HarnessResponse:
        if not self._store.has(target_key):
            return self._sys(f"未找到 session: {target_key}。用 /sessions 查看可用")
        if target_key == self._session_key:
            self._fresh_session_pending = False  # 显式表态后续在此 session 继续
            return self._sys(f"当前已是 {target_key}")
        # 切走当前 session 前：够厚就蒸一次。
        self._maybe_auto_distill("switch", self._auto_distill_switch_ratio)
        self._switch_session(target_key)
        self._fresh_session_pending = False
        return self._sys(f"已切回 {target_key}")

    def _list_sessions(self, limit: Optional[int] = 10) -> HarnessResponse:
        """默认只列最近 limit 个；limit=None 列全部。

        历史会话能攒到上千条（实测 1700+），无上限直接刷屏。默认折叠到最近 10，
        `/sessions all` 展开、`/sessions <n>` 自定数量、再 `/sessions` 即收回。
        """
        rows = self._store.list_sessions()  # 已按 last_used_at 倒序
        if not rows:
            return self._sys("(暂无会话)")
        total = len(rows)
        shown = rows if limit is None else rows[:limit]
        lines = ["session 列表（按最近使用倒序）："]
        for r in shown:
            marker = "*" if r["key"] == self._session_key else " "
            title = r["title"] or "(no title)"
            last = r["last_used_at"] or "-"
            lines.append(
                f" {marker} [{r['key']}] turns={r['turn_count']} "
                f"last={last}  {title}"
            )
        if limit is not None and total > limit:
            lines.append(
                f"…仅显示最近 {limit}/{total} 个。"
                "/sessions all 看全部；/sessions <n> 看最近 n 个"
            )
        lines.append("用 /resume <key> 切回；/new 开新会话")
        return self._sys("\n".join(lines))

    def _switch_session(self, new_key: str) -> None:
        """切到目标 session_key 并刷新 _current_skill。"""
        self._session_key = new_key
        # 重新从新 session 的 meta 加载 skill（可能是 None）
        self._current_skill = self._load_current_skill()

    def _handle_profile(self, text: str) -> HarnessResponse:
        parts = text.split(None, 2)  # ["/profile"] / ["/profile", "set", "<k> <v>"]
        if len(parts) == 1:
            return self._sys(self._render_profile_view())

        sub = parts[1]
        if sub == "set":
            if len(parts) < 3:
                return self._sys("用法：/profile set <key> <value>")
            rest = parts[2].split(None, 1)
            if len(rest) < 2:
                return self._sys("用法：/profile set <key> <value>")
            key, value = rest
            try:
                self._user_facts.set(key, value, source="user_declared")
            except ValueError as e:
                return self._sys(f"写入失败：{e}")
            return self._sys(f"已写入 {key} = {value}")

        if sub == "delete":
            if len(parts) < 3:
                return self._sys("用法：/profile delete <key>")
            key = parts[2].strip()
            if self._user_facts.delete(key):
                return self._sys(f"已删除 {key}")
            return self._sys(f"键 {key} 不存在")

        return self._sys("用法：/profile [set <k> <v> | delete <k>]")

    def _render_profile_view(self) -> str:
        """展示全局 user facts（key: value + source）。"""
        all_facts = self._user_facts.get_all()
        lines: list[str] = []
        if all_facts:
            lines.append("当前用户画像：")
            lines.extend(
                f"  {k}: {entry['value']}  (source={entry['source']})"
                for k, entry in all_facts.items()
            )
        else:
            lines.append("（暂无用户画像。用 /profile set <key> <value> 写入）")
        return "\n".join(lines)

    # ============================================================
    # 对话路径
    # ============================================================

    def _handle_dialogue(self, text: str, *, publish: bool = False) -> HarnessResponse:
        # 启动后首条对话消息：默认开新 session，避免无意识续上次。
        # /resume / /new / /clear 已显式表态过则 _fresh_session_pending=False，跳过。
        # 保留 _current_skill 跨 fork：如果用户在 fresh 状态显式 /skill X，
        # 这条 dialogue 应该用该 skill 跑（之前会被静默丢掉，是已知边界）
        if self._fresh_session_pending:
            self._create_and_switch_preserving_current_skill()
        self._loop.system_prompt = self._compose_system_prompt()
        history = self._store.get(self._session_key)
        base_history = history.get_history() if history is not None else []
        turn_messages = base_history + [{"role": "user", "content": text}]

        # Session 记账基准：本轮跑前快照累计用量，turn 结束算 delta。
        # run_bot 多 chat 共享同一 llm 实例，但单 worker 串行执行 → 本轮"前→后"
        # 差值仍只反映本轮（不被其他 chat 干扰）。
        usage_before = self._loop.llm.usage.snapshot()

        # 发布轮可能触发一轮 critic revise（复用同一 tracer），故延后 trace 落盘。
        defer_save = publish and self._critic is not None
        answer = self._loop.run(
            messages=turn_messages,
            save_on_finish=not defer_save,
        )

        if defer_save:
            try:
                answer = self._maybe_critic_revise(answer, turn_messages)
            finally:
                self._loop.finalize_trace()

        # ingest 在 outcome 之前——新 candidate 写入与已有 lesson 的
        # stats 更新读写不冲突，但顺序上"先记新失败、再评估已用 lesson"更直观
        self._maybe_ingest_trace()
        self._maybe_consume_trace_outcomes()
        # 所有数据写入后扫 backend 决定 promote / retire
        self._maybe_run_promotion_gate()

        self._store.append_turn(self._session_key, text, answer)
        token_delta = self._loop.llm.usage.total_tokens - usage_before.total_tokens
        notice = self._update_session_meta_after_turn(text, token_delta)
        # 发布副作用只在发布轮跑：抽机器块条目 → 出处核对 → 自动 mark → archive。
        # 非发布轮（随手查）什么副作用都不跑，结构上杜绝污染去重历史。
        if publish:
            self._run_publish_sideeffects(answer)
        # 撑爆触发：本轮结束后若上下文 ≥overflow 阈值，趁老消息被截断前先蒸一次。
        self._maybe_auto_distill("overflow", self._auto_distill_overflow_ratio)
        content = answer if notice is None else f"{answer}\n\n{notice}"
        return HarnessResponse(
            kind="assistant",
            content=content,
            skill=self._current_skill,
            usage=str(self._loop.llm.usage),
        )

    def _update_session_meta_after_turn(self, user_text: str, token_delta: int) -> Optional[str]:
        """每 turn 结束更新 session.meta，并返回可选的"会话过长"软提示。

        三类更新合并到一次 set_meta（一次落盘）：
        - title 首次写入、last_used_at 每次刷新（原有逻辑）
        - session_tokens 累计用量：{total, turns}，跨重启持久（用量记账）
        - 上下文体量软告警：历史 token 体量首次越过阈值 → 标记 warned 并返回提示串。
          只量+提醒，不截断历史（硬底线仍是 HistoryManager.max_messages）。
        返回 None 表示本轮无提示。
        """
        existing = self._store.get_meta(self._session_key)
        updates: dict[str, object] = {
            "last_used_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        if not existing.get("title"):
            title = user_text.strip().replace("\n", " ")
            if len(title) > _TITLE_MAX_CHARS:
                title = title[:_TITLE_MAX_CHARS - 1] + "…"
            updates["title"] = title

        # 累计本 session token 用量（shallow-copy 后改，不动 store 里的原对象）
        acc = dict(existing.get("session_tokens") or {})
        acc["total"] = int(acc.get("total", 0)) + max(0, token_delta)
        acc["turns"] = int(acc.get("turns", 0)) + 1

        # 软告警：估当前历史体量，首次越过阈值发一次提示（warned 标志去重）
        notice: Optional[str] = None
        if self._session_history_warn_tokens > 0 and not acc.get("warned"):
            history = self._store.get(self._session_key)
            hist_tokens = (
                self._token_counter.count_messages(history.get_history())
                if history is not None else 0
            )
            if hist_tokens >= self._session_history_warn_tokens:
                acc["warned"] = True
                _logger.info(
                    f"[session-budget] {self._session_key} 历史 ~{hist_tokens} tokens "
                    f"≥ 阈值 {self._session_history_warn_tokens}，建议开新会话"
                )
                notice = "（本会话已较长，建议 /new 开新会话以保持响应质量）"

        updates["session_tokens"] = acc
        self._store.set_meta(self._session_key, **updates)
        return notice

    def _maybe_ingest_trace(self) -> None:
        """Trigger LessonIngestor on the just-finalized trace; fail-open."""
        if self._lesson_ingestor is None:
            return
        tracer = getattr(self._loop, "_tracer", None)
        if tracer is None:
            return
        trace_path = getattr(tracer, "_trace_path", None)
        if trace_path is None:
            return
        try:
            result = self._lesson_ingestor.process_trace_path(trace_path)
            if result.error:
                _logger.warning(
                    f"[LessonIngestor] error on {trace_path}: {result.error}"
                )
            elif result.skipped_reason in (None, "no_failures", "already_processed", "trace_not_found"):
                # 只在真有 lesson 写入时记 info，避免每个清白 trace 都打日志
                if result.total_lessons_touched > 0:
                    _logger.info(
                        f"[LessonIngestor] episode={result.episode_id} "
                        f"+{result.lessons_added} new / +{result.lessons_extended} extended"
                    )
        except Exception as e:
            _logger.warning(f"LessonIngestor 处理 trace 异常（已 fail-open）: {e}")

    def _maybe_run_promotion_gate(self) -> None:
        """扫 backend 全部 candidate / probation / promoted，按阈值规则自动转移状态。

        在 Ingestor + OutcomeTracker 之后跑——保证看到的是本 turn 完整数据。
        fail-open：异常 log warning 不阻塞 user-turn 主流程。

        **why turn 结束才 sweep**（设计意图，非 bug —— PromotionGate 在 turn
        边界 commit）：
        - same-turn 内 main_loop 多 iter 时 failure_memory.maybe_augment 在每个
          tool call 失败时已经查 backend；此时 lesson 必须**已是 PROMOTED 或
          PROBATION** 才能召回
        - 把 sweep 提前到 ingest 后立即跑也无济于事——maybe_augment 已经发生过了
        - 真正避免 race 的方式是依赖跨 turn 数据沉淀：Turn N 失败 ingest →
          turn 边界 sweep → Turn N+1 才能召回。这是飞轮的设计转速
        - 不要为 same-turn lesson recall 改系统。否则会把 memory 写入、eval、
          retry、promotion 搅成难 debug 的循环
        """
        if self._promotion_gate is None:
            return
        try:
            decisions = self._promotion_gate.sweep(
                audit_callback=self._promotion_audit_callback
            )
            if decisions:
                _logger.info(
                    f"[PromotionGate] sweep applied {len(decisions)}: "
                    + ", ".join(
                        f"{d.lesson_id}({d.from_status.value}→{d.to_status.value},{d.reason})"
                        for d in decisions
                    )
                )
        except Exception as e:
            _logger.warning(f"PromotionGate sweep 异常（已 fail-open）: {e}")

    def _maybe_consume_trace_outcomes(self) -> None:
        """Let OutcomeTracker consume the finalized trace, fail-open.

        每条 OutcomeUpdate 通过 RunTracer.append_post_save 写回 trace 末尾的
        ACTION_OUTCOME_UPDATE 事件，让 grader 能从 trace 抽
        lesson_helped/hurt/ineffective——这是飞轮"真改善"的可量化信号。
        """
        if self._outcome_tracker is None:
            return
        tracer = getattr(self._loop, "_tracer", None)
        if tracer is None:
            return
        trace_path = getattr(tracer, "_trace_path", None)
        if trace_path is None:
            return
        try:
            updates = self._outcome_tracker.process_trace_path(trace_path)
            if updates:
                # 把每条 outcome 追加到 trace 末尾——append_post_save 内部 fail-open
                for u in updates:
                    tracer.append_post_save(
                        action=ts.ACTION_OUTCOME_UPDATE,
                        lesson_id=u.lesson_id,
                        outcome=u.outcome.value,
                        new_confidence=u.new_confidence,
                        new_hit_count=u.new_hit_count,
                    )
                    # metric sink：飞轮 helped/hurt/ineffective 分桶（派生率交 metric 层
                    # 暴露、不重算——outcome_tracker 已算好 confidence）
                    metrics.incr("lesson_outcome_total", outcome=u.outcome.value)
                _logger.info(
                    f"[OutcomeTracker] 更新 {len(updates)} 条 lesson outcome: "
                    + ", ".join(f"{u.lesson_id}({u.outcome.value},conf={u.new_confidence:.2f})"
                                for u in updates)
                )
        except Exception as e:
            _logger.warning(f"OutcomeTracker 处理 trace 异常（已 fail-open）: {e}")

    def _handle_digest(self) -> HarnessResponse:
        """`/digest` 命令：确定性发布入口。把 ai-digest 设为本轮 skill，跑合成的
        "生成今日 AI 日报"任务并带 publish=True，触发发布流程。走命令而非自然语言
        识别——不让 harness 去猜"这句是不是要正式日报"（那会重开 lexical 那条老缝）。"""
        if _DIGEST_SKILL not in self._loader.list_skills():
            return self._sys(f"未安装 skill: {_DIGEST_SKILL}，无法发布日报。")
        self._current_skill = _DIGEST_SKILL
        self._store.set_meta(self._session_key, current_skill=_DIGEST_SKILL)
        return self._handle_dialogue(_DIGEST_TASK, publish=True)

    def _maybe_critic_revise(self, answer: str, base_history: list[dict]) -> str:
        """critic 审 + 至多一轮 revise。**非阻断**：判不合格也照常发布，verdict 只记录、
        不 gate。critic 是 reflection 的"挑错"半，不是收尾硬闸。"""
        if self._critic is None or self._critic_max_revise <= 0:
            return answer
        verdict = self._critic.review(answer)
        self._trace_critic(verdict)
        if verdict.passed or not verdict.critique:
            return answer
        # 一轮 revise（封顶）：注入 NL 批评，主 LLM 修订重出（含末尾机器块）。
        retry_messages = list(base_history) + [
            {"role": "assistant", "content": answer},
            {"role": "user", "content": (
                f"{verdict.critique}\n\n请据以上质量反馈修订日报后重新输出完整版"
                f"（含末尾的 adopted_items 机器块）。"
            )},
        ]
        try:
            return self._loop.run_continuation(
                messages=retry_messages, tracer=self._loop._tracer, save_on_finish=False,
            )
        except Exception as e:
            _logger.warning(f"critic revise 续跑异常（已 fail-open 用原稿）: {e}")
            return answer

    def _trace_critic(self, verdict) -> None:
        """把 critic verdict 记进 trace（仅观测，非阻断）。"""
        tracer = getattr(self._loop, "_tracer", None)
        if tracer is None:
            return
        try:
            tracer.step(
                action=ts.ACTION_EVALUATOR_CALL_END,
                passed=verdict.passed,
                critique=(verdict.critique or "")[:200],
            )
        except Exception:
            pass

    def _run_publish_sideeffects(self, answer: str) -> None:
        """发布副作用：抽机器块条目 → 出处核对（剔除编造）→ 自动写去重历史 → archive。
        全部 fail-open——已交付的日报不受登记/归档失败影响。"""
        turn_ctx = getattr(self._loop, "_turn_ctx", None)
        candidates = list(turn_ctx.candidates) if turn_ctx is not None else []
        items = extract_adopted_items(answer)
        kept, dropped = check_provenance(items, candidates)
        if dropped:
            _logger.warning(
                f"[publish] 出处核对剔除 {len(dropped)} 条凭空条目（不在本轮候选里）："
                + ", ".join(d["fingerprint"] for d in dropped)
            )
        written = mark_published_items(kept)
        _logger.info(f"[publish] 去重历史新写入 {written} 条（本期采用 {len(kept)}）")
        self._archive_published(answer)

    def _archive_published(self, answer: str) -> None:
        """发布即落盘 digests/——archive 是 publish 的副作用，不再靠 answer 命中 markers。

        文件名 = 日期 + answer 内容哈希：同一篇日报重复发布（如发布流程重试）落到
        同名文件、覆盖而非堆叠，保证 archive 幂等（不重复落盘）。内容不同则各自一份。
        """
        now = datetime.now()
        target_dir = self._digests_dir if self._digests_dir is not None else data_dir("digests")
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            digest_hash = hashlib.sha1(answer.encode("utf-8")).hexdigest()[:8]
            path = target_dir / f"{now.strftime('%Y-%m-%d')}_{digest_hash}.md"
            header = (
                f"<!-- session_key: {self._session_key} | "
                f"skill: {self._current_skill or '(auto)'} | "
                f"published_at: {now.isoformat(timespec='seconds')} -->\n\n"
            )
            path.write_text(header + answer, encoding="utf-8")
        except OSError:
            pass

    def _maybe_auto_distill(self, reason: str, ratio: float) -> None:
        """全局自动记忆触发：当前 session 上下文 ≥ ratio×窗口，且该触发点本 session 还没
        蒸过 → 副 LLM 把当前 session 对话蒸进全局长期记忆（独立文件）。fail-open。

        每个触发点（reason ∈ {overflow, switch}）一个 per-session 布尔、各最多触发一次——
        挡住"同一 session 反复进出反复蒸"。distiller=None → 静默跳过。
        已知局限（保守 v1）：触发后该 session 再增长的对话本轮不补蒸；填不满又没切走则不蒸
        （nanobot Dream 同款、有意取舍）。
        """
        if self._global_memory_distiller is None:
            return
        flag_key = f"auto_distilled_{reason}"
        if self._store.get_meta(self._session_key).get(flag_key):
            return  # 该触发点本 session 已蒸过
        history = self._store.get(self._session_key)
        if history is None:
            return
        msgs = history.get_history()
        if not msgs:
            return
        hist_tokens = self._token_counter.count_messages(msgs)
        if hist_tokens < ratio * self._context_window_tokens:
            return
        try:
            self._global_memory_distiller.distill(msgs)
        except Exception as e:
            _logger.warning(f"[global-memory] {reason} 蒸馏异常（已 fail-open）: {e}")
        # 不论成败都置位——best-effort，避免反复重蒸同一 session。
        self._store.set_meta(self._session_key, **{flag_key: True})

    def _compose_system_prompt(self) -> str:
        skill_body = self._loader.render(self._current_skill) if self._current_skill else None
        facts_block = self._user_facts.render_block() or None
        auto_block = (self._global_memory.render_block() or None) if self._global_memory else None
        return build_system_prompt(self._base_identity, skill_body, facts_block, auto_block)

    def _load_current_skill(self) -> Optional[str]:
        """Restore pinned skill from session metadata."""
        skill = self._store.get_meta(self._session_key).get("current_skill")
        if not isinstance(skill, str) or not skill.strip():
            return None
        if skill not in self._loader.list_skills():
            return None
        return skill

    # ============================================================
    # 内部便利
    # ============================================================

    def _sys(self, content: str) -> HarnessResponse:
        return HarnessResponse(kind="system", content=content, skill=self._current_skill)
