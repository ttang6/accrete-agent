"""Channel-agnostic user-turn orchestrator."""

import logging
import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from nanoagent.core import trace_schema as ts
from nanoagent.core.paths import data_dir
from nanoagent.memory.user_facts import UserFacts
from nanoagent.runtime.evaluator import DigestEvaluator
from nanoagent.runtime.main_loop import MainLoop
from nanoagent.runtime.session import SessionStore
from nanoagent.runtime.token_counter import TokenCounter
from nanoagent.skills.loader import SkillLoader

if TYPE_CHECKING:
    from nanoagent.evolution.preference_pipeline.pipeline import PreferenceDistillPipeline
    from nanoagent.evolution.reflexion import ReflexionStore
    from nanoagent.evolution.runtime_memory.lesson_ingestor import LessonIngestor
    from nanoagent.evolution.runtime_memory.outcome_tracker import OutcomeTracker
    from nanoagent.evolution.runtime_memory.promotion_gate import (
        AuditCallback,
        PromotionGate,
    )
    from nanoagent.evolution.skill_preference_store import SkillPreferenceStore

_logger = logging.getLogger("nanoagent.harness")

# 不能作为 /<skill> 快捷命令使用。
_RESERVED_SLASH_CMDS = frozenset({"clear", "new", "sessions", "resume", "skill", "skills", "profile", "feedback"})

# /feedback 单 skill 上限，超出 FIFO 丢最旧——避免 SKILL.md 前置块无限膨胀。
_FEEDBACK_MAX_PER_SKILL = 20

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
) -> str:
    """Compose system prompt sections, leaving datetime for MainLoop."""
    parts = [base_identity]
    if skill_body:
        parts.append(skill_body)
    if user_facts_block:
        parts.append(user_facts_block)
    parts.append("当前时间：{current_datetime}")
    return "\n\n".join(parts)


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
        evaluator: Optional[DigestEvaluator] = None,
        evaluator_max_retries: int = 1,
        outcome_tracker: Optional["OutcomeTracker"] = None,
        lesson_ingestor: Optional["LessonIngestor"] = None,
        promotion_gate: Optional["PromotionGate"] = None,
        promotion_audit_callback: Optional["AuditCallback"] = None,
        session_key_prefix: str = _DEFAULT_SESSION_PREFIX,
        reflexions_store: Optional["ReflexionStore"] = None,
        distill_pipeline: Optional["PreferenceDistillPipeline"] = None,
        preference_store: Optional["SkillPreferenceStore"] = None,
        session_history_warn_tokens: int = _SESSION_HISTORY_WARN_TOKENS_DEFAULT,
    ):
        self._loop = loop
        self._store = store
        self._loader = loader
        self._user_facts = user_facts
        self._session_key = session_key
        self._base_identity = base_identity
        self._digests_dir = digests_dir
        self._evaluator = evaluator
        self._evaluator_max_retries = evaluator_max_retries
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
        # HITL feedback 通道：用户用 /feedback 显式声明的 skill 偏好（行为 / 风格 /
        # 隐性约束），写到 ReflexionStore 的 skill scope，渲染时由 SkillLoader
        # 注入到 SKILL.md 前置 `# 历史教训` 块。跟飞轮（trace-level lesson）
        # 不重叠：飞轮学 tool 协议失败修复，feedback 学 skill 行为偏好。
        self._reflexions = reflexions_store
        # turn_end / skill_switch / session_end 触发偏好蒸馏，沉淀 NL preference summary。
        # store 单独传一份给 /profile skill-prefs 子命令读 + 用户手动 delete。
        # 装配时未注入 → 全套 hook 静默跳过（向后兼容）。
        self._distill_pipeline = distill_pipeline
        self._preference_store = preference_store
        # Session 级 context budget（最小版，对齐 nanobot：只量+提醒、不自动截断）。
        # Harness 自持一份 TokenCounter（谁用谁持），不伸手进 MainLoop 的内部字段。
        self._token_counter = TokenCounter()
        self._session_history_warn_tokens = session_history_warn_tokens
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
                self._maybe_distill(event="skill_switch")
                self._current_skill = None
                self._store.set_meta(self._session_key, current_skill=None)
                return self._sys("已退出 skill 模式，回到 base。")
            if name not in self._loader.list_skills():
                return self._sys(f"未知 skill: {name}。用 /skills 查看可用")
            # 切走前先对**旧** skill 做一次蒸馏（信号在该 skill 上下文里最准）
            if self._current_skill and self._current_skill != name:
                self._maybe_distill(event="skill_switch")
            self._current_skill = name
            self._store.set_meta(self._session_key, current_skill=name)
            return self._sys(f"已切入 skill: {name}")

        if text == "/profile" or text.startswith("/profile "):
            return self._handle_profile(text)

        if text == "/feedback" or text.startswith("/feedback "):
            return self._handle_feedback(text)

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
        # 切走当前 session 前先把**当前** skill 的对话蒸一次——
        # session_end 等价于"该 skill 的当前一段对话告一段落"，是稳定信号点
        self._maybe_distill(event="session_end")
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
        self._switch_session(target_key)
        self._fresh_session_pending = False
        return self._sys(f"已切回 {target_key}")

    def _list_sessions(self) -> HarnessResponse:
        rows = self._store.list_sessions()
        if not rows:
            return self._sys("(暂无会话)")
        lines = ["session 列表（按最近使用倒序）："]
        for r in rows:
            marker = "*" if r["key"] == self._session_key else " "
            title = r["title"] or "(no title)"
            last = r["last_used_at"] or "-"
            lines.append(
                f" {marker} [{r['key']}] turns={r['turn_count']} "
                f"last={last}  {title}"
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
        if sub == "skill-prefs":
            rest = parts[2] if len(parts) > 2 else ""
            return self._handle_skill_prefs(rest.strip())

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

        return self._sys("用法：/profile [set <k> <v> | delete <k> | skill-prefs ...]")

    def _render_profile_view(self) -> str:
        """统一展示 user facts + distilled skill preferences（物理分开、视图统一）。"""
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

        if self._preference_store is not None and len(self._preference_store) > 0:
            lines.append("")
            lines.append("Auto-distilled skill preferences:")
            for skill in self._preference_store.list_skills():
                pref = self._preference_store.get(skill)
                if pref is None:
                    continue
                date = pref.updated_at[:10] if pref.updated_at else "-"
                lines.append(
                    f"  {skill} [updated {date}, confidence={pref.confidence}]:"
                )
                lines.append(f"    {pref.value}")
            lines.append(
                "用 /profile skill-prefs delete <skill> 删单个；"
                "/profile skill-prefs clear 全清"
            )
        return "\n".join(lines)

    def _handle_skill_prefs(self, rest: str) -> HarnessResponse:
        """/profile skill-prefs [list | delete <skill> | clear]"""
        if self._preference_store is None:
            return self._sys("（skill preferences 未启用：装配时未注入 preference_store）")

        if not rest or rest == "list":
            store = self._preference_store
            if len(store) == 0:
                return self._sys("（暂无自动推断的 skill preferences）")
            lines = ["Auto-distilled skill preferences:"]
            for skill in store.list_skills():
                pref = store.get(skill)
                if pref is None:
                    continue
                date = pref.updated_at[:10] if pref.updated_at else "-"
                lines.append(f"  {skill} [updated {date}, confidence={pref.confidence}]:")
                lines.append(f"    {pref.value}")
            return self._sys("\n".join(lines))

        if rest == "clear":
            n = self._preference_store.clear_all()
            return self._sys(f"已清除全部 skill preferences（共 {n} 条）。")

        tokens = rest.split(None, 1)
        if tokens[0] == "delete":
            if len(tokens) < 2:
                return self._sys("用法：/profile skill-prefs delete <skill>")
            skill = tokens[1].strip()
            if self._preference_store.delete(skill):
                return self._sys(f"已删除 skill={skill} 的 distilled preference。")
            return self._sys(f"skill={skill} 没有 distilled preference 可删。")

        return self._sys(
            "用法：/profile skill-prefs [list | delete <skill> | clear]"
        )

    # ============================================================
    # 偏好蒸馏 hook（turn_end / skill_switch / session_end → pipeline）
    # ============================================================

    def _maybe_distill(self, event: str) -> None:
        """触发一次偏好蒸馏。fail-open。

        feedback 取自 ReflexionStore；window / summary / marker 由 pipeline 自行从
        history + meta + store 取。pipeline 返回新 marker 时写回 session.meta。
        """
        if self._distill_pipeline is None or not self._current_skill:
            return
        history = self._store.get(self._session_key)
        if history is None:
            return
        feedback_history: list[str] = []
        if self._reflexions is not None:
            try:
                fb = self._reflexions.read_recent("skill", self._current_skill, n=_FEEDBACK_MAX_PER_SKILL)
                feedback_history = [r.content for r in fb]
            except Exception:
                feedback_history = []
        try:
            new_marker = self._distill_pipeline.maybe_distill(
                skill=self._current_skill,
                event=event,
                history=history.get_history(),
                meta=self._store.get_meta(self._session_key),
                feedback_history=feedback_history,
                now=datetime.now(),
            )
            if new_marker is not None:
                self._store.set_meta(self._session_key, distill=new_marker)
        except Exception as e:
            _logger.warning(f"distill pipeline 异常（已 fail-open）: {e}")

    # ============================================================
    # /feedback：HITL 显式 skill 偏好反馈通道
    # ============================================================
    #
    # 用户在 CLI / TG 里说 `/feedback 以后日报每条 ≤50 字`，写入当前 skill 的
    # ReflexionStore；下次该 skill 渲染时由 SkillLoader 注入到 SKILL.md 前置
    # 块。跟飞轮（trace-level lesson backend）不重叠——飞轮学的是 tool 调用
    # 协议失败修复，feedback 学的是 skill 行为偏好（输出长度 / 类目排序 /
    # 隐性约束等用户主观期望）。
    #
    # 设计简化（MVP）：
    # - 只支持当前 skill scope（要先 /skill <name>）
    # - FIFO 容量上限 _FEEDBACK_MAX_PER_SKILL（默认 20）防 SKILL.md 膨胀
    # - 不做 LLM judge / consolidator——纯文本累积，可解释、可手动清理

    def _handle_feedback(self, text: str) -> HarnessResponse:
        """/feedback <text> | /feedback list | /feedback clear"""
        if self._reflexions is None:
            return self._sys("（feedback 未启用：装配时未注入 reflexions_store）")
        if not self._current_skill:
            return self._sys(
                "用法：先 /skill <name> 切到具体 skill，再 /feedback <text> 写反馈"
            )

        # 拆 token：/feedback / /feedback list / /feedback clear / /feedback <free text>
        parts = text.split(None, 1)
        if len(parts) == 1:
            return self._sys(
                "用法：\n"
                "  /feedback <text>     写一条反馈到当前 skill\n"
                "  /feedback list       看当前 skill 的反馈历史\n"
                "  /feedback clear      清除当前 skill 的反馈"
            )

        body = parts[1].strip()
        if body == "list":
            return self._feedback_list()
        if body == "clear":
            return self._feedback_clear()
        return self._feedback_append(body)

    def _feedback_append(self, content: str) -> HarnessResponse:
        from nanoagent.evolution.reflexion import ReflexionRecord

        skill = self._current_skill
        assert skill is not None  # _handle_feedback 已守

        existing = self._reflexions.read_recent(
            "skill", skill, n=_FEEDBACK_MAX_PER_SKILL
        )
        # FIFO 容量上限：超出先 trim 保留最后 N-1 条，让出槽位给新 record。
        if len(existing) >= _FEEDBACK_MAX_PER_SKILL:
            keep = existing[-(_FEEDBACK_MAX_PER_SKILL - 1):]
            self._reflexions.replace_all("skill", skill, keep)

        # trace_id 用 "feedback:<timestamp>" 占位（用户反馈不绑 trace）
        trace_id = f"feedback:{datetime.now().isoformat(timespec='seconds')}"
        record = ReflexionRecord(
            trace_id=trace_id,
            scope="skill",
            scope_target=skill,
            content=content,
            error_class=None,
            context_tags={"source": "user_feedback"},
        )
        self._reflexions.append(record)
        new_count = min(len(existing) + 1, _FEEDBACK_MAX_PER_SKILL)
        return self._sys(
            f"已记录到 skill={skill} 的反馈历史（共 {new_count} 条）。"
            f"下次 /skill {skill} 时会注入到 SKILL.md 前置块。"
        )

    def _feedback_list(self) -> HarnessResponse:
        skill = self._current_skill
        assert skill is not None
        records = self._reflexions.read_recent(
            "skill", skill, n=_FEEDBACK_MAX_PER_SKILL
        )
        if not records:
            return self._sys(f"（skill={skill} 暂无反馈）")
        lines = [f"skill={skill} 反馈历史（最近 {len(records)} 条）："]
        for i, r in enumerate(records, 1):
            tag = r.context_tags.get("source", "unknown") if r.context_tags else "unknown"
            lines.append(f"  {i}. [{r.created_at[:10]}] [{tag}] {r.content}")
        lines.append("用 /feedback clear 清除全部")
        return self._sys("\n".join(lines))

    def _feedback_clear(self) -> HarnessResponse:
        skill = self._current_skill
        assert skill is not None
        if self._reflexions.clear("skill", skill):
            return self._sys(f"已清除 skill={skill} 的全部反馈。")
        return self._sys(f"（skill={skill} 暂无反馈，无需清除）")

    # ============================================================
    # 对话路径
    # ============================================================

    def _handle_dialogue(self, text: str) -> HarnessResponse:
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

        # RequiredActionGate：从所有声明 contract 的 skill 里匹配
        # 用户本轮 lexical_hints 命中的 action_contracts，注入 loop.run 用于
        # finish 前的 obligation 检查。无任何 skill 声明 contract → 空列表，
        # loop 行为完全不变（向后兼容）。
        pending_contracts = self._collect_matched_action_contracts(text)

        # Session 记账基准：本轮跑前快照累计用量，turn 结束算 delta。
        # run_bot 多 chat 共享同一 llm 实例，但单 worker 串行执行 → 本轮"前→后"
        # 差值仍只反映本轮（不被其他 chat 干扰）。
        usage_before = self._loop.llm.usage.snapshot()

        defer_save = self._evaluator is not None
        answer = self._loop.run(
            messages=turn_messages,
            save_on_finish=not defer_save,
            pending_action_contracts=pending_contracts,
        )

        if defer_save:
            try:
                answer = self._maybe_run_evaluator_retry(answer, turn_messages)
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
        self._maybe_archive_digest(answer)
        self._maybe_distill(event="turn_end")
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
                _logger.info(
                    f"[OutcomeTracker] 更新 {len(updates)} 条 lesson outcome: "
                    + ", ".join(f"{u.lesson_id}({u.outcome.value},conf={u.new_confidence:.2f})"
                                for u in updates)
                )
        except Exception as e:
            _logger.warning(f"OutcomeTracker 处理 trace 异常（已 fail-open）: {e}")

    def _collect_matched_action_contracts(self, user_text: str) -> list:
        """跨所有 skill 收集本轮命中的 action_contracts（lexical_hints 子串匹配）。

        无 skill 声明 contract → 返回空列表（loop.run 行为同旧版）。
        """
        out: list = []
        for name in self._loader.list_skills():
            contract = self._loader.get_contract(name)
            if contract is None:
                continue
            out.extend(contract.matches_action_triggers(user_text))
        return out

    def _is_evaluator_triggered_by_any_skill(self, answer: str) -> bool:
        """遍历所有 skill 的 SkillContract，任一命中 evaluation trigger 即触发。

        无 skill 声明 contract（如全部 skill 都没有 skill.yaml）→ 不触发，安全。
        遍历是 O(skills × triggers)，当前规模忽略不计；扩展成多 skill 时可加索引。
        """
        for name in self._loader.list_skills():
            contract = self._loader.get_contract(name)
            if contract is None:
                continue
            if contract.is_evaluation_triggered(answer):
                return True
        return False

    def _maybe_run_evaluator_retry(self, answer: str, base_history: list[dict]) -> str:
        """Run optional digest evaluator and one bounded retry path.

        Trigger 来源：
        skill manifest（skills/<name>/skill.yaml）的 evaluation.triggers 声明，
        harness 不再持有 skill-specific markers。任一已声明 contract 的 skill
        命中即触发；未声明 contract 的 skill 永不触发（默认安全）。
        """
        if self._evaluator is None:
            return answer
        if not self._is_evaluator_triggered_by_any_skill(answer):
            return answer
        if self._evaluator_max_retries <= 0:
            return answer

        attempts = 0
        current_answer = answer
        current_messages = list(base_history)
        while attempts < self._evaluator_max_retries:
            self._trace_evaluator_event(ts.ACTION_EVALUATOR_CALL_START, attempt=attempts)
            decision = self._evaluator.evaluate(current_answer, current_messages)
            self._trace_evaluator_decision(decision, attempts)
            if not decision.should_retry():
                return current_answer

            hint = self._build_evaluator_hint(decision)
            retry_messages = current_messages + [
                {"role": "assistant", "content": current_answer},
                {"role": "user", "content": hint},
            ]
            self._trace_evaluator_event(
                ts.ACTION_EVALUATOR_RETRY_TRIGGERED,
                attempt=attempts,
                recommended_action=decision.recommended_action,
                missing=decision.missing,
            )
            current_answer = self._loop.run_continuation(
                messages=retry_messages,
                tracer=self._loop._tracer,
                save_on_finish=False,
            )
            current_messages = retry_messages
            attempts += 1

        final_decision_note = (
            f"\n\n> 注：本次日报经 evaluator 判定仍有改进空间"
            f"（missing={decision.missing or '-'}），已达到 evaluator 重试上限。"
        )
        return current_answer + final_decision_note

    @staticmethod
    def _build_evaluator_hint(decision) -> str:
        """Build the user hint for evaluator retry."""
        return (
            f"[evaluator] coverage_ok={decision.coverage_ok}, "
            f"missing={decision.missing}, "
            f"soft_issues={decision.soft_issues}, "
            f"recommended_action={decision.recommended_action}, "
            f"reason={decision.reason or '-'}。\n"
            f"请按 recommended_action 调对应 tool 补齐，再重新生成完整日报。"
        )

    def _trace_evaluator_decision(self, decision, attempt: int) -> None:
        """Write evaluator decision to trace when available."""
        tracer = getattr(self._loop, "_tracer", None)
        if tracer is None:
            return
        try:
            tracer.step(
                action=ts.ACTION_EVALUATOR_CALL_END,
                attempt=attempt,
                coverage_ok=decision.coverage_ok,
                recommended_action=decision.recommended_action,
                missing=decision.missing,
                soft_issues=decision.soft_issues,
                fail_open=decision.fail_open,
                reason=decision.reason[:120] if decision.reason else "",
            )
        except Exception:
            pass

    def _trace_evaluator_event(self, action: str, **fields) -> None:
        """Write a generic evaluator trace event."""
        tracer = getattr(self._loop, "_tracer", None)
        if tracer is None:
            return
        try:
            tracer.step(action=action, **fields)
        except Exception:
            pass

    def _maybe_archive_digest(self, answer: str) -> None:
        """按 skill manifest archive 段决定是否归档。失败不影响本轮。

        之前 _DIGEST_ARCHIVE_MARKERS 6 元组硬编码在本文件顶部，违反 framework
        边界。现在遍历 loader.list_skills() 走 contract dispatch。

        触发逻辑：第一个命中 manifest.is_archive_triggered(answer) 的 skill 决定
        是否落盘 + 落到哪——避免同一 answer 被多 skill 重复归档。
        路径优先级：manifest.output_dir → self._digests_dir → data_dir("digests")
        """
        triggered_output_dir: Optional[str] = None
        triggered_skill: Optional[str] = None
        for name in self._loader.list_skills():
            contract = self._loader.get_contract(name)
            if contract is None:
                continue
            if contract.is_archive_triggered(answer):
                triggered_skill = name
                triggered_output_dir = contract.archive_output_dir()
                break
        if triggered_skill is None:
            return

        now = datetime.now()
        if triggered_output_dir:
            target_dir = data_dir(triggered_output_dir)
        elif self._digests_dir is not None:
            target_dir = self._digests_dir
        else:
            target_dir = data_dir("digests")
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{now.strftime('%Y-%m-%d_%H%M%S')}.md"
        header = (
            f"<!-- session_key: {self._session_key} | "
            f"skill: {self._current_skill or '(auto)'} | "
            f"archive_skill: {triggered_skill} | "
            f"archived_at: {now.isoformat(timespec='seconds')} -->\n\n"
        )
        try:
            path.write_text(header + answer, encoding="utf-8")
        except OSError:
            pass

    def _compose_system_prompt(self) -> str:
        skill_body = self._loader.render(self._current_skill) if self._current_skill else None
        facts_block = self._user_facts.render_block() or None
        return build_system_prompt(self._base_identity, skill_body, facts_block)

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
