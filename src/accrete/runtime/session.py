"""SessionStore：按 session_key 索引多个 HistoryManager（持久化层）。

定位：
- 每次 append_* 自动原子落盘（long-running bot 友好）
- 启动时自动 hydrate 所有主 session 文件
- Snapshot API 专为 debug / replay / eval / evolution 归档预留，不是主路径

主路径（普通 agent loop 用）：
    store = SessionStore(persist_dir=Path("data/runtime/sessions"))
    store.append_user("cli:default", "hi")
    history = store.get("cli:default")   # HistoryManager
    messages = history.get_history()     # 喂给 MainLoop.run

Snapshot 路径（debug / replay / eval harness）：
    store.save_snapshot("cli:default", "pre-evolution-v1")
    ...
    history = store.fork_snapshot("pre-evolution-v1")  # 游离 HistoryManager
    # replay 用，不回写

文件布局：
    data/runtime/sessions/
    ├ cli_default.json                  # 主 session（key="cli:default"）
    ├ telegram_6302437207.json          # 另一 channel session
    └ snapshots/
        ├ pre-evolution-v1.json         # 独立 snapshot，不自动 hydrate
        ├ replay-buggy-abc.json
        └ eval-baseline-v1.json

关键边界：
- `_load_all()` **只扫主文件，不扫 snapshots 子目录**——snapshot 显式 fork 才复活
- `fork_snapshot()` 返回游离 HistoryManager：不进 `self._sessions`，不持久化
- 主 session 的 append/autosave 循环不受 snapshot 数量影响
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional

from accrete.core.logger import get_logger
from accrete.runtime.history import HistoryManager

_logger = get_logger("session")

_SNAPSHOT_SUBDIR = "snapshots"


def _safe_filename(key: str) -> str:
    """session_key 形如 'cli:default' / 'telegram:6302437207'。
    Windows 文件名不允许 ':'，映射到 '_'。"""
    return key.replace(":", "_").replace("/", "_").replace("\\", "_")


class SessionStore:
    """按 key 索引 HistoryManager 实例。persist_dir 非 None 时每次 append 原子落盘。"""

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        max_messages_per_session: int = 100,
    ):
        self._sessions: dict[str, HistoryManager] = {}
        self._meta: dict[str, dict] = {}
        self._max_messages = max_messages_per_session
        self._persist_dir: Optional[Path] = Path(persist_dir) if persist_dir else None
        if self._persist_dir is not None:
            self._persist_dir.mkdir(parents=True, exist_ok=True)
            (self._persist_dir / _SNAPSHOT_SUBDIR).mkdir(parents=True, exist_ok=True)
            self._load_all()

    # ============================================================
    # 主路径：自动持久化
    # ============================================================

    def get_or_create(self, key: str) -> HistoryManager:
        sess = self._sessions.get(key)
        if sess is None:
            sess = HistoryManager(max_messages=self._max_messages)
            self._sessions[key] = sess
        return sess

    def get(self, key: str) -> Optional[HistoryManager]:
        return self._sessions.get(key)

    def get_meta(self, key: str) -> dict:
        """返回 session 级元数据副本。当前用于恢复 harness 状态（如 current_skill）。"""
        return dict(self._meta.get(key, {}))

    def set_meta(self, key: str, **meta_fields) -> None:
        """更新 session 级元数据并持久化。

        `None` 值表示删除该字段。若 session 尚不存在，会创建一个空 HistoryManager，
        以便 meta 可以独立于历史持久化。
        """
        self.get_or_create(key)  # 明确创建意图：只为让 meta 有主文件可落盘
        meta = dict(self._meta.get(key, {}))
        for field, value in meta_fields.items():
            if value is None:
                meta.pop(field, None)
            else:
                meta[field] = value
        self._meta[key] = meta
        self._save(key)

    def append_turn(self, key: str, user_content: str, assistant_content: str) -> HistoryManager:
        """原子追加一整轮 user+assistant，对异常路径更友好。"""
        sess = self.get_or_create(key)
        sess.append_user(user_content)
        sess.append_assistant(assistant_content)
        self._save(key)
        return sess

    def rewind_turns(self, key: str, n: int) -> int:
        """回退某 session 最近 n 轮（活跃层截断）并落盘。返回真正回退的轮数。

        rewind 归档（把废弃分支存成 fork 副本）不在这里做——由调用方（Harness）在调用
        本方法前先 save_snapshot，保证归档的是截断前的完整历史。

        Args:
            key: session_key。
            n: 回退几轮。

        Returns:
            真正回退的轮数；session 不存在或无可退时 0。
        """
        sess = self._sessions.get(key)
        if sess is None:
            return 0
        removed = sess.rewind_turns(n)
        if removed:
            self._save(key)
        return removed

    def list_sessions(self) -> list[dict]:
        """列所有已 hydrate 的 session（不含 snapshots）。

        返回字段：key / turn_count / title / last_used_at / current_skill。
        按 last_used_at 倒序——最近用过的排前。无 last_used_at 的排最后。
        """
        rows = []
        for key, sess in self._sessions.items():
            meta = self._meta.get(key, {})
            rows.append({
                "key": key,
                "turn_count": sess.turn_count(),
                "title": meta.get("title") or "",
                "last_used_at": meta.get("last_used_at") or "",
                "current_skill": meta.get("current_skill"),
            })
        rows.sort(key=lambda r: r["last_used_at"] or "", reverse=True)
        return rows

    def has(self, key: str) -> bool:
        """判断某 key 的主 session 是否已 hydrate（用于 /resume 校验）。"""
        return key in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    # ============================================================
    # Snapshot 路径：debug / replay / eval / evolution 归档
    # ============================================================

    def save_snapshot(self, key: str, name: str) -> Path:
        """把当前主 session 冷存一份到 snapshots/{name}.json，独立于主文件。

        后续主 session 的 append 不会影响这份 snapshot。
        """
        if self._persist_dir is None:
            raise RuntimeError("SessionStore 未配置 persist_dir，无法 save_snapshot")
        sess = self._sessions.get(key)
        if sess is None:
            raise KeyError(f"session '{key}' 不存在，无法 save_snapshot")

        path = self._persist_dir / _SNAPSHOT_SUBDIR / f"{_safe_filename(name)}.json"
        payload = {
            "name": name,
            "source_key": key,
            "saved_at": datetime.now().isoformat(),
            "history": sess.to_dict(),
        }
        self._atomic_write(path, payload)
        _logger.info(f"[session] snapshot saved: {path.name} (from key={key})")
        return path

    def fork_snapshot(self, name: str) -> HistoryManager:
        """从 snapshot 文件复活出一个游离的 HistoryManager。

        - 不放回 self._sessions
        - 不自动持久化
        - 调用方用完可扔。典型用途：replay 一段老对话跑 eval
        """
        if self._persist_dir is None:
            raise RuntimeError("SessionStore 未配置 persist_dir，无法 fork_snapshot")
        path = self._persist_dir / _SNAPSHOT_SUBDIR / f"{_safe_filename(name)}.json"
        if not path.exists():
            raise FileNotFoundError(f"snapshot 不存在: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return HistoryManager.from_dict(data["history"])

    def list_snapshots(self) -> list[dict]:
        """列出所有 snapshot 元数据（name / source_key / saved_at）。"""
        if self._persist_dir is None:
            return []
        snap_dir = self._persist_dir / _SNAPSHOT_SUBDIR
        if not snap_dir.exists():
            return []
        out = []
        for p in snap_dir.glob("*.json"):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                out.append({
                    "file": p.name,
                    "name": data.get("name"),
                    "source_key": data.get("source_key"),
                    "saved_at": data.get("saved_at"),
                })
            except (json.JSONDecodeError, OSError, KeyError) as e:
                _logger.warning(f"[session] 跳过损坏 snapshot {p.name}: {e}")
        out.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return out

    # ============================================================
    # 内部：原子写 + hydrate
    # ============================================================

    def _main_path(self, key: str) -> Path:
        assert self._persist_dir is not None
        return self._persist_dir / f"{_safe_filename(key)}.json"

    def _save(self, key: str) -> None:
        if self._persist_dir is None:
            return
        sess = self._sessions.get(key)
        if sess is None:
            return
        payload = {
            "key": key,
            "history": sess.to_dict(),
            "meta": dict(self._meta.get(key, {})),
        }
        self._atomic_write(self._main_path(key), payload)

    def _load_all(self) -> None:
        """启动时 hydrate 主 session 文件。**不扫 snapshots 子目录**。"""
        if self._persist_dir is None or not self._persist_dir.exists():
            return
        loaded = 0
        for p in self._persist_dir.glob("*.json"):
            # 只扫主目录顶层 *.json，snapshots/ 子目录由 glob("*.json") 自然排除
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                key = data.get("key")
                history_data = data.get("history")
                if not key or not isinstance(history_data, dict):
                    _logger.warning(f"[session] 跳过格式不符文件 {p.name}")
                    continue
                self._sessions[key] = HistoryManager.from_dict(history_data)
                meta = data.get("meta")
                self._meta[key] = dict(meta) if isinstance(meta, dict) else {}
                loaded += 1
            except (json.JSONDecodeError, OSError, KeyError) as e:
                _logger.warning(f"[session] 跳过损坏文件 {p.name}: {type(e).__name__}: {e}")
        if loaded:
            _logger.info(f"[session] 已从 {self._persist_dir} 加载 {loaded} 个 session")

    @staticmethod
    def _atomic_write(path: Path, payload: dict) -> None:
        """原子写：tempfile + os.replace，避免进程崩溃留半文件。"""
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{path.stem}_",
                suffix=".tmp",
                dir=str(path.parent),
            )
            try:
                # errors="replace" 容忍输入里漏进来的 lone surrogate（如跨平台
                # pipe / copy-paste 带坏字节），用 � 替代，不让存盘崩溃
                with os.fdopen(fd, "w", encoding="utf-8", errors="replace") as f:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                os.replace(tmp_name, path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as e:
            _logger.warning(f"[session] 写 {path.name} 失败（已忽略）: {e}")
