"""HistoryManager：内存中的消息列表管理器（活跃层）。

定位：
- 每 turn 的消息追加单位。SessionStore 之上的薄层抽象
- 不关心文件系统、不关心 key 路由（SessionStore 管那些）
- 可独立单元测试（不依赖 IO）

公共 API 保持非常干净：只有追加 / 读取 / 硬截断 / 序列化。
**不暴露** compress / compact / summarize。

扩展方向（暂**不写**，真需要 LLM summary 压缩时再加）：
    (a) `SummarizingHistoryManager(HistoryManager)` 子类，
        覆盖 `append_*` 在超阈值时触发 summary
    (b) 给 `__init__` 加 `compactor: Optional[HistoryCompactor] = None` 参数，
        委托给外部 compactor 对象处理压缩

两条路径都比公开一个 `compress()` 方法更稳——API 只承诺硬截断，
压缩逻辑作为特定场景的扩展注入。

存 tool role 消息？—— **不存**。沿用 archive 决策：
只保留 user + assistant，tool_call 消息在单次 MainLoop.run() 内有效，
run 结束不入 history。
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class HistoryManager:
    """消息列表容器。OpenAI 格式 dict（{"role":"user"|"assistant", "content": str}）。"""

    max_messages: int = 100
    messages: list[dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    # ---------- 追加 ----------

    def append_user(self, content: str) -> None:
        self.messages.append({"role": "user", "content": content})
        self.updated_at = datetime.now()
        self.truncate_if_needed()

    def append_assistant(self, content: str) -> None:
        self.messages.append({"role": "assistant", "content": content})
        self.updated_at = datetime.now()
        self.truncate_if_needed()

    # ---------- 读取 ----------

    def get_history(self, max_messages: Optional[int] = None) -> list[dict]:
        """返回消息列表副本。调用方修改副本不影响内部状态。"""
        if max_messages is None or len(self.messages) <= max_messages:
            return list(self.messages)
        return list(self.messages[-max_messages:])

    def turn_count(self) -> int:
        """粗估轮次数。一轮 = user + assistant。"""
        return len(self.messages) // 2

    # ---------- 容量 ----------

    def truncate_if_needed(self) -> None:
        """超 max_messages 时删最旧。硬截断，不做智能压缩。"""
        excess = len(self.messages) - self.max_messages
        if excess > 0:
            del self.messages[:excess]

    def clear(self) -> None:
        self.messages.clear()
        self.updated_at = datetime.now()

    def rewind_turns(self, n: int) -> int:
        """回退最近 n 轮：从倒数第 n 条 user 消息起（含）截断到末尾。

        一轮 = 一条 user 打头的对话段。按 user 消息定界（而非死板 user+assistant 配对），
        对被打断/半截的历史也稳。截断是不可逆的活跃层动作；调用方若要留档，应在调用前
        自行归档。

        Args:
            n: 回退几轮。

        Returns:
            真正回退的轮数（不足 n 轮时按现有轮数封顶；无 user 消息则 0）。
        """
        if n <= 0:
            return 0
        user_idxs = [i for i, m in enumerate(self.messages) if m.get("role") == "user"]
        if not user_idxs:
            return 0
        n = min(n, len(user_idxs))
        cut = user_idxs[-n]
        self.messages = self.messages[:cut]
        self.updated_at = datetime.now()
        return n

    # ---------- 序列化（给 SessionStore 用）----------

    def to_dict(self) -> dict:
        return {
            "max_messages": self.max_messages,
            "messages": list(self.messages),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HistoryManager":
        return cls(
            max_messages=int(data.get("max_messages", 100)),
            messages=list(data.get("messages", [])),
            created_at=datetime.fromisoformat(data["created_at"])
            if data.get("created_at") else datetime.now(),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if data.get("updated_at") else datetime.now(),
        )
