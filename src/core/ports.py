"""Core 对观测与会话持久化的边界接口。"""

from typing import Protocol

from .types import Message


class Tracer(Protocol):
    """接收运行时轨迹事件。"""

    def emit(self, action: str, **fields: object) -> None:
        """提交一个不可阻断主流程的事件。"""


class NullTracer:
    """用于独立运行与测试的空轨迹器。"""

    def emit(self, action: str, **fields: object) -> None:
        return None


class OutputStore(Protocol):
    """大体量工具输出的落盘接口。"""

    def put(self, content: str) -> str | None:
        """存下完整内容并返回可供追溯的位置；无处可存时返回 None。"""


class NullOutputStore:
    """不落盘的输出存储实现；截断掉的内容直接丢弃。"""

    def put(self, content: str) -> str | None:
        return None


class SessionStore(Protocol):
    """会话事实的追加式持久化接口。"""

    def begin(self, task: str) -> None: ...
    def append(self, msg: Message) -> None: ...
    def load(self) -> list[Message]: ...
    def close(self) -> None: ...


class NullSessionStore:
    """不落盘的会话存储实现。"""

    def begin(self, task: str) -> None:
        return None

    def append(self, msg: Message) -> None:
        return None

    def load(self) -> list[Message]:
        return []

    def close(self) -> None:
        return None
