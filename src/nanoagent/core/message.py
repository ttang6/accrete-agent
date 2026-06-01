from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class Message:
    """
    统一的消息数据结构。
    role 限定为四种：system / user / assistant / tool
    content 是消息内容文本
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: Optional[list[dict]] = None # assistant 消息中的工具调用
    tool_call_id: Optional[str] = None # tool 消息中，对应哪次调用
    tool_name: Optional[str] = None # tool 消息中，工具名称

    def to_dict(self) -> dict:
        """转为 OpenAI API 需要的 dict 格式"""
        d = {"role": self.role, "content": self.content}

        if self.tool_calls is not None:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_name is not None:
            d["name"] = self.tool_name

        return d
    
    @classmethod
    def system(cls, content: str) -> "Message":
        """快捷创建 system 消息"""
        return cls(role="system", content=content)
    
    @classmethod
    def user(cls, content: str) -> "Message":
        """快捷创建 user 消息"""
        return cls(role="user", content=content)
    
    @classmethod
    def assistant(cls, content: str) -> "Message":
        """快捷创建 assistant 消息"""
        return cls(role="assistant", content=content)
    
    @classmethod
    def tool_result(cls, content: str, tool_call_id: str, tool_name: str) -> "Message":
        """工具执行结果消息"""
        return cls(role="tool", content=content, tool_call_id=tool_call_id, tool_name=tool_name)