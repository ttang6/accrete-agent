# events.py
from enum import Enum

class AgentEvent(str, Enum):
    RUN_START = "run_start"
    RUN_END = "run_end"
    TURN_START = "turn_start"
    TURN_END = "turn_end"
    BEFORE_CONTEXT = "before_context"
    AFTER_CONTEXT = "after_context"
    BEFORE_LLM = "before_llm"
    AFTER_LLM = "after_llm"
    BEFORE_TOOL = "before_tool"
    AFTER_TOOL = "after_tool"
    ON_ERROR = "on_error"
    # 仅在 max_turns 的强制收尾前触发，供外部注入收尾引导而无需改 loop。
    ON_AGENT_END = "on_agent_end"
