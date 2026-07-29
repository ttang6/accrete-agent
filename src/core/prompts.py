"""Core 的内置默认提示词。

这些是上层未提供提示词时的兜底默认值。提示词的正式来源是运行时配置，由上层解析
后注入；本模块不读文件、不查环境变量，也不合并配置优先级。
"""

# 保守估算不得超过约 1500 tokens，ContextManager 在构造时会校验。
DEFAULT_SYSTEM_PROMPT = """You are a coding agent working in a fixed working directory: {workdir}.

Rules:
- Use the provided tools to inspect and modify files. Do not guess file contents.
- Before each tool call, state your reason in one or two sentences.
- bash runs in a fresh subshell each time: `cd` and environment variables do
  not persist. Use `cd sub/dir && cmd` or `VAR=x cmd` instead.
- All paths are relative to the working directory, except the fixed persistent
  documents `/USER.md`, `/MEMORY.md`, `/PLAN.md`, and `/TODO.md`.
- When the task is complete, reply with your final answer and no tool calls.
- Report honestly: if you could not finish, say what remains.
"""

COMPACT_SUMMARIZER_PROMPT = "你负责生成精确的工作摘要。"

DEFAULT_COMPACT_PROMPT = """
按以下结构总结对话：任务目标与当前进展 / 关键决定及理由 / 读过的文件 / 改过的文件 / 未完成事项 / 必须保留的约束。仅输出摘要。
"""


def render(template: str, **variables: object) -> str:
    """用 str.format 填充模板变量。

    模板里出现未提供的变量会直接抛 KeyError，让缺参在组装期暴露，
    而不是把 `{workdir}` 原样发给模型。
    """
    return template.format(**variables)
