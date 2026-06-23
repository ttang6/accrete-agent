"""prompt_assets — agent 提示词资产化加载。

把所有 agent 的提示词（主 agent 的 base identity + 各副 LLM 的 system prompt）集中放到
仓库根的 `prompts/<name>.md`，**改文件即改提示词、不动代码**。

向后兼容铁律：`load_prompt(name, default)` 读不到文件 / 文件空 / 只剩 frontmatter →
**回退传入的 default**（= 代码里原来的默认提示词）。所以没建文件、或文件清空，行为与
改造前完全一致。

md 可选 YAML frontmatter（`--- ... ---`）放说明备注，会被剥掉、不进提示词；其余正文即
提示词原文。提示词在进程启动时一次性读入（模块级常量），改 md 后需重启进程生效。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# 相对包位置定位仓库根，避免依赖 cwd（eval driver / cron / bot 的 cwd 不固定）。
# src/nanoagent/core/prompt_assets.py → parents[3] = 仓库根
_PROMPTS_DIR = Path(__file__).resolve().parents[3] / "prompts"


def _strip_frontmatter(text: str) -> str:
    """剥掉开头的 `--- ... ---` frontmatter（放备注用）；没有就原样返回。"""
    body = text.lstrip()
    if not body.startswith("---"):
        return text
    end = body.find("\n---", 3)
    if end == -1:
        return text
    nl = body.find("\n", end + 1)
    return body[nl + 1:] if nl != -1 else ""


def load_prompt(name: str, default: str, prompts_dir: Optional[Path] = None) -> str:
    """读 `prompts/<name>.md` 当提示词；文件缺失 / 正文空 → 回退 default。"""
    path = (prompts_dir or _PROMPTS_DIR) / f"{name}.md"
    try:
        raw = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return default
    content = _strip_frontmatter(raw).strip()
    return content if content else default
