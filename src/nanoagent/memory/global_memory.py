"""GlobalMemory：自动蒸馏的全局长期记忆（自由文本 markdown 单文件）。

跟 UserFacts 物理分开，是有意的双文件隔离（V3 用户轴 Part 2.2）：
  - UserFacts   = 用户**手动钉**的 KV（user_declared，精确可寻址、可删，绝不被机器改写）
  - GlobalMemory = 副 LLM **从对话蒸**的自由文本块（best-effort，整块重写）

物理分开 → 自动写永远碰不到手写的，连"自动让路给手写"那道判断都省。对标
nanobot Dream 的 MEMORY.md：自由文本、整块重写、无 RAG / 无 eval / 无来源标 / 无遗忘修剪。
"""

import os
import tempfile
from pathlib import Path

from nanoagent.core.logger import get_logger

_logger = get_logger("global_memory")


class GlobalMemory:
    """一个自由文本长期记忆文件的读 / 整块覆盖写 / 渲染。"""

    def __init__(self, path: Path):
        self._path = Path(path)

    def read(self) -> str:
        if not self._path.exists():
            return ""
        try:
            return self._path.read_text(encoding="utf-8").strip()
        except (OSError, UnicodeDecodeError) as e:
            _logger.warning(f"{self._path} 读失败（当空处理）: {e}")
            return ""

    def write(self, text: str) -> None:
        """整块覆盖，原子写（tempfile + os.replace）。空文本也照写（= 清空）。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self._path.stem}_", suffix=".tmp", dir=str(self._path.parent)
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text.rstrip() + "\n")
                os.replace(tmp_name, self._path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as e:
            _logger.warning(f"写 {self._path} 失败（已忽略）: {e}")

    def render_block(self) -> str:
        """渲染成注入 system prompt 的块；无内容返空串（调用方据此判断是否注入）。"""
        body = self.read()
        return f"# 长期记忆（自动）\n\n{body}" if body else ""
