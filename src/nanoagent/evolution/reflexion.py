"""evolution/reflexion：Reflexion 记录 + skill 级存储（骨架位）。

当前职责：
  - 定义 `ReflexionRecord` 数据结构，含 `scope` 字段（skill / tool / framework enum）
  - 实现 `ReflexionStore` 读写 **只 skill 级**（scope="skill"）的 jsonl
  - `SkillLoader.render()` 调 `render_for_skill(name)` 注入 `# 历史教训` 前置块

后续激活时要加的（当前骨架已预留）：
  - `append` 支持 scope="tool" / "framework" 路径
  - ReflexionGate 噪音过滤（is_transient / duplicate / stale）
  - Consolidator 聚类 + promotion
  - `read_recent` 支持 tool / framework 级

存储：
    data/reflexions/
    ├ skills/
    │  ├ <skill_name>.jsonl       ← 当前唯一写入路径
    │  └ ...
    ├ tools/                       ← 后续扩展
    │  └ <tool_name>.jsonl
    └ framework.jsonl              ← 后续扩展

每行 = 一条 ReflexionRecord as JSON。append-only，无 rewrite。加字段对旧数据透明。
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

from nanoagent.core.logger import get_logger

_logger = get_logger("reflexion")

Scope = Literal["skill", "tool", "framework"]


@dataclass
class ReflexionRecord:
    """单条 reflexion 记录。"""
    trace_id: str
    scope: Scope                    # "skill" / "tool" / "framework"
    scope_target: str               # skill_name / tool_name / "framework"
    content: str                    # 教训文本，供 LLM 读取
    error_class: Optional[str] = None
    context_tags: dict = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat(timespec="seconds")
    )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ReflexionRecord":
        return cls(
            trace_id=data["trace_id"],
            scope=data["scope"],
            scope_target=data["scope_target"],
            content=data["content"],
            error_class=data.get("error_class"),
            context_tags=dict(data.get("context_tags", {})),
            created_at=data.get(
                "created_at", datetime.now().isoformat(timespec="seconds")
            ),
        )


class ReflexionStore:
    """按 scope 索引 reflexion jsonl 文件。当前只实现 scope='skill'。"""

    def __init__(self, root: Path):
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / "skills").mkdir(parents=True, exist_ok=True)

    def _path_for(self, scope: Scope, scope_target: str) -> Path:
        if scope == "skill":
            return self._root / "skills" / f"{scope_target}.jsonl"
        # 后续扩展位（当前调用会被 append/read_recent 的 scope 过滤挡住）
        if scope == "tool":
            return self._root / "tools" / f"{scope_target}.jsonl"
        if scope == "framework":
            return self._root / "framework.jsonl"
        raise ValueError(f"未知 scope: {scope!r}")

    def append(self, record: ReflexionRecord) -> None:
        """追加一条。**只处理 scope='skill'**；其他 scope 忽略不报错。"""
        if record.scope != "skill":
            _logger.debug(
                f"当前只支持 scope='skill'；收到 {record.scope!r}，跳过"
            )
            return
        path = self._path_for(record.scope, record.scope_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            _logger.warning(f"写 {path} 失败: {e}")

    def read_recent(
        self, scope: Scope, scope_target: str, n: int = 5
    ) -> list[ReflexionRecord]:
        """读最近 n 条。文件不存在 / scope 未支持 → 返回空列表。"""
        if scope != "skill":
            return []
        path = self._path_for(scope, scope_target)
        if not path.exists():
            return []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as e:
            _logger.warning(f"读 {path} 失败: {e}")
            return []

        records: list[ReflexionRecord] = []
        for line in lines[-n:]:
            if not line.strip():
                continue
            try:
                records.append(ReflexionRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                _logger.warning(f"跳过损坏记录 in {path.name}: {e}")
        return records

    def replace_all(
        self, scope: Scope, scope_target: str, records: list["ReflexionRecord"]
    ) -> None:
        """覆盖写：用 records 重写 jsonl（用于 FIFO trim / 手动整理）。

        当前只支持 scope='skill'；其他 scope 静默跳过保持跟 append 一致。
        records 空列表 = 文件清空（仍保留 0 字节文件，不删）。
        """
        if scope != "skill":
            _logger.debug(
                f"当前只支持 scope='skill'；replace_all 收到 {scope!r}，跳过"
            )
            return
        path = self._path_for(scope, scope_target)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
        except OSError as e:
            _logger.warning(f"replace_all 写 {path} 失败: {e}")

    def clear(self, scope: Scope, scope_target: str) -> bool:
        """删除 scope_target 对应的 jsonl 文件。

        返回 True = 删除成功；False = 文件不存在 / 删除失败。
        当前只支持 scope='skill'。
        """
        if scope != "skill":
            return False
        path = self._path_for(scope, scope_target)
        if not path.exists():
            return False
        try:
            path.unlink()
            return True
        except OSError as e:
            _logger.warning(f"clear {path} 失败: {e}")
            return False

    def render_for_skill(self, skill_name: str, n: int = 5) -> str:
        """把最近 n 条渲染成 markdown 块（无标题，供 SkillLoader 前置拼装）。

        无记录时返回空串；调用方据此判断是否注入。
        """
        records = self.read_recent("skill", skill_name, n=n)
        if not records:
            return ""
        lines = []
        for r in records:
            prefix = f"[{r.created_at[:10]}]"
            if r.error_class:
                prefix += f" [{r.error_class}]"
            lines.append(f"- {prefix} {r.content}")
        return "\n".join(lines)
