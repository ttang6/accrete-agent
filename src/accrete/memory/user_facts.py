"""UserFacts：显式维护的跨 skill user profile（扁平 KV 单层）。

设计原则：
  - 跨 skill 共享的身份 / 约束 / 偏好的唯一合法存储位
  - **lazy + 事件驱动**写入——无主动 onboarding
  - **绝对不做**从对话自动推断（那是 episodic 的事，v2 不走此路）
  - `source` 字段必填（evolution drift detection 的必要信息）

结构：
    data/memory/user_facts.json
    {
      "timezone":  {"value": "Asia/Shanghai", "source": "user_declared", "updated_at": "..."},
      "role":      {"value": "Python 工程师",  "source": "asked_by_agent", "updated_at": "..."}
    }

source 枚举：
    user_declared        用户直接声明
    asked_by_agent       agent 主动询问后用户回答
    promoted_from_skill  skill 层信号跨 scope 提升（consolidator 用）
    system_derived       系统从元数据推导

后续扩展路径：升级为 `{"core": {...}, "contexts": {...}}` 两层 +
`context` 默认参数。纯加字段升级，非破坏性。
"""

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Optional

from accrete.core.logger import get_logger

_logger = get_logger("user_facts")


class UserFacts:
    SOURCES: Final[set[str]] = {
        "user_declared",
        "asked_by_agent",
        "promoted_from_skill",
        "system_derived",
    }

    def __init__(self, path: Path):
        self._path = Path(path)
        self._facts: dict[str, dict] = {}
        self._load()

    # ============================================================
    # 读 / 写 / 删
    # ============================================================

    def get(self, key: str) -> Optional[Any]:
        entry = self._facts.get(key)
        return entry["value"] if entry else None

    def get_entry(self, key: str) -> Optional[dict]:
        """返回完整记录副本（含 source / updated_at）。"""
        entry = self._facts.get(key)
        return dict(entry) if entry else None

    def set(self, key: str, value: Any, source: str) -> None:
        if source not in self.SOURCES:
            raise ValueError(
                f"source 必须是 {sorted(self.SOURCES)} 之一，收到 {source!r}"
            )
        self._facts[key] = {
            "value": value,
            "source": source,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._save()

    def delete(self, key: str) -> bool:
        if key in self._facts:
            del self._facts[key]
            self._save()
            return True
        return False

    def get_all(self) -> dict[str, dict]:
        """返回副本（调用方改动不影响内部状态）。"""
        return {k: dict(v) for k, v in self._facts.items()}

    def __len__(self) -> int:
        return len(self._facts)

    def __contains__(self, key: str) -> bool:
        return key in self._facts

    # ============================================================
    # 渲染成 prompt block
    # ============================================================

    def render_block(self) -> str:
        """返回 `# 用户画像\n\n- key: value\n...` 格式；无 facts 时返回空串。

        调用方据返回空串判断是否注入 system_prompt。
        """
        if not self._facts:
            return ""
        lines = ["# 用户画像", ""]
        for key, entry in self._facts.items():
            lines.append(f"- {key}: {entry['value']}")
        return "\n".join(lines)

    # ============================================================
    # 持久化
    # ============================================================

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            _logger.warning(
                f"{self._path} 加载失败: {type(e).__name__}: {e}；从空开始"
            )
            return
        if not isinstance(data, dict):
            _logger.warning(f"{self._path} 格式不符（非 dict），忽略")
            return
        # 逐条校验 entry 结构
        valid: dict[str, dict] = {}
        for key, entry in data.items():
            if (
                isinstance(entry, dict)
                and "value" in entry
                and "source" in entry
            ):
                valid[key] = entry
            else:
                _logger.warning(f"{self._path} 跳过损坏条目 {key!r}")
        self._facts = valid

    def _save(self) -> None:
        """原子写：tempfile + os.replace。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd, tmp_name = tempfile.mkstemp(
                prefix=f".{self._path.stem}_",
                suffix=".tmp",
                dir=str(self._path.parent),
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._facts, f, ensure_ascii=False, indent=2)
                os.replace(tmp_name, self._path)
            except Exception:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError as e:
            _logger.warning(f"写 {self._path} 失败（已忽略）: {e}")
