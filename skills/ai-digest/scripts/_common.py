"""ai-digest skill 内部公共工具。

不对外暴露（文件名以 `_` 开头提示）：只给同 skill 下的 fetch_*.py 脚本 import。

功能：
  load_sources(): 读 skills/ai-digest/sources.yaml → dict list
  read_args(): 从 stdin 读取 JSON 参数（所有 fetch_*.py 的统一入口）
  ensure_utf8_stdout(): Windows 控制台 CP936 下避免中文输出乱码
"""

import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

# ai-digest skill 根目录 = _common.py 的父目录的父目录（scripts/ 下）
_SKILL_ROOT = Path(__file__).resolve().parent.parent
_SOURCES_PATH = _SKILL_ROOT / "sources.yaml"


def ensure_utf8_stdout() -> None:
    """保证 stdout/stderr 以 UTF-8 输出。Windows CP936 下不 reconfigure 会乱码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def read_args() -> dict:
    """从 stdin 读 JSON 参数。空 stdin / 解析失败 → 返回空 dict。"""
    try:
        raw = sys.stdin.read()
    except Exception:
        return {}
    if not raw or not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def load_sources(category: Optional[str] = None) -> list[dict]:
    """加载 sources.yaml 的 source 列表。

    Args:
        category: 按 category 字段过滤。None = 返回全部。

    Returns:
        source dict list，每条含 {name, rss_url, category, priority}。
        文件缺失 / 格式错 → 返回空 list。
    """
    if not _SOURCES_PATH.exists():
        return []
    try:
        with open(_SOURCES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []
    sources = data.get("sources") or []
    if not isinstance(sources, list):
        return []
    if category:
        sources = [
            s for s in sources
            if isinstance(s, dict)
            and (s.get("category") or "").lower() == category.lower()
        ]
    return [s for s in sources if isinstance(s, dict)]


def print_error(msg: str) -> None:
    """统一错误输出。LLM 看到 `[error]` 前缀知道是失败。"""
    print(f"[error] {msg}", file=sys.stderr)


def load_x_users(category: Optional[str] = None) -> list[dict]:
    """加载 sources.yaml 的 x_users curated 列表。

    结构与 sources 同款：{username, category, priority, name?}。
    category 可选过滤；缺省字段 / 格式错 → 返回空 list（兜底不崩）。
    """
    if not _SOURCES_PATH.exists():
        return []
    try:
        with open(_SOURCES_PATH, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []
    users = data.get("x_users") or []
    if not isinstance(users, list):
        return []
    users = [u for u in users if isinstance(u, dict) and u.get("username")]
    if category:
        users = [u for u in users if (u.get("category") or "").lower() == category.lower()]
    return users
