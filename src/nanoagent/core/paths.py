"""统一的数据路径管理。

所有运行时产物（记忆、RAG 向量、笔记、日志等）统一存放在项目根目录的
data/ 下，按组件分子目录。根目录可通过环境变量 NANOAGENT_DATA 覆盖。

目录结构（按组件分子目录，列实际用到的）：
    data/
    ├── memory/        ← UserFacts 用户画像 + 历史去重（digest_reported.jsonl）
    ├── logs/          ← 运行日志 + traces/
    ├── digests/       ← 归档的日报
    ├── tool_outputs/  ← 超阈值 tool 输出落盘
    ├── runtime/       ← sessions / lessons 等运行时状态
    └── cache/         ← API 缓存

用法：
    from nanoagent.core.paths import data_dir

    data_dir("memory")   → data/memory/
    data_dir("logs")     → data/logs/
    data_dir("digests")  → data/digests/

    # 部署时切换根目录
    export NANOAGENT_DATA=/var/lib/nanoagent
"""

import os
from pathlib import Path

_DEFAULT_ROOT = Path("data")


def project_root() -> Path:
    """返回项目根目录（.env 所在目录）。用于定位 config/ 等非数据目录。"""
    from dotenv import find_dotenv
    env_path = find_dotenv()
    return Path(env_path).parent if env_path else Path.cwd()


def get_data_root() -> Path:
    """获取数据根目录，优先读环境变量 NANOAGENT_DATA。"""
    env = os.getenv("NANOAGENT_DATA")
    return Path(env) if env else _DEFAULT_ROOT


def data_dir(component: str, *, auto_create: bool = True) -> Path:
    """获取某个组件的数据子目录。

    Args:
        component: 组件名（如 "memory"、"rag"、"notes"）。
        auto_create: 是否自动创建目录，默认 True。
    """
    path = get_data_root() / component
    if auto_create:
        path.mkdir(parents=True, exist_ok=True)
    return path
