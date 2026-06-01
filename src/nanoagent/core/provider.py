"""Provider 配置中心：新增 provider 只改这一个文件。

每个 provider 定义：
  - base_url:        API 地址
  - env_key:         .env 中 API Key 的环境变量名（None 表示不需要 key）
  - temperature:     是否支持 temperature 参数（有些模型不支持会报错）
  - model_overrides: per-model 参数覆盖逃生口（pattern 子串匹配 → 覆盖 dict）

能力粒度是有意分两层、不是 bug：
  - temperature 是 **per-provider** 能力位（整个 provider 一刀切）。
  - max_completion_tokens vs max_tokens 是 **per-model-family** 区分
    （见 llm_client._uses_max_completion_tokens，按 gpt-5/o1 等前缀判断）。
  - 个别 model 的零散怪癖（如某模型强制 temperature=1.0）走 model_overrides
    统一收口，不必为单个 model 新开 provider。
"""

import os
from dataclasses import dataclass
from typing import Dict, Optional

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


@dataclass
class ProviderConfig:
    base_url: str
    env_key: Optional[str] = None
    temperature: bool = True
    # per-model 参数覆盖：(pattern, overrides) 序列。pattern 子串命中当前 model
    # 名（小写）即把 overrides 合进请求 kwargs，首个命中生效。学 nanobot 的逃生口。
    model_overrides: tuple[tuple[str, dict], ...] = ()

    def get_api_key(self) -> Optional[str]:
        """从环境变量读取 API Key。"""
        if self.env_key is None:
            return None
        return os.getenv(self.env_key)

    def overrides_for(self, model: str) -> dict:
        """返回命中当前 model 的参数覆盖（无命中则空 dict）。"""
        model_lower = (model or "").lower()
        for pattern, overrides in self.model_overrides:
            if pattern in model_lower:
                return dict(overrides)
        return {}


# ---------- 所有 provider 在这里注册 ----------

PROVIDERS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        base_url="https://api.openai.com/v1",
        env_key="OPENAI_API_KEY",
        temperature=True,
    ),
    "dashscope": ProviderConfig(
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        env_key="DASHSCOPE_API_KEY",
        temperature=True,
    ),
    "zhipu": ProviderConfig(
        base_url="https://open.bigmodel.cn/api/paas/v4",
        env_key="ZHIPU_API_KEY",
        temperature=True,
    ),
    "deepseek": ProviderConfig(
        base_url="https://api.deepseek.com/v1",
        env_key="DEEPSEEK_API_KEY",
        temperature=True,
    ),
    "ollama": ProviderConfig(
        base_url="http://localhost:11434/v1",
        env_key=None,
        temperature=False,
    ),
    "vllm": ProviderConfig(
        base_url="http://localhost:8000/v1",
        env_key=None,
        temperature=True,
    ),
}


# ---------- 查询接口 ----------

def get_base_url(provider: str) -> Optional[str]:
    cfg = PROVIDERS.get(provider)
    return cfg.base_url if cfg else None


def get_api_key(provider: str) -> Optional[str]:
    cfg = PROVIDERS.get(provider)
    return cfg.get_api_key() if cfg else None


def supports_temperature(provider: str) -> bool:
    cfg = PROVIDERS.get(provider)
    return cfg.temperature if cfg else True


def list_providers() -> list[str]:
    return list(PROVIDERS.keys())


def model_overrides_for(provider: str, model: str) -> dict:
    """返回命中当前 provider/model 的参数覆盖（无 provider 或无命中则空 dict）。"""
    cfg = PROVIDERS.get(provider)
    return cfg.overrides_for(model) if cfg else {}
