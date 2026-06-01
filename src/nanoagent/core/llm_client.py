import os
import time
import weakref
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv
from openai import APIConnectionError, APITimeoutError, OpenAI, RateLimitError


load_dotenv(find_dotenv())


@dataclass
class TokenUsage:
    """Token 用量统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    call_count: int = 0

    def add(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.call_count += 1

    def snapshot(self) -> "TokenUsage":
        """返回当前累计值的浅拷贝（不可变快照）。

        MainLoop.run() 开头记快照作 baseline，finalize 时算
        delta = current - baseline，得到单 turn 真实 token 成本，
        而不是同 LLM 实例的累计值（否则 trace 里 tokens.total 会一路累加）。
        """
        return TokenUsage(
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            call_count=self.call_count,
        )

    def estimate_cost(self, input_per_million: float, output_per_million: float) -> float:
        return (self.prompt_tokens * input_per_million + self.completion_tokens * output_per_million) / 1_000_000

    def __str__(self):
        return (f"Token 用量: {self.total_tokens} "
                f"(输入: {self.prompt_tokens}, 输出: {self.completion_tokens}, "
                f"调用: {self.call_count} 次)")


class LLMClient:
    """统一的 LLM 客户端，兼容 OpenAI 接口。"""

    _instances: "weakref.WeakSet[LLMClient]" = weakref.WeakSet()

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        provider: Optional[str] = None,
        instance_name: Optional[str] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        base_delay: float = 1.0,
        max_tokens: Optional[int] = None,
        max_completion_tokens: Optional[int] = None,
        extra_body: Optional[dict] = None,
    ):
        from nanoagent.core.provider import get_base_url, get_api_key, supports_temperature, list_providers
        from nanoagent.core.logger import get_logger
        self._logger = get_logger("llm")

        self.model = model or os.getenv("OPENAI_MODEL_ID")
        timeout = timeout or int(os.getenv("OPENAI_TIMEOUT", 60))
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.usage = TokenUsage()
        self._stats = {"calls": 0, "retries": 0, "failures": 0}
        self._provider = provider
        self._instance_name = (instance_name or "llm").strip() or "llm"
        # provider-specific 透传字段（如 qwen 的 enable_thinking）。dict copy 防 caller mutate。
        self._extra_body = dict(extra_body) if extra_body else None
        if max_completion_tokens is not None:
            self._ping_token_kwarg = {"max_completion_tokens": max_completion_tokens}
        elif max_tokens is not None:
            self._ping_token_kwarg = {"max_tokens": max_tokens}
        else:
            self._ping_token_kwarg = {"max_tokens": 1}

        # 从 provider 解析连接信息
        if provider:
            base_url = base_url or get_base_url(provider)
            api_key = api_key or get_api_key(provider)
            if not base_url:
                raise ValueError(f"未知 provider: {provider}，可选: {list_providers()}")
            if not api_key and provider != "ollama":
                raise ValueError(f"provider '{provider}' 的 API Key 未配置，请检查 .env")
            self._supports_temperature = supports_temperature(provider)
        else:
            self._supports_temperature = True

        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL")

        if provider == "ollama" and not api_key:
            api_key = "ollama"

        if not all([self.model, api_key, base_url]):
            raise ValueError("请提供 model、API 密钥和服务地址。")

        self.base_url = base_url
        self.client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
        self.__class__._instances.add(self)

    @classmethod
    def describe_instances(cls) -> list[dict[str, str]]:
        rows = []
        for instance in list(cls._instances):
            rows.append({
                "instance_name": instance._instance_name,
                "model": instance.model or "",
            })
        rows.sort(key=lambda item: (item["instance_name"], item["model"]))
        return rows

    def _should_send_temperature(self) -> bool:
        return self._supports_temperature

    def _uses_max_completion_tokens(self) -> bool:
        # max_completion_tokens vs max_tokens 是 per-model-family 区分（按前缀判断），
        # 跟 supports_temperature 的 per-provider 粒度有意不同，不是 bug：见 provider.py 顶部。
        # gpt-5 系列和 o1/o3 系列用 max_completion_tokens，旧模型用 max_tokens。
        model = (self.model or "").lower()
        return model.startswith(("gpt-5", "o1", "o3", "o4"))

    def _build_request_kwargs(self, messages: List[Dict[str, str]], temperature: float = 0, **kwargs) -> dict:
        # max_tokens → max_completion_tokens（gpt-5 / o1 / o3 系列要求）
        if "max_tokens" in kwargs and self._uses_max_completion_tokens():
            kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
        request_kwargs = {
            "model": self.model,
            "messages": messages,
            **kwargs,
        }
        if self._should_send_temperature():
            request_kwargs["temperature"] = temperature
        if self._extra_body:
            request_kwargs["extra_body"] = self._extra_body
        # per-model 参数覆盖（如某模型强制 temperature=1.0）最后生效，压过上面默认。
        if self._provider:
            from nanoagent.core.provider import model_overrides_for
            request_kwargs.update(model_overrides_for(self._provider, self.model or ""))
        return request_kwargs

    def _create_completion_with_retry(self, **kwargs) -> Any:
        retryable_errors = (APITimeoutError, APIConnectionError, RateLimitError)
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                self._stats["calls"] += 1
                response = self.client.chat.completions.create(**kwargs)
                self._track_usage(response)
                return response
            except retryable_errors as error:
                last_error = error
                if attempt >= self.max_retries:
                    self._stats["failures"] += 1
                    break

                delay = self.base_delay * (2 ** attempt)
                self._stats["retries"] += 1
                print(
                    f"[LLM 重试 {attempt + 1}/{self.max_retries}] "
                    f"{type(error).__name__}: {error}"
                )
                print(f"  等待 {delay} 秒后重试...")
                time.sleep(delay)

        raise last_error

    def ping(self, timeout: Optional[int] = 10) -> tuple[bool, str]:
        """启动期自检：发一个最小请求验证模型可达、API Key 有效、模型名正确。

        Args:
            timeout: 单次 ping 的超时秒数（覆盖默认 60s 避免卡住）

        Returns:
            (ok, 错误信息)。ok=True 时错误信息为空字符串。
        """
        old_timeout = self.client.timeout
        try:
            # 临时缩短超时，避免自检本身把启动卡住
            if timeout is not None:
                self.client.timeout = timeout
            request_kwargs = {
                "model": self.model,
                "messages": [{"role": "user", "content": "ping"}],
                **self._ping_token_kwarg,
            }
            if self._should_send_temperature():
                request_kwargs["temperature"] = 0
            self.client.chat.completions.create(**request_kwargs)
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {e}"
        finally:
            self.client.timeout = old_timeout

    def think(self, messages: List[Dict[str, str]], temperature: float = 0) -> str:
        """调用 LLM，并返回完整响应文本。"""
        response = self._create_completion_with_retry(**self._build_request_kwargs(
            messages=messages,
            temperature=temperature,
            stream=False,
        ))
        return response.choices[0].message.content or ""

    def think_with_tools(
        self,
        messages: List[Dict[str, str]],
        tools: List[Dict[str, str]],
        temperature: float = 0,
    ) -> Any:
        """OpenAI Function Calling 模式调用 LLM，并返回 message 对象。"""
        response = self._create_completion_with_retry(**self._build_request_kwargs(
            messages=messages,
            temperature=temperature,
            tools=tools,
            tool_choice="auto",
            stream=False,
        ))
        return response.choices[0].message

    def _track_usage(self, response: Any):
        """从响应中提取并记录 token 用量（含 per-call 日志）。"""
        if hasattr(response, 'usage') and response.usage:
            pin = response.usage.prompt_tokens
            pout = response.usage.completion_tokens
            self.usage.add(prompt=pin, completion=pout)
            self._logger.debug(
                f"LLM call: +{pin} in / +{pout} out "
                f"(total: {self.usage.total_tokens})"
            )
    
    def usage_summary(self) -> str:
        """格式化 token 用量汇总。"""
        u = self.usage
        return (f"Token: {u.total_tokens} "
                f"(in {u.prompt_tokens} / out {u.completion_tokens} / {u.call_count} 次调用)")
