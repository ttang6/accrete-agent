# provider.py
from __future__ import annotations

from typing import Any, Protocol
from .types import LLMResponse, Message


class LLMProvider(Protocol):
    def complete(
        self,
        messages: list[Message],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...