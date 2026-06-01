"""通用 SubAgent 层：受限、隔离的子 agent 执行引擎。

镜像 Claude Code subagent 理念（https://code.claude.com/docs/en/sub-agents）：
- **context 隔离**：子 agent 全新开始，只见显式传入的 `input_blocks` + override system
  prompt；父对话历史 / memory / user_facts 一律不漏进来。这是本层的命门。
- **system prompt 完全 override**（不是叠加）。
- **工具 allowlist**：从父 registry 按名裁剪；硬剔除任何带 `is_subagent` 标记的 tool
  —— 递归 guard，子 agent 不得再开子 agent。
- **只回结果**：返回结构化 decision / 文本，中间过程不外泄。
- **便宜模型**：可 per-request override（如 distill 用 qwen-flash）。

本层不绑定任何用例。偏好蒸馏的 policy sub-agent（零工具）是第一个消费者。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from nanoagent.core.logger import get_logger
from nanoagent.tool.registry import ToolRegistry

_logger = get_logger("subagent")


class _LLMLike(Protocol):
    def think(self, messages: list[dict], temperature: float = 0) -> str: ...  # noqa: E704


# (model, provider) -> LLM 实例。注入式，便于测试替换。
LLMBuilder = Callable[[Optional[str], Optional[str]], _LLMLike]


@dataclass(frozen=True)
class SubAgentRequest:
    """一次子 agent 调用的自包含信封。子 agent 只见这里面的东西。"""

    system_prompt: str
    input_blocks: tuple[tuple[str, str], ...] = ()   # 具名 block：(name, content)
    allowed_tools: tuple[str, ...] = ()              # 工具名 allowlist；() = 零工具
    output_schema: Optional[dict] = None             # 非 None → 解析结构化 JSON
    model: Optional[str] = None                      # 模型 override；None = builder 默认
    provider: Optional[str] = None
    max_iterations: int = 2
    timeout_s: Optional[int] = None


@dataclass
class SubAgentResult:
    text: str = ""
    structured: Optional[dict] = None                # output_schema 解析成功时填
    ok: bool = True
    error: str = ""


class SubAgentContextBuilder:
    """通用打包：具名 block → 自包含 SubAgentRequest。

    价值不在"省几行"，而在统一所有 subagent 用例的**隔离入口**：经它产出的
    request 只含显式传入内容，杜绝父 context 隐式漏入。领域适配器（如偏好蒸馏）
    只负责产出 block，封装/隔离交给这里。
    """

    @staticmethod
    def build(
        *,
        system_prompt: str,
        blocks: list[tuple[str, str]],
        allowed_tools: tuple[str, ...] = (),
        output_schema: Optional[dict] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_iterations: int = 2,
        timeout_s: Optional[int] = None,
    ) -> SubAgentRequest:
        clean: list[tuple[str, str]] = []
        seen: set[str] = set()
        for name, content in blocks:
            name, content = str(name), str(content)
            if not content.strip():
                continue  # 空 block 丢弃，避免给 LLM 喂噪声
            if name in seen:
                raise ValueError(f"duplicate block name: {name!r}")
            seen.add(name)
            clean.append((name, content))
        return SubAgentRequest(
            system_prompt=system_prompt,
            input_blocks=tuple(clean),
            allowed_tools=tuple(allowed_tools),
            output_schema=output_schema,
            model=model,
            provider=provider,
            max_iterations=max_iterations,
            timeout_s=timeout_s,
        )


class SubAgentRunner:
    """构造受限、隔离的子 agent，跑一次，返回结果。fail-open（异常 → ok=False）。"""

    def __init__(self, *, llm_builder: LLMBuilder, base_tool_registry: Optional[ToolRegistry] = None):
        self._llm_builder = llm_builder
        self._base_registry = base_tool_registry   # 按 allowlist 裁剪的来源

    def run(self, request: SubAgentRequest) -> SubAgentResult:
        try:
            llm = self._llm_builder(request.model, request.provider)
            user_content = self._render_blocks(request.input_blocks)
            if not request.allowed_tools:
                # 零工具：单次结构化 think（避免空 tools 列表的 function-calling 兼容问题）
                messages = [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": user_content},
                ]
                text = llm.think(messages=messages, temperature=0)
            else:
                text = self._run_with_tools(request, llm, user_content)
        except Exception as e:
            _logger.warning(f"[subagent] run 异常（已 fail-open）: {type(e).__name__}: {e}")
            return SubAgentResult(ok=False, error=f"{type(e).__name__}: {e}")

        result = SubAgentResult(text=text or "")
        if request.output_schema is not None:
            parsed = self._parse_json(text)
            if parsed is None:
                result.ok = False
                result.error = "output_schema set but response was not valid JSON"
            else:
                result.structured = parsed
        return result

    # ------------------------------------------------------------------

    @staticmethod
    def _render_blocks(blocks: tuple[tuple[str, str], ...]) -> str:
        """具名 block → 单条自包含 user content。父级任何东西不进这里。"""
        return "\n\n".join(f"# {name}\n{content}" for name, content in blocks)

    def build_trimmed_registry(self, allowed_tools: tuple[str, ...]) -> ToolRegistry:
        """按 allowlist 从父 registry 裁剪；硬剔除任何 is_subagent tool（递归 guard）。"""
        trimmed = ToolRegistry()
        if self._base_registry is None:
            return trimmed
        allow = set(allowed_tools)
        for tool in self._base_registry.get_all_tools():
            if getattr(tool, "is_subagent", False):
                continue  # 递归 guard
            if tool.name in allow:
                trimmed.register(tool)
        return trimmed

    def _run_with_tools(self, request: SubAgentRequest, llm: _LLMLike, user_content: str) -> str:
        from nanoagent.runtime.main_loop import MainLoop

        trimmed = self.build_trimmed_registry(request.allowed_tools)
        loop = MainLoop(
            name="subagent",
            llm=llm,
            tool_registry=trimmed,
            max_iterations=request.max_iterations,
            system_prompt=request.system_prompt,   # MainLoop 自己拼 system，故只传 user turn
        )
        return loop.run([{"role": "user", "content": user_content}], save_on_finish=True)

    @staticmethod
    def _parse_json(raw: str) -> Optional[dict]:
        """容忍 LLM 在 JSON 外包 ```json``` / 解释文字。失败返 None。"""
        text = (raw or "").strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match is None:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        return data if isinstance(data, dict) else None


def build_subagent_llm(model: Optional[str] = None, provider: Optional[str] = None) -> _LLMLike:
    """生产用默认 llm_builder：构造一个 instance_name='subagent' 的 LLMClient。"""
    from nanoagent.core.llm_client import LLMClient

    return LLMClient(model=model, provider=provider or None, instance_name="subagent")
