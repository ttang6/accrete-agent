"""SubAgentTool：把一个受限子 agent 暴露给主 LLM 的 function calling（Agent-as-Tool）。

主 agent 调用时只给一个 task 字符串；tool 内部把它打包成隔离 SubAgentRequest 交给
SubAgentRunner 跑，只回结果。`is_subagent=True` 让 runner 裁剪子 agent registry 时
硬剔除它（递归 guard，子 agent 不得再开子 agent）。
"""

from __future__ import annotations

import json
from typing import Optional

from accrete.runtime.subagent import SubAgentContextBuilder, SubAgentRunner
from accrete.tool.base import BaseTool


class SubAgentTool(BaseTool):
    is_subagent = True   # 递归 guard 标记

    def __init__(
        self,
        *,
        name: str,
        description: str,
        system_prompt: str,
        runner: SubAgentRunner,
        allowed_tools: tuple[str, ...] = (),
        output_schema: Optional[dict] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        max_iterations: int = 2,
        input_param: str = "task",
    ):
        self._name = name
        self._description = description
        self._system_prompt = system_prompt
        self._runner = runner
        self._allowed_tools = tuple(allowed_tools)
        self._output_schema = output_schema
        self._model = model
        self._provider = provider
        self._max_iterations = max_iterations
        self._input_param = input_param

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                self._input_param: {"type": "string", "description": "交给子 agent 的任务描述"},
            },
            "required": [self._input_param],
        }

    def _execute(self, **kwargs) -> str:
        task = str(kwargs.get(self._input_param, "")).strip()
        request = SubAgentContextBuilder.build(
            system_prompt=self._system_prompt,
            blocks=[("task", task)],
            allowed_tools=self._allowed_tools,
            output_schema=self._output_schema,
            model=self._model,
            provider=self._provider,
            max_iterations=self._max_iterations,
        )
        result = self._runner.run(request)
        if not result.ok:
            return f"[sub-agent error] {result.error}"
        if self._output_schema is not None and result.structured is not None:
            return json.dumps(result.structured, ensure_ascii=False)
        return result.text
