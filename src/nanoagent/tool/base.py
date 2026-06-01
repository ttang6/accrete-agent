from abc import ABC, abstractmethod
from typing import Callable, Optional


class BaseTool(ABC):
    # 每个 tool 自主声明是否启用 OpenAI strict mode。
    # strict 模式下 OpenAI 用约束解码强制 LLM 输出符合 schema（不再"软提示"），
    # 但要求 schema 满足额外约束：所有 properties 都进 required（可选用
    # ["type","null"] 联合）+ 每个 object 加 additionalProperties:false +
    # 不支持 minLength/pattern/format/etc。
    # 默认 False；只有 schema 已 strict-compliant 的子类（如 SkillExecTool）覆盖为 True。
    strict_mode: bool = False

    # SubAgent 递归 guard 标记：SubAgentTool 覆盖为 True。SubAgentRunner 裁剪
    # 子 agent registry 时硬剔除带此标记的 tool —— 子 agent 不得再开子 agent。
    is_subagent: bool = False

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    def validate(self, **kwargs) -> Optional[str]:
        return None

    def run(self, **kwargs) -> str:
        error = self.validate(**kwargs)
        if error:
            return f"[参数错误] {error}"
        try:
            return self._execute(**kwargs)
        except Exception as e:
            return f"[执行错误] {self.name}: {e}"

    @abstractmethod
    def _execute(self, **kwargs) -> str: ...

    def __repr__(self) -> str:
        return f"<Tool: {self.name}>"

    @property
    @abstractmethod
    def parameters(self) -> dict:
        """工具的参数定义（JSON Schema 格式），子类必须实现。

        LLM 通过此 schema 知道如何传参，每个参数都应有明确的名称和描述。
        示例:
            return {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                },
                "required": ["query"],
            }
        """
        ...

    def to_openai_schema(self) -> dict:
        function: dict = {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }
        if self.strict_mode:
            # OpenAI per-function strict flag。每个 function 独立开关，可与
            # 默认软模式 tool 同时存在于一次请求里。
            function["strict"] = True
        return {"type": "function", "function": function}


class FunctionTool(BaseTool):
    """将普通函数包装成 BaseTool，走完整的 validate → _execute 链路。

    用法：
        tool = FunctionTool("greet", "打招呼", lambda name: f"Hello, {name}!")
        tool.run(name="World")  # → "Hello, World!"

    通常不直接构造，而是通过 ToolRegistry.tool() 装饰器创建。
    """

    def __init__(self, name: str, description: str, func: Callable,
                 params: Optional[dict] = None):
        self._name = name
        self._description = description
        self._func = func
        self._params = params or self._infer_parameters(func)

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._params

    def _execute(self, **kwargs) -> str:
        result = self._func(**kwargs)
        return str(result)

    @staticmethod
    def _infer_parameters(func: Callable) -> dict:
        """从函数签名自动推断 JSON Schema 参数定义。"""
        import inspect
        sig = inspect.signature(func)
        properties = {}
        required = []
        for pname, param in sig.parameters.items():
            if pname in ("self", "kwargs"):
                continue
            prop = {"type": "string", "description": pname}
            properties[pname] = prop
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }
