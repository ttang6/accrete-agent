import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Literal, Optional


def _args_hash(kwargs: dict) -> str:
    """对 kwargs 做稳定 hash（sorted-json sha256 前 12 位），用于 op_key 默认投影。"""
    try:
        canonical = json.dumps(
            kwargs, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        )
    except (TypeError, ValueError):
        canonical = str(sorted(kwargs.items())) if isinstance(kwargs, dict) else str(kwargs)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


# klass = policy 信号（不是 key）：熔断器据此选 N + 是否退避。
#   transient  → 短暂可恢复（超时 / 连接 / 5xx / 429）：值得重试，N=2 + 退避
#   permanent  → 不会自愈（4xx client error）：别重试，N=1
#   None       → 工具说不清（subprocess 字符串 / 未知）：best-effort，N=2 无退避
FailureKlass = Literal["transient", "permanent"]


@dataclass
class ToolFailure:
    """工具对一次失败的结构化自述。熔断器 / FailureMemory 读字段，不猜字符串。

    op_key：per-op 计数 / 熔断键（见 BaseTool.op_key）。
    klass：policy 信号（transient/permanent/None）。
    is_mutating：不可逆写 → 熔断器 fail-fast（N=1，不自动重试）。
    retry_after：服务器建议的重试等待秒数（429/503 的 Retry-After），无则 None。
    """
    op_key: str
    klass: Optional[FailureKlass] = None
    is_mutating: bool = False
    retry_after: Optional[float] = None


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

    # 写/有副作用工具标记：不应被确定性自动重试（避免重复副作用）。默认 False。
    is_mutating: bool = False

    def op_key(self, kwargs: dict) -> str:
        """失败计数 / 熔断的 operation key —— 工具自声明的 args 投影。

        默认：`tool_name:<全 args 的 hash>`（不同 args 集 = 不同 op）。子类覆盖以
        声明更合适的粒度（fetch→url、arxiv→action、skill_exec→skill/script:args）。
        FailureMemory 的 per-op 计数与（后续）熔断器都用它做键 —— 键由工具拥有、
        不由框架按 tool_name 猜粒度。kwargs 是本次调用解析后的参数 dict。
        """
        return f"{self.name}:{_args_hash(kwargs or {})}"

    def classify_failure(
        self, kwargs: dict, output: str, exc: Optional[BaseException] = None
    ) -> ToolFailure:
        """失败时返回结构化 ToolFailure（op_key + klass + is_mutating + retry_after）。

        默认 klass=None（best-effort：工具说不清 → 交给 LLM / 退化）。能从 typed
        信号判定 klass 的工具（如 fetch 从 requests.exceptions.*）覆盖本方法给出
        硬 klass。只在 result 已判定为失败时调用。
        """
        return ToolFailure(
            op_key=self.op_key(kwargs), klass=None, is_mutating=self.is_mutating
        )

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
