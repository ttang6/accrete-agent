import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


def _args_hash(kwargs: dict) -> str:
    """对 kwargs 做稳定 hash（sorted-json sha256 前 12 位），用于 call_key 默认投影。"""
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

    call_key：per-op 计数 / 熔断键（见 BaseTool.call_key）。
    klass：policy 信号（transient/permanent/None）。
    is_mutating：不可逆写 → 熔断器 fail-fast（N=1，不自动重试）。
    retry_after：服务器建议的重试等待秒数（429/503 的 Retry-After），无则 None。
    """
    call_key: str
    klass: Optional[FailureKlass] = None
    is_mutating: bool = False
    retry_after: Optional[float] = None


@dataclass(frozen=True)
class ValidationOutcome:
    """`check_args` 富钩子的校验失败结果。中性位置（BaseTool 概念，不绑任何 skill）。

    `repair_text` 必填——下游（MainLoop / FailureMemory / OutcomeTracker）靠它的固定
    前缀识别为协议级失败。其余字段是可选的结构化修复信息，一般工具只给 repair_text 即可。
    """
    repair_text: str
    error_message: str = ""
    required_fields: list = field(default_factory=list)
    allowed_values: dict = field(default_factory=dict)
    repair_example: Optional[dict] = None


class BaseTool(ABC):
    # 每个 tool 自主声明是否启用 OpenAI strict mode。strict 下 provider 用约束解码
    # 强制 LLM 输出符合 schema（不再"软提示"），但要求 schema 满足额外约束：所有
    # properties 进 required（可选字段用 ["type","null"] 联合）+ 每个 object 加
    # additionalProperties:false + 不支持 minLength/pattern/format。
    #
    # 取舍按"这条约束 strict 能不能表达"分流，不按 provider 身份：schema 完全静态的
    # 工具（grep / glob）默认就把 strict_mode 开着，让 provider 兜住结构合规；
    # SkillExecTool 的 args 是 free-form object，strict 表达不了，保持 False、靠
    # 自己的 arg validator 每次校验。class 默认留 False 是因为多数现有工具 schema
    # 还没做到 strict-compliant（缺 additionalProperties:false / 有可选字段），贸然
    # 全局翻 True 会被 provider 当场拒——所以是各工具按能力 opt-in，而非全局默认。
    strict_mode: bool = False

    # SubAgent 递归 guard 标记：SubAgentTool 覆盖为 True。SubAgentRunner 裁剪
    # 子 agent registry 时硬剔除带此标记的 tool —— 子 agent 不得再开子 agent。
    is_subagent: bool = False

    # 写/有副作用工具标记：不应被确定性自动重试（避免重复副作用）。默认 False。
    is_mutating: bool = False

    def op(self, kwargs: dict) -> str:
        """**操作身份**（粗粒度）——工具自声明,全系统唯一共享的基座概念。

        「哪个操作」,层级结构一个字段,不含 args 指纹。默认 = tool_name;子类覆盖给
        更合适的粒度（fetch→'fetch'、arxiv→'arxiv'、skill_exec→'skill_exec:skill/script'）。
        消费者:lesson 存储/召回键直接用 op;Reflector 观测恢复谓词 = op 相等 ∧ args_hash 不等。
        F1 单一携带源:emit 写进操作节点,热路径召回（FailureMemory）与冷路径 episode 存储
        （EpisodeExtractor）都读这把键,不再两处各算一遍靠手工对齐。
        """
        return self.name

    def args_hash(self, kwargs: dict) -> str:
        """**参数指纹**——本次调用参数的稳定 hash。op 的正交轴。

        默认 = 全 kwargs 的 canonical sha256[:12]。子类极少覆盖（op 已声明了粒度,
        args_hash 只回答"同一操作下哪组参数"）。
        """
        return _args_hash(kwargs or {})

    def call_key(self, kwargs: dict) -> str:
        """**细粒度调用键 = (op, args_hash) 派生物**——服务失败计数 / 熔断。

        默认 = `f"{op}:{args_hash}"`。同 op 同 args = 同 call_key（一次具体调用）。
        子类可覆盖给更可读的表示（如 fetch 直接嵌 URL 便于熔断消息可读),但语义仍是
        「同一操作的同一组参数」。**lesson 召回键用 op 不用 call_key**（后者太细配不上一类操作）。
        """
        return f"{self.op(kwargs)}:{self.args_hash(kwargs)}"

    def classify_failure(
        self, kwargs: dict, output: str, exc: Optional[BaseException] = None
    ) -> ToolFailure:
        """失败时返回结构化 ToolFailure（call_key + klass + is_mutating + retry_after）。

        默认 klass=None（best-effort：工具说不清 → 交给 LLM / 退化）。能从 typed
        信号判定 klass 的工具（如 fetch 从 requests.exceptions.*）覆盖本方法给出
        硬 klass。只在 result 已判定为失败时调用。
        """
        return ToolFailure(
            call_key=self.call_key(kwargs), klass=None, is_mutating=self.is_mutating
        )

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    def validate(self, **kwargs) -> Optional[str]:
        return None

    def check_args(self, **kwargs) -> Optional[ValidationOutcome]:
        """运行期富钩子：validate 之后、_execute 之前的一次结构化 arg 校验。

        默认返 None（不拦）。有声明式 schema 的工具（如 SkillExecTool）override 它——
        校验失败返 ValidationOutcome，run() 直接把 repair_text 当 tool_result 返回、不进
        _execute。与 strict_mode 正交：strict 是解码期 provider 拦，check_args 是运行期拦。
        向后兼容：默认 None → 现有工具零迁移（validate 签名不变）。
        """
        return None

    def run(self, **kwargs) -> str:
        error = self.validate(**kwargs)
        if error:
            return f"[参数错误] {error}"
        outcome = self.check_args(**kwargs)
        if outcome is not None:
            return outcome.repair_text
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
