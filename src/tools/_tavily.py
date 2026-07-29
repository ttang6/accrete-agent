"""Tavily 客户端构造与错误映射。"""

from tavily import TavilyClient
# ForbiddenError 与 TimeoutError 未在 tavily 顶层导出，统一从 errors 子模块取。
from tavily.errors import (BadRequestError, ForbiddenError, InvalidAPIKeyError,
                           MissingAPIKeyError, UsageLimitExceededError)
from tavily.errors import TimeoutError as TavilyTimeoutError

from core.tool import ToolExecutionError


def make_client(api_key: str) -> TavilyClient:
    """构造 Tavily 客户端。缺 key 是组装期错误，不要留到运行时才失败。"""
    if not api_key:
        raise ValueError("缺少 Tavily API key")
    return TavilyClient(api_key=api_key)


def as_tool_error(exc: Exception) -> ToolExecutionError:
    """把 Tavily 异常收敛到 ToolResult 允许的错误分类。

    额度耗尽只能落进 exec_error，会和真正的执行失败混在同一类里；消息中保留厂商
    异常名，让轨迹分析仍能把「账号没额度」和「这个网站抓不动」分开。
    """
    if isinstance(exc, TavilyTimeoutError):
        return ToolExecutionError(f"Tavily 请求超时: {exc}", "timeout")
    if isinstance(exc, BadRequestError):
        return ToolExecutionError(f"Tavily 拒绝了请求参数: {exc}", "schema_error")
    if isinstance(exc, (ForbiddenError, InvalidAPIKeyError, MissingAPIKeyError)):
        return ToolExecutionError(f"Tavily 拒绝访问 ({type(exc).__name__}): {exc}", "permission")
    if isinstance(exc, UsageLimitExceededError):
        return ToolExecutionError(f"Tavily 额度耗尽 (UsageLimitExceededError): {exc}", "exec_error")
    return ToolExecutionError(f"Tavily 调用失败 ({type(exc).__name__}): {exc}", "exec_error")
