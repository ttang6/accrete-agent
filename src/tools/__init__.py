"""内置工具实现包。

BUILTIN_TOOLS 是工具名到工具类的唯一映射，装配层按白名单从这里取类并实例化；
表里查不到的名字是组装期错误。
"""

from core.tool import BaseTool

from .web_fetch import WebFetchTool
from .web_search import WebSearchTool

BUILTIN_TOOLS: dict[str, type[BaseTool]] = {
    WebSearchTool.name: WebSearchTool,
    WebFetchTool.name: WebFetchTool,
}
