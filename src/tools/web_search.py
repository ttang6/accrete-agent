"""基于 Tavily 的网页搜索工具。"""

from core.ports import OutputStore
from core.tool import AccessType, BaseTool, ToolExecutionError, clip_output

from ._tavily import as_tool_error, make_client


class WebSearchTool(BaseTool):
    """按查询词检索网页，返回带来源链接的摘要列表。"""

    name = "web_search"
    description = (
        "搜索网页，返回若干条带标题、URL 和摘要的结果。"
        "用于查找事实、新闻和文档入口；已经知道具体网址、需要整页内容时改用 web_fetch。"
        "检索面向英文互联网，query 必须用英文写。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string",
                      "description": "英文检索词。任务是其它语言时，先自行译成英文再填这里，"
                                     "不要直接填原文——检索的是英文互联网。"},
            "max_results": {"type": "integer", "description": "返回结果条数，默认 5"},
            "topic": {"type": "string", "enum": ["general", "news"],
                      "description": "news 会优先返回时效性内容，默认 general"},
            "days": {"type": "integer", "description": "仅 topic=news 生效，限定最近几天，默认 7"},
            "include_domains": {"type": "array", "description": "只在这些域名内检索"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }
    access_type = AccessType.NETWORK

    def __init__(self, api_key: str, max_output_chars: int = 8000,
                 search_depth: str = "advanced", country: str | None = "united states",
                 store: OutputStore | None = None) -> None:
        self.client = make_client(api_key)
        self.max_output_chars = max_output_chars
        self.search_depth = search_depth
        self.country = country
        self.store = store

    def _run(self, args: dict) -> str:
        query = args["query"]
        topic = args.get("topic", "general")
        request = {"query": query, "topic": topic, "search_depth": self.search_depth,
                   "max_results": args.get("max_results", 5)}
        # country 只在 topic=general 时被 Tavily 接受，news 走时效性排序，不吃地区偏好。
        if topic == "general" and self.country:
            request["country"] = self.country
        if topic == "news":
            request["days"] = args.get("days", 7)
        if args.get("include_domains"):
            request["include_domains"] = args["include_domains"]
        try:
            data = self.client.search(**request)
        except Exception as exc:
            raise as_tool_error(exc) from exc

        results = data.get("results") or []
        if not results:
            raise ToolExecutionError(f"没有找到与 {query!r} 相关的结果", "not_found")
        blocks = [f"[{index}] {item.get('title') or '(无标题)'}\n{item.get('url') or ''}\n"
                  f"{(item.get('content') or '').strip()}"
                  for index, item in enumerate(results, 1)]
        return clip_output("\n\n".join(blocks), self.max_output_chars, self.store)
