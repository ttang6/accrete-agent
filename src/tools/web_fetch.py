"""网页抓取工具：直连优先，正文抽取回退 Tavily。"""

import httpx

from core.ports import OutputStore
from core.tool import AccessType, BaseTool, ToolExecutionError, clip_output

from ._tavily import as_tool_error, make_client

# 这些类型直连拿到就是可读文本，没必要再花一次 Tavily 额度去抽正文。
PLAIN_TEXT_TYPES = frozenset({
    "text/plain", "text/markdown", "text/csv", "text/xml",
    "application/json", "application/xml", "application/x-yaml",
})
USER_AGENT = "accrete-agent/0.1 (+https://github.com/ttang6/accrete-agent)"


class WebFetchTool(BaseTool):
    """抓取单个 URL 的正文内容。"""

    name = "web_fetch"
    description = (
        "抓取一个网址的正文并转成可读文本。纯文本、Markdown、JSON 等直接原样返回；"
        "HTML 页面会尝试剥掉样板，但对重度依赖脚本渲染的站点效果有限。"
        "读 GitHub 仓库里的文件时，用 raw.githubusercontent.com 的原始链接"
        "（如 https://raw.githubusercontent.com/<owner>/<repo>/main/README.md），"
        "直接抓仓库首页只会拿到导航栏。需要先找网址时用 web_search。"
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string",
                    "description": "完整网址，必须带 https:// 或 http:// 前缀，不能只写域名"},
        },
        "required": ["url"],
        "additionalProperties": False,
    }
    access_type = AccessType.NETWORK

    def __init__(self, api_key: str, max_output_chars: int = 8000, timeout_s: float = 20.0,
                 store: OutputStore | None = None) -> None:
        self.client = make_client(api_key)
        self.max_output_chars = max_output_chars
        self.timeout_s = timeout_s
        self.store = store

    def _run(self, args: dict) -> str:
        url = args["url"]
        if not url.startswith(("http://", "https://")):
            raise ToolExecutionError(f"只支持 http/https 网址，收到 {url!r}", "unsupported")
        text = self._fetch_plain(url)
        if text is None:
            text = self._extract(url)
        if not text.strip():
            raise ToolExecutionError(f"{url} 没有可读正文", "not_found")
        return clip_output(text, self.max_output_chars, self.store)

    def _fetch_plain(self, url: str) -> str | None:
        """直连抓取，只在确定是纯文本类型时采用。

        返回 None 表示「这个 URL 交给 Tavily 抽正文」——HTML 页面和直连失败都算，
        因为 Tavily 能处理 JS 渲染和反爬，而直连拿到的 HTML 还得自己剥样板。
        404 例外：那是确定的不存在，再走一次 Tavily 只会浪费额度并给出更含糊的错误。
        """
        try:
            response = httpx.get(url, timeout=self.timeout_s, follow_redirects=True,
                                 headers={"User-Agent": USER_AGENT})
        except httpx.TimeoutException as exc:
            raise ToolExecutionError(f"抓取 {url} 超时: {exc}", "timeout") from exc
        except httpx.HTTPError:
            return None
        if response.status_code == 404:
            raise ToolExecutionError(f"{url} 不存在（HTTP 404）", "not_found")
        if response.is_error:
            return None
        content_type = response.headers.get("content-type", "").split(";")[0].strip().lower()
        return response.text if content_type in PLAIN_TEXT_TYPES else None

    def _extract(self, url: str) -> str:
        """用 Tavily 抽取正文并转成 Markdown。"""
        try:
            data = self.client.extract(urls=[url], extract_depth="advanced", format="markdown")
        except Exception as exc:
            raise as_tool_error(exc) from exc
        results = data.get("results") or []
        if not results:
            failed = data.get("failed_results") or []
            reason = failed[0].get("error") if failed else "未返回内容"
            raise ToolExecutionError(f"抓取 {url} 失败: {reason}", "exec_error")
        return results[0].get("raw_content") or ""
