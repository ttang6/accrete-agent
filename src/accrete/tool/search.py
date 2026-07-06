"""网页搜索工具：纯数据管道，不做翻译/总结。

设计原则：
  - Tool 只负责调 API、返回原始结果
  - 翻译、总结、筛选、判断相关性 → 全部交给主循环的 LLM
  - 支持 Tavily 和 Exa，优先 Tavily；Exa 用于学术、论文和语义检索补充
"""

import os
from urllib.parse import urlparse

import requests
from accrete.tool.base import BaseTool


class SearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web and return relevant page titles, snippets, and URLs. "
            "Use this for current information, news, technical documentation, "
            "academic topics, and other web-backed queries. "
            "Use source=tavily for general web search and current information. "
            "Use source=exa for academic papers, research literature, semantic "
            "search, and deeper technical discovery. "
            "For Tavily news searches, set topic=news. "
            "When generating search queries: default to English regardless of "
            "the user's input language, because English yields higher-quality "
            "results for most technical, academic, and general topics. Only "
            "use non-English when the topic is inherently local, such as local "
            "news, regional policy, or culture-specific content. "
            "Return raw search results without summarizing them."
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Search query. Default to English regardless of the user's "
                        "input language; only use non-English for inherently local "
                        "topics such as local news, regional policy, or "
                        "culture-specific content."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default is 5.",
                },
                "source": {
                    "type": "string",
                    "enum": ["tavily", "exa", "auto"],
                    "description": (
                        "Search backend. Default tavily. Use exa for academic "
                        "papers, research literature, semantic search, and deeper "
                        "technical discovery."
                    ),
                },
                "topic": {
                    "type": "string",
                    "enum": ["general", "news"],
                    "description": (
                        "Tavily topic. Default general. Use news for current events, "
                        "breaking news, politics, sports, and mainstream news coverage."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": (
                        "For Tavily topic=news only: number of days back to include. "
                        "Default is Tavily's service default."
                    ),
                },
            },
            "required": ["query"],
        }

    def _execute(self, query: str = "", max_results: int = 5,
                 source: str = "tavily", topic: str = "general",
                 days: int | None = None, **kwargs) -> str:
        query = query.strip()
        if not query:
            return "搜索关键词不能为空。"

        tavily_key = os.getenv("TAVILY_API_KEY")
        exa_key = os.getenv("EXA_API_KEY")
        source = (source or "tavily").strip().lower()
        topic = (topic or "general").strip().lower()

        if topic not in {"general", "news"}:
            topic = "general"
        if source not in {"tavily", "exa", "auto"}:
            source = "tavily"

        if source == "exa":
            if not exa_key:
                return "未配置 Exa API Key。请配置 EXA_API_KEY。"
            results, error = self._search_exa(query, exa_key, max_results)
            if results:
                return self._format(query, results, "Exa")
            return f"搜索失败: Exa: {error}"

        # 默认 Tavily；失败时用 Exa 兜底（如果已配置）。
        if tavily_key:
            results, error = self._search_tavily(
                query, tavily_key, max_results, topic, days
            )
            if results:
                return self._format(query, results, f"Tavily/{topic}")
            if exa_key:
                exa_results, exa_error = self._search_exa(query, exa_key, max_results)
                if exa_results:
                    return self._format(query, exa_results, "Exa")
                return f"搜索失败: Tavily: {error}; Exa: {exa_error}"
            return f"搜索失败: Tavily: {error}"

        if exa_key:
            results, error = self._search_exa(query, exa_key, max_results)
            if results:
                return self._format(query, results, "Exa")
            return f"搜索失败: Exa: {error}"

        return "未配置搜索 API Key。请配置 TAVILY_API_KEY 或 EXA_API_KEY。"

    def _search_tavily(self, query: str, api_key: str, max_results: int,
                       topic: str, days: int | None
                       ) -> tuple[list[dict], str]:
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "include_answer": False,
            "topic": topic,
        }
        if topic == "news" and days:
            payload["days"] = days

        try:
            resp = requests.post(
                "https://api.tavily.com/search",
                json=payload,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
            results = []
            for item in data.get("results", []):
                url = item.get("url", "")
                results.append({
                    "title": item.get("title", ""),
                    "snippet": item.get("content", ""),
                    "url": url,
                    "domain": self._domain(url),
                    "published_date": item.get("published_date", ""),
                    "score": item.get("score", ""),
                })
            return results, ""
        except Exception as e:
            return [], str(e)

    def _search_exa(self, query: str, api_key: str, max_results: int
                    ) -> tuple[list[dict], str]:
        try:
            from exa_py import Exa
            client = Exa(api_key=api_key)
            response = client.search(
                query,
                num_results=max_results,
                type="auto",
                contents={
                    "highlights": {
                        "num_sentences": 2,
                        "highlights_per_url": 2,
                    }
                },
            )
            results = []
            for item in response.results[:max_results]:
                url = getattr(item, "url", "") or ""
                highlights = getattr(item, "highlights", None) or []
                snippet = " [...] ".join(highlights)
                results.append({
                    "title": getattr(item, "title", "") or "",
                    "snippet": snippet,
                    "url": url,
                    "domain": self._domain(url),
                    "published_date": getattr(item, "published_date", "") or "",
                    "author": getattr(item, "author", "") or "",
                    "score": getattr(item, "score", "") or "",
                })
            return results, ""
        except Exception as e:
            return [], str(e)

    @staticmethod
    def _domain(url: str) -> str:
        try:
            return urlparse(url).netloc
        except Exception:
            return ""

    def _format(self, query: str, results: list[dict], source: str) -> str:
        lines = [
            "Search results",
            f"Query: {query}",
            f"Backend: {source}",
            f"Result count: {len(results)}",
            "",
        ]
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. Title: {r.get('title', '')}")
            if r.get("url"):
                lines.append(f"   URL: {r['url']}")
            if r.get("domain"):
                lines.append(f"   Domain: {r['domain']}")
            if r.get("published_date"):
                lines.append(f"   Published: {r['published_date']}")
            if r.get("author"):
                lines.append(f"   Author: {r['author']}")
            if r.get("score") not in ("", None):
                lines.append(f"   Score: {r['score']}")
            if r.get("snippet"):
                lines.append(f"   Snippet: {r['snippet'][:500]}")
            lines.append("")
        return "\n".join(lines)
