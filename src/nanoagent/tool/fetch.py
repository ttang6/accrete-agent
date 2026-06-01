"""网页抓取工具：通用 fetch，公开四档 extract 模式（含 auto 启发式路由）。

设计原则：
  - Tool 保持通用，不硬编码 site 特化业务逻辑；但通过 auto 模式提供 URL 启发式
    路由，避免 LLM 每次都走默认 main 模式处理索引/列表/纯文本等不适合场景
  - LLM 一般不需要传 extract；auto 会覆盖普通文章正文抽取场景
  - LLM 仅在明确需要 markdown/raw/llm 时传 extract 显式覆盖 auto 路由
  - extractor_llm 通过构造函数注入，可以用更便宜的模型

公开 extract 模式：
  auto（默认）：按 URL 启发式路由 main / markdown / raw。**不会路由到 llm**
                （llm 需要 extract_prompt，无法自动生成）
  markdown    ：html2text 转 Markdown，保留结构信号，适合列表/索引/卡片页
  raw         ：只去 HTML 标签，返回原始文本，兜底用
  llm         ：先 markdown 预处理，再用 extractor_llm 按 extract_prompt 提取

内部模式：
  main        ：Trafilatura 通用正文抽取，适合博客/论文摘要页/新闻，由 auto 选择

auto 路由规则（当前）：
  - URL 结尾是 .md / .txt / .rst → raw（纯文本文件）
  - 包含 github.com/trending / news.ycombinator.com / reddit.com/r/ / arxiv.org/list
    / openreview.net/group 等索引页特征 → markdown
  - 其他 → main

示例：
    fetch(url="https://example.com/blog/post")              # auto → main（内部）
    fetch(url="https://github.com/trending")                # auto → markdown
    fetch(url="https://raw.githubusercontent.com/a/b.md")   # auto → raw
    fetch(url="...", extract="llm",                          # 显式 LLM 提取
          extract_prompt="列出前 10 篇论文")
"""

import html2text
import requests
import trafilatura
from bs4 import BeautifulSoup
from nanoagent.core.logger import get_logger
from nanoagent.tool.base import BaseTool

_logger = get_logger("fetch")


def _build_html2text() -> html2text.HTML2Text:
    """构造统一的 html2text 转换器（保留链接，不折行）。"""
    h = html2text.HTML2Text()
    h.ignore_links = False
    h.ignore_images = True     # 图片对 LLM 没用，省 token
    h.body_width = 0            # 不按列宽折行，保持原行结构
    h.skip_internal_links = True
    h.ignore_emphasis = False
    return h


# ============================================================
# Auto routing
# ============================================================

# 索引/列表页特征（命中任一 → markdown）。加条目的判据：
# 该 URL 类型通常是"条目列表式"结构，markdown 能保留 heading / list / link 结构
# 让 LLM 读出条目；走 main 会被 Trafilatura 按"文章式"抽取，列表被吃掉。
_MARKDOWN_INDICATORS = (
    "github.com/trending",
    "news.ycombinator.com",
    "reddit.com/r/",
    "arxiv.org/list",
    "openreview.net/group",
)

# 纯文本扩展名（命中任一 → raw）。
_RAW_SUFFIXES = (".md", ".txt", ".rst")


def _auto_route_mode(url: str) -> str:
    """根据 URL 启发式选择 extract 模式。**不选 llm**（它需要额外 extract_prompt）。"""
    lower = url.lower()

    for ind in _MARKDOWN_INDICATORS:
        if ind in lower:
            return "markdown"

    # 先去掉 query string 再判扩展名，避免 '?foo=bar' 影响
    path = lower.split("?", 1)[0]
    for suf in _RAW_SUFFIXES:
        if path.endswith(suf):
            return "raw"

    return "main"


# ============================================================
# Tool
# ============================================================


class FetchTool(BaseTool):
    def __init__(self, extractor_llm=None):
        """
        Args:
            extractor_llm: 可选的 LLM 客户端，仅 extract="llm" 时使用。
                          不传则 extract="llm" 会降级到 markdown 模式。
        """
        self.extractor_llm = extractor_llm

    @property
    def name(self) -> str:
        return "fetch"

    @property
    def description(self) -> str:
        return (
            "抓取指定 URL 的页面内容。支持 4 种对外抽取模式：\n"
            "- auto（默认）：按 URL 启发式选 main/markdown/raw。普通文章、博客、文档页不要传 extract，用 auto 即可\n"
            "- markdown：HTML→Markdown，适合 arxiv list / github trending / HN 等索引页\n"
            "- raw：只去 HTML 标签，兜底用\n"
            "- llm：需要同时提供 extract_prompt，先 markdown 再 LLM 按指令提炼\n"
            "不要显式传 main；auto 会在适合时自动使用内部 main 抽取。"
        )

    @property
    def parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "要抓取的网页 URL",
                },
                "extract": {
                    "type": "string",
                    "enum": ["auto", "markdown", "raw", "llm"],
                    "description": (
                        "抽取模式。默认 auto，通常应省略此参数。"
                        "只有明确需要保留列表/链接结构时选 markdown，"
                        "需要纯文本兜底时选 raw，需要按指令提炼时选 llm。"
                    ),
                },
                "extract_prompt": {
                    "type": "string",
                    "description": "仅当 extract=llm 时使用的提取指令，例如 '列出前 10 篇论文的标题和摘要'",
                },
            },
            "required": ["url"],
        }

    def _execute(self, url: str = "", extract: str = "auto",
                 extract_prompt: str = "", **kwargs) -> str:
        url = url.strip()
        if not url:
            return "URL 不能为空。"
        if not url.startswith("http"):
            return "请提供完整的 URL，以 http:// 或 https:// 开头。"

        # PDF 早退：二进制 PDF 文本抽取没意义，给 LLM 明确提示
        lower_url = url.lower()
        if lower_url.endswith(".pdf") or "/pdf/" in lower_url:
            return (
                f"这是 PDF 文件 ({url})，fetch 无法解析 PDF 二进制内容。\n"
                f"如果是 arxiv 论文，请改用 /abs/ 页面（如 https://arxiv.org/abs/xxx）。"
            )

        requested_mode = (extract or "auto").strip().lower()
        if requested_mode == "main":
            requested_mode = "auto"
        if requested_mode not in {"auto", "markdown", "raw", "llm"}:
            requested_mode = "auto"

        # 解析 auto 模式
        resolved_mode = requested_mode
        if requested_mode == "auto":
            resolved_mode = _auto_route_mode(url)
            _logger.info(f"[fetch] auto-route: {url} → {resolved_mode}")

        # 下载页面
        _logger.info(f"[fetch] GET {url} (extract={requested_mode}, resolved={resolved_mode})")
        try:
            resp = requests.get(
                url,
                timeout=15,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            _logger.warning(f"[fetch] timeout: {url}")
            return f"请求超时: {url}"
        except requests.exceptions.HTTPError as e:
            _logger.warning(f"[fetch] HTTP error on {url}: {e}")
            return f"HTTP 错误: {e}"
        except requests.exceptions.RequestException as e:
            _logger.warning(f"[fetch] request error on {url}: {e}")
            return f"请求失败: {e}"

        html = resp.text
        metadata = self._extract_metadata(html, resp.url)
        _logger.debug(f"[fetch] got {len(html)} chars HTML from {url}")

        # 按模式分派
        if resolved_mode == "raw":
            content = self._extract_raw(html)
            return self._format_result(metadata, requested_mode, resolved_mode, content)
        if resolved_mode == "markdown":
            content = self._extract_markdown(html)
            return self._format_result(metadata, requested_mode, resolved_mode, content)
        if resolved_mode == "llm":
            content = self._extract_llm(html, extract_prompt, url)
            return self._format_result(metadata, requested_mode, resolved_mode, content)
        # 默认 main（含 auto 路由到 main 的情况）
        content = self._extract_main(html)
        return self._format_result(metadata, requested_mode, resolved_mode, content)

    def _extract_metadata(self, html: str, url: str) -> dict:
        """抽取轻量页面元数据，用于工具输出头部，不参与正文清洗。"""
        metadata = {
            "url": url,
            "title": "",
            "published_time": "",
            "description": "",
        }
        try:
            soup = BeautifulSoup(html, "html.parser")
            if soup.title and soup.title.string:
                metadata["title"] = soup.title.string.strip()

            selectors = {
                "published_time": [
                    ("property", "article:published_time"),
                    ("name", "pubdate"),
                    ("name", "publishdate"),
                    ("name", "date"),
                ],
                "description": [
                    ("name", "description"),
                    ("property", "og:description"),
                ],
            }
            for field, candidates in selectors.items():
                for attr, value in candidates:
                    tag = soup.find("meta", attrs={attr: value})
                    if tag and tag.get("content"):
                        metadata[field] = tag["content"].strip()
                        break
        except Exception:
            pass
        return metadata

    def _format_result(self, metadata: dict, requested_mode: str,
                       resolved_mode: str, content: str) -> str:
        lines = [
            "Page content",
            f"URL: {metadata.get('url', '')}",
            f"Title: {metadata.get('title', '') or '(unknown)'}",
            f"Requested extract: {requested_mode}",
            f"Resolved extract: {resolved_mode}",
            f"Content characters: {len(content)}",
        ]
        if metadata.get("published_time"):
            lines.append(f"Published: {metadata['published_time']}")
        if metadata.get("description"):
            lines.append(f"Description: {metadata['description'][:500]}")
        lines.extend(["", "Content:", content])
        return "\n".join(lines)

    def _extract_main(self, html: str) -> str:
        """Trafilatura 通用正文抽取，失败降级到 raw。"""
        try:
            text = trafilatura.extract(html, include_comments=False,
                                        include_tables=True, favor_recall=True)
            if text and text.strip():
                return text
        except Exception:
            pass
        # 降级：Trafilatura 拿不到正文时回退到 raw 清理
        return self._extract_raw(html)

    def _extract_markdown(self, html: str) -> str:
        """HTML → Markdown：保留 heading、list、link 等结构，不做正文抽取。

        适合列表/索引/卡片页——LLM 读 Markdown 能识别出结构化条目，
        比 Trafilatura 的"文章式"正文抽取靠谱。
        """
        # 先用 BS4 去掉脚本/样式等纯污染节点，其余交给 html2text
        try:
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "svg", "link", "meta", "noscript"]):
                tag.decompose()
            cleaned_html = str(soup)
        except Exception:
            cleaned_html = html

        try:
            md = _build_html2text().handle(cleaned_html)
            # 折叠 3 个以上连续空行，保持可读性
            lines = md.splitlines()
            compact = []
            blank = 0
            for line in lines:
                if line.strip() == "":
                    blank += 1
                    if blank <= 1:
                        compact.append(line)
                else:
                    blank = 0
                    compact.append(line)
            return "\n".join(compact).strip()
        except Exception as e:
            _logger.warning(f"[fetch] html2text failed: {e}, fallback to raw")
            return self._extract_raw(html)

    def _extract_raw(self, html: str) -> str:
        """BeautifulSoup 去标签，返回纯文本（兜底模式）。"""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines)

    def _extract_llm(self, html: str, extract_prompt: str, url: str) -> str:
        """先 html2text 转 Markdown（保结构），再让 LLM 按 prompt 提取。

        对索引页效果远好于用 Trafilatura 抽正文——Markdown 保留了
        heading/list/link，LLM 能识别出"这是论文列表"、"这是项目卡片"。
        """
        if self.extractor_llm is None:
            # 没配 LLM → 降级到 markdown 并提示
            md = self._extract_markdown(html)
            return (
                "[warning] extract=llm 需要 extractor_llm，但未配置。"
                "已降级到 markdown 模式。\n\n" + md
            )

        if not extract_prompt.strip():
            return "[参数错误] extract=llm 时必须提供 extract_prompt"

        # 用 markdown 作为 LLM 的输入——保结构比纯文本好得多
        md = self._extract_markdown(html)
        if not md.strip():
            _logger.warning(f"[fetch] empty markdown content for {url}")
            return f"页面内容为空或无法抽取: {url}"

        # 限制喂给 LLM 的长度，防止超 context
        if len(md) > 20000:
            md = md[:20000] + "\n...(页面过长，已截断)"

        prompt = (
            f"以下是从 URL {url} 抓取并转为 Markdown 的页面内容（保留了列表/标题/链接结构）。\n"
            f"请根据下面的指令从中提取信息：\n\n"
            f"指令：{extract_prompt}\n\n"
            f"---页面 Markdown 开始---\n{md}\n---页面 Markdown 结束---\n\n"
            f"直接输出提取结果，保持结构清晰。如果页面中没有对应信息，说明'未找到'。"
        )

        _logger.info(f"[fetch] extract=llm calling extractor_llm "
                     f"(prompt_chars={len(prompt)}) for {url}")
        try:
            result = self.extractor_llm.think(
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            _logger.info(f"[fetch] extract=llm done ({len(result)} chars) for {url}")
            return result
        except Exception as e:
            # LLM 调用失败 → fallback 到 markdown 原文
            _logger.error(f"[fetch] extract=llm FAILED for {url}: {e}")
            return (
                f"[warning] LLM 提取失败 ({e})，返回原始 Markdown：\n\n"
                + md
            )
