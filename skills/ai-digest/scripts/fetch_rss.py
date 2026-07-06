"""fetch_rss.py — 从 sources.yaml 里的 RSS/Atom 源拉近期文章。

LLM 调用：
  skill_exec(skill="ai-digest", script="fetch_rss",
             args={"category": "AI Research", "max_results": 15})

流程：
  stdin JSON → _common.load_sources(category) → 并发 feedparser 解析
  → 48h 窗口过滤 → 去噪黑名单 → guid/link 去重
  → high priority 源或摘要过短的并发 trafilatura 全文提取
  → 按源内原始次序稳定排序 → stdout 格式化 markdown

category 可选值：AI Research / AI Engineering / Open Source / Industry News。空 = 全部。
"""

import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional

import fastfeedparser as feedparser
import requests
import trafilatura

from _common import ensure_utf8_stdout, load_sources, print_error, read_args

_NOISE_KEYWORDS = {
    "hiring", "we're hiring", "job", "jobs", "career",
    "event", "webinar", "conference", "meetup",
    "fundraising", "funding", "series a", "series b",
    "press release", "partnership", "sponsor",
}

_WINDOW_HOURS = 48
_USER_AGENT = "Mozilla/5.0 (compatible; accrete/0.2)"
_FEED_WORKERS = 10
_FULLTEXT_WORKERS = 8
_FULLTEXT_TIMEOUT = 10


def _parse_entry_datetime(entry) -> Optional[datetime]:
    """兼容 feedparser 的 tuple 字段 + fastfeedparser 的 ISO 字符串字段。

    feedparser 给 `*_parsed` 9-tuple；fastfeedparser 只给 `published` / `updated`
    ISO 8601 字符串。两种都试，都没拿到 → None（调用方按 fail-closed 处理）。
    """
    # feedparser tuple 风格
    for attr in ("published_parsed", "updated_parsed", "created_parsed"):
        value = getattr(entry, attr, None)
        if value:
            try:
                return datetime(*value[:6], tzinfo=timezone.utc)
            except Exception:
                pass
    # fastfeedparser 字符串风格
    for attr in ("published", "updated", "created"):
        value = entry.get(attr) if hasattr(entry, "get") else None
        if isinstance(value, str) and value:
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except Exception:
                pass
    return None


def _is_recent(entry, cutoff: datetime) -> bool:
    dt = _parse_entry_datetime(entry)
    if dt is None:
        # 没有任何可解析的时间字段 → 拒绝，避免把源的历史 entries 全放行
        return False
    return dt >= cutoff


def _format_pub_date(entry) -> str:
    dt = _parse_entry_datetime(entry)
    return dt.strftime("%Y-%m-%d") if dt else ""


def _is_noise(title: str) -> bool:
    lower = (title or "").lower()
    return any(keyword in lower for keyword in _NOISE_KEYWORDS)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", " ", text).strip()


def _extract_summary(entry) -> str:
    summary = ""
    content = entry.get("content")
    if content:
        first = content[0] if isinstance(content, list) and content else None
        if isinstance(first, dict):
            summary = first.get("value", "")
    if not summary:
        summary = entry.get("summary", "") or ""
    return _strip_html(summary)


def _fetch_source(source: dict) -> tuple[dict, object | None, str]:
    name = source.get("name", "未知")
    url = source.get("rss_url", "")
    if not url:
        return source, None, "missing rss_url"
    try:
        feed = feedparser.parse(url)
        return source, feed, ""
    except Exception as exc:
        return source, None, str(exc)


def _fetch_fulltext(url: str) -> str:
    try:
        resp = requests.get(
            url,
            timeout=_FULLTEXT_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
        return (
            trafilatura.extract(
                resp.text,
                include_comments=False,
                include_tables=False,
            )
            or ""
        )
    except Exception:
        return ""


def _format_article(article: dict) -> str:
    summary = article["summary"]
    summary_trunc = summary[:500] + ("..." if len(summary) > 500 else "")
    return (
        f"- **{article['title']}**\n"
        f"  来源: {article['source']}  |  发布: {article['published']}  |  分类: {article['category']}\n"
        f"  {summary_trunc}\n"
        f"  链接: {article['link']}"
    )


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()
    category = (args.get("category") or "").strip() or None
    try:
        max_results = min(int(args.get("max_results") or 15), 30)
    except (TypeError, ValueError):
        print_error("max_results 必须是整数")
        return 1

    sources = load_sources(category=category)
    if not sources:
        if category:
            print(f"[rss] 分类 '{category}' 无订阅源")
        else:
            print("[rss] 无订阅源（sources.yaml 缺失或格式错）")
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=_WINDOW_HOURS)
    seen: set[str] = set()
    articles: list[dict] = []

    indexed_sources = list(enumerate(sources))
    feed_workers = min(_FEED_WORKERS, max(1, len(indexed_sources)))
    with ThreadPoolExecutor(max_workers=feed_workers) as executor:
        future_map = {
            executor.submit(_fetch_source, source): source_index
            for source_index, source in indexed_sources
        }
        for future in as_completed(future_map):
            source_index = future_map[future]
            source, feed, _error = future.result()
            if feed is None:
                continue

            entries = getattr(feed, "entries", []) or []
            name = source.get("name", "未知")
            priority = source.get("priority", "normal")
            cat = source.get("category", "")
            for entry_index, entry in enumerate(entries):
                link = entry.get("link", "")
                guid = entry.get("id", link)
                dedup_key = guid or link
                if not link or dedup_key in seen:
                    continue

                title = (entry.get("title") or "").strip()
                if not title or _is_noise(title):
                    continue
                if not _is_recent(entry, cutoff):
                    continue

                seen.add(dedup_key)
                summary = _extract_summary(entry)
                articles.append(
                    {
                        "_order": (source_index, entry_index),
                        "_needs_fulltext": (priority == "high" or len(summary) < 200),
                        "title": title,
                        "link": link,
                        "source": name,
                        "published": _format_pub_date(entry),
                        "summary": summary,
                        "category": cat,
                    }
                )

    if not articles:
        print(f"# RSS 近 {_WINDOW_HOURS}h 无新文章")
        return 0

    fulltext_targets = [a for a in articles if a["_needs_fulltext"]]
    if fulltext_targets:
        fulltext_workers = min(_FULLTEXT_WORKERS, max(1, len(fulltext_targets)))
        with ThreadPoolExecutor(max_workers=fulltext_workers) as executor:
            future_map = {
                executor.submit(_fetch_fulltext, a["link"]): a
                for a in fulltext_targets
            }
            for future in as_completed(future_map):
                article = future_map[future]
                fulltext = future.result()
                if fulltext and len(fulltext) > len(article["summary"]):
                    article["summary"] = fulltext

    articles.sort(key=lambda a: a["_order"])
    articles = articles[:max_results]

    print(f"# RSS 订阅近期文章（{len(articles)} 篇，近 {_WINDOW_HOURS}h）")
    print()
    for article in articles:
        print(_format_article(article))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
