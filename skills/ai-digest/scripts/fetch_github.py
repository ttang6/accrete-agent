"""fetch_github.py — 拉取 GitHub Trending 仓库列表（HTML 抓取 + 按天缓存）。

LLM 调用：
  skill_exec(skill="ai-digest", script="fetch_github",
             args={"language": "python", "since": "daily", "max_results": 10})

流程：
  stdin JSON → 查当日缓存（data/cache/github_trending/）
              → 命中则直接输出；未命中抓 github.com/trending → BeautifulSoup 解析
              → 写缓存 → stdout 格式化 markdown

GitHub 没有官方 trending API，这里走 HTML 抓取 + 按天缓存是既定做法。
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup

from _common import ensure_utf8_stdout, print_error, read_args

# skill 根目录 = 本文件 → scripts/ → ai-digest/。向上再两级到项目根。
_SKILL_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SKILL_ROOT.parent.parent
_CACHE_DIR = _PROJECT_ROOT / "data" / "cache" / "github_trending"

HTTP_TIMEOUT = 15
_USER_AGENT = "Mozilla/5.0 (compatible; nanoagent/0.2)"


def _cache_path(language: str, since: str) -> Path:
    today = datetime.now().strftime("%Y-%m-%d")
    lang_key = language.lower().replace(" ", "-") if language else "all"
    return _CACHE_DIR / f"{today}_{lang_key}_{since}.json"


def _clean_text(value: str) -> str:
    return " ".join((value or "").split())


def _parse_trending(html: str, max_results: int) -> list[dict]:
    soup = BeautifulSoup(html, "lxml")
    repos = []
    for article in soup.select("article.Box-row")[:max_results]:
        name_tag = article.select_one("h2 a")
        if not name_tag:
            continue

        full_name = (name_tag.get("href", "") or "").lstrip("/")
        if not full_name:
            continue

        desc_tag = article.select_one("p")
        description = _clean_text(desc_tag.get_text()) if desc_tag else ""

        lang_tag = article.select_one("[itemprop='programmingLanguage']")
        language = _clean_text(lang_tag.get_text()) if lang_tag else ""

        stars = ""
        forks = ""
        for anchor in article.select("a"):
            href = anchor.get("href", "") or ""
            text = _clean_text(anchor.get_text())
            if not stars and "stargazers" in href:
                stars = text
            elif not forks and "network/members" in href:
                forks = text

        stars_today = ""
        stars_today_tag = article.select_one("span.d-inline-block.float-sm-right")
        if stars_today_tag:
            stars_today = _clean_text(stars_today_tag.get_text())

        repos.append(
            {
                "full_name": full_name,
                "description": description,
                "language": language,
                "stars": stars,
                "forks": forks,
                "stars_today": stars_today,
                "url": f"https://github.com/{full_name}",
            }
        )
    return repos


def _format_repo(repo: dict) -> str:
    meta_parts = []
    if repo.get("stars"):
        meta_parts.append(f"总 stars: {repo['stars']}")
    if repo.get("forks"):
        meta_parts.append(f"forks: {repo['forks']}")
    if repo.get("stars_today"):
        meta_parts.append(repo["stars_today"])
    meta_line = " | ".join(meta_parts)
    language_line = f"  语言: {repo['language']}\n" if repo.get("language") else ""

    return (
        f"- **{repo['full_name']}**\n"
        f"{language_line}"
        f"  {meta_line}\n"
        f"  {repo['description']}\n"
        f"  链接: {repo['url']}"
    )


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()

    language = (args.get("language") or "").strip()
    since = args.get("since") or "daily"
    if since not in ("daily", "weekly", "monthly"):
        since = "daily"
    try:
        max_results = min(int(args.get("max_results") or 10), 25)
    except (TypeError, ValueError):
        print_error("max_results 必须是整数")
        return 1

    cache_file = _cache_path(language, since)
    if cache_file.exists():
        try:
            repos = json.loads(cache_file.read_text(encoding="utf-8"))
            repos = repos[:max_results]
            print(f"# GitHub Trending（{since}，缓存）{len(repos)} 个仓库")
            print()
            for repo in repos:
                print(_format_repo(repo))
                print()
            return 0
        except Exception:
            pass

    url = f"https://github.com/trending/{language}?since={since}"
    try:
        resp = requests.get(
            url,
            timeout=HTTP_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
        )
        resp.raise_for_status()
    except requests.Timeout:
        print_error("GitHub Trending 请求超时")
        return 1
    except requests.HTTPError as e:
        print_error(f"GitHub Trending HTTP 错误: {e}")
        return 1
    except requests.RequestException as e:
        print_error(f"GitHub Trending 请求失败: {e}")
        return 1

    repos = _parse_trending(resp.text, max_results=25)
    if not repos:
        print("# GitHub Trending\n\n（未解析到仓库，页面结构可能已变更）")
        return 0

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(repos, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass

    repos = repos[:max_results]
    print(f"# GitHub Trending（{since}）{len(repos)} 个仓库")
    print()
    for repo in repos:
        print(_format_repo(repo))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
