"""fetch_hf.py — 拉取 Hugging Face Daily Papers。

LLM 调用：
  skill_exec(skill="ai-digest", script="fetch_hf",
             args={"max_results": 10, "sort": "hot"})

流程：
  stdin JSON → 调 HF daily_papers API → 格式化 markdown → stdout

sort 可选值：hot / rising / new。留空使用接口默认排序。
"""

import sys

import requests

from _common import ensure_utf8_stdout, print_error, read_args

HF_DAILY_PAPERS_API = "https://huggingface.co/api/daily_papers"
HTTP_TIMEOUT = 15
_SORT_MAP = {
    "hot": "Hot",
    "rising": "Rising",
    "new": "New",
}


def _format_paper(item: dict) -> str:
    paper = item.get("paper") or {}
    arxiv_id = paper.get("id", "")
    title = (paper.get("title") or item.get("title") or "").replace("\n", " ").strip()
    summary = (paper.get("summary") or "").replace("\n", " ").strip()
    upvotes = item.get("numUpvotes") or item.get("upvotes") or 0
    published = (paper.get("publishedAt") or "")[:10]
    github_repo = item.get("githubRepo") or paper.get("githubRepo") or ""

    authors_raw = paper.get("authors") or []
    authors = ", ".join(a.get("name", "") for a in authors_raw[:3])
    if len(authors_raw) > 3:
        authors += f", et al. (+{len(authors_raw) - 3})"

    abs_url = f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else ""
    code_line = f"  代码: {github_repo}\n" if github_repo else ""
    summary_trunc = summary[:400] + ("..." if len(summary) > 400 else "")

    return (
        f"- **{title}**\n"
        f"  arxiv_id: {arxiv_id}  |  upvotes: {upvotes}  |  发布: {published}\n"
        f"  作者: {authors}\n"
        f"  摘要: {summary_trunc}\n"
        f"{code_line}"
        f"  链接: {abs_url}"
    )


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()
    try:
        max_results = min(int(args.get("max_results") or 10), 20)
    except (TypeError, ValueError):
        print_error("max_results 必须是整数")
        return 1

    sort = (args.get("sort") or "").strip().lower()
    sort_param = _SORT_MAP.get(sort)

    params = {"limit": max_results}
    if sort_param:
        params["sort"] = sort_param

    try:
        resp = requests.get(HF_DAILY_PAPERS_API, params=params, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
    except requests.Timeout:
        print_error("HF API 请求超时")
        return 1
    except requests.HTTPError as e:
        print_error(f"HF API HTTP 错误: {e}")
        return 1
    except requests.RequestException as e:
        print_error(f"HF API 请求失败: {e}")
        return 1

    try:
        data = resp.json()
    except Exception as e:
        print_error(f"HF API JSON 解析失败: {e}")
        return 1

    if not isinstance(data, list) or not data:
        print("# Hugging Face 今日精选论文\n\n（今日暂无精选论文）")
        return 0

    papers = list(data[:max_results])
    print(f"# Hugging Face 今日精选论文（{len(papers)} 篇）")
    print()
    for item in papers:
        print(_format_paper(item))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
