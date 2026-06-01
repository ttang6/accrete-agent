"""fetch_x_topic.py — 按 topic 搜 X 最近推文（仅作补漏，不作主入口）。

LLM 调用：
  skill_exec(skill="ai-digest", script="fetch_x_topic", args={
      "query": "(from:OpenAI OR from:AnthropicAI) (agent OR eval) -is:retweet lang:en",
      "since_time": "2026-04-23T00:00:00Z",   # 可选 ISO-8601 UTC
      "max_results": 20                         # 10-100
  })

**契约**：query 必填，且应为**精确的 boolean / operator 查询**，不鼓励泛关键词搜索。
"人驱动"优先（fetch_x_users / fetch_x_curated），本 script 用于补漏已知账号列表之外
的高信号讨论。

依赖 env：X_BEARER_TOKEN
"""

import asyncio
import sys

from _common import ensure_utf8_stdout, print_error, read_args
from _x_api import (
    X_BEARER_TOKEN,
    XApiClient,
    XApiError,
    format_tweet,
    parse_tweets,
)


async def _run(query: str, since_time: str, max_results: int) -> int:
    client = XApiClient(bearer_token=X_BEARER_TOKEN)
    payload = await client.search_recent_tweets(
        query=query,
        max_results=max_results,
        start_time=since_time or None,
    )
    tweets = parse_tweets(payload)
    tweets.sort(key=lambda t: (t.get("created_at") or ""), reverse=True)

    if not tweets:
        print(f"# X topic 搜索（query 无匹配推文）\n\nquery: `{query}`")
        return 0

    print(f"# X topic 搜索（{len(tweets)} 条）")
    print(f"\nquery: `{query}`")
    if since_time:
        print(f"since_time: `{since_time}`")
    print()
    for tw in tweets:
        print(format_tweet(tw))
        print()
    return 0


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()

    query = (args.get("query") or "").strip()
    if not query:
        print_error("query 必填——本 script 是精确 topic 搜索，不提供空查询兜底")
        return 1

    try:
        max_results = max(10, min(int(args.get("max_results") or 20), 100))
    except (TypeError, ValueError):
        print_error("max_results 必须是整数")
        return 1

    if not X_BEARER_TOKEN:
        print_error("未配置 X_BEARER_TOKEN 环境变量")
        return 1

    try:
        return asyncio.run(
            _run(
                query=query,
                since_time=(args.get("since_time") or "").strip(),
                max_results=max_results,
            )
        )
    except XApiError as e:
        print_error(f"X API 调用失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
