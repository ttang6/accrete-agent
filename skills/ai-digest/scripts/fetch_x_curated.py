"""fetch_x_curated.py — 拉 sources.yaml 的 x_users curated 列表的近期推文。

LLM 调用：
  skill_exec(skill="ai-digest", script="fetch_x_curated", args={
      "category": "AI Research",     # 可选：按 category 过滤 curated 列表
      "since_id": "...",              # 可选
      "max_results_per_user": 10,     # 可选
      "include_replies": false,       # 可选
      "include_retweets": false       # 可选
  })

**契约**：本 script 语义明确 = "读 curated 运营池"。调用者不需要知道具体 user_ids。
若 sources.yaml 的 x_users 为空则返回空列表。

依赖 env：X_BEARER_TOKEN
"""

import asyncio
import sys

from _common import ensure_utf8_stdout, load_x_users, print_error, read_args
from _x_api import (
    X_BEARER_TOKEN,
    UserCache,
    XApiClient,
    XApiError,
    fetch_users_updates,
    format_tweet,
)


async def _resolve_usernames(client: XApiClient, usernames: list[str]) -> dict[str, dict]:
    """username → {user_id, username, name}。优先走 SQLite cache；缺的走 API。"""
    cache = UserCache()
    cached = cache.lookup_by_usernames(usernames)
    missing = [u for u in usernames if u.lower() not in cached]

    if missing:
        fresh = await client.resolve_users(usernames=missing)
        cache.upsert(fresh)
        for u in fresh:
            uname = u.get("username", "")
            if uname:
                cached[uname.lower()] = {
                    "user_id": u["id"],
                    "username": uname,
                    "name": u.get("name", ""),
                }
    return cached


async def _run(
    category: str,
    since_id: str,
    max_results_per_user: int,
    include_replies: bool,
    include_retweets: bool,
) -> int:
    curated = load_x_users(category=category or None)
    if not curated:
        msg = f"# X curated 动态（分类 '{category}' 无配置）" if category else "# X curated 动态（sources.yaml 的 x_users 为空）"
        print(msg)
        return 0

    usernames = [u["username"].lstrip("@") for u in curated]
    client = XApiClient(bearer_token=X_BEARER_TOKEN)

    resolved = await _resolve_usernames(client, usernames)
    user_ids = [resolved[u.lower()]["user_id"] for u in usernames if u.lower() in resolved]
    missing_usernames = [u for u in usernames if u.lower() not in resolved]

    if not user_ids:
        print_error(f"所有 curated username 均无法 resolve（{len(usernames)} 个）")
        return 1

    tweets, errors = await fetch_users_updates(
        client,
        user_ids=user_ids,
        since_id=since_id or None,
        max_results_per_user=max_results_per_user,
        include_replies=include_replies,
        include_retweets=include_retweets,
    )

    header = f"# X curated 动态（{len(curated)} 账号"
    if category:
        header += f"，category={category}"
    header += f"，{len(tweets)} 条新推文）"
    print(header)
    print()
    for tw in tweets:
        print(format_tweet(tw))
        print()

    if missing_usernames or errors:
        print("## 问题")
        for u in missing_usernames:
            print(f"- 无法 resolve username: @{u}")
        for err in errors:
            print(f"- user_id={err['user_id']}: {err['error']}")
    return 0


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()

    category = (args.get("category") or "").strip()

    try:
        max_results_per_user = max(5, min(int(args.get("max_results_per_user") or 10), 100))
    except (TypeError, ValueError):
        print_error("max_results_per_user 必须是整数")
        return 1

    if not X_BEARER_TOKEN:
        print_error("未配置 X_BEARER_TOKEN 环境变量")
        return 1

    try:
        return asyncio.run(
            _run(
                category=category,
                since_id=(args.get("since_id") or "").strip(),
                max_results_per_user=max_results_per_user,
                include_replies=bool(args.get("include_replies", False)),
                include_retweets=bool(args.get("include_retweets", False)),
            )
        )
    except XApiError as e:
        print_error(f"X API 调用失败: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
