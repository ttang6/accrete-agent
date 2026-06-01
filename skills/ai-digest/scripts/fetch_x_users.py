"""fetch_x_users.py — 监控指定 X 账号的近期推文。

LLM 调用：
  skill_exec(skill="ai-digest", script="fetch_x_users", args={
      "user_ids": ["1619707188"],           # 必填：X 数字 user_id 列表
      "since_id": "...",                      # 可选：增量轮询边界
      "max_results_per_user": 10,             # 可选，5-100
      "include_replies": false,               # 可选
      "include_retweets": false               # 可选
  })

**契约**：user_ids 必填——"监控特定账号"是本 script 的核心语义，不兜底 curated list。
想读 curated 列表请用 fetch_x_curated。

依赖 env：X_BEARER_TOKEN
"""

import asyncio
import sys

from _common import ensure_utf8_stdout, print_error, read_args
from _x_api import (
    X_BEARER_TOKEN,
    XApiClient,
    XApiError,
    fetch_users_updates,
    format_tweet,
)


async def _run(
    user_ids: list[str],
    since_id: str,
    max_results_per_user: int,
    include_replies: bool,
    include_retweets: bool,
) -> int:
    client = XApiClient(bearer_token=X_BEARER_TOKEN)
    tweets, errors = await fetch_users_updates(
        client,
        user_ids=user_ids,
        since_id=since_id or None,
        max_results_per_user=max_results_per_user,
        include_replies=include_replies,
        include_retweets=include_retweets,
    )

    if not tweets and not errors:
        print(f"# X 用户动态（监控 {len(user_ids)} 账号，无新推文）")
        return 0

    print(f"# X 用户动态（监控 {len(user_ids)} 账号，{len(tweets)} 条新推文）")
    print()
    for tw in tweets:
        print(format_tweet(tw))
        print()
    if errors:
        print("## 失败账号")
        for err in errors:
            print(f"- user_id={err['user_id']}: {err['error']}")
    return 0


def main() -> int:
    ensure_utf8_stdout()
    args = read_args()

    user_ids = args.get("user_ids") or []
    if not isinstance(user_ids, list) or not user_ids:
        print_error(
            "user_ids 必填且非空——本 script 是'监控特定账号'，不兜底 curated。"
            "读 curated 列表请用 fetch_x_curated。"
        )
        return 1
    user_ids = [str(uid).strip() for uid in user_ids if str(uid).strip()]
    if not user_ids:
        print_error("user_ids 元素全为空字符串")
        return 1

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
                user_ids=user_ids,
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
