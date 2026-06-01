"""_x_api.py — ai-digest skill 内部共享：X API v2 轻客户端 + fingerprint 抽取。

精简自项目根的 x_api_tools_fixed.py。对 nanoagent 的改动：
  - 去 OPENAI_TOOLS（schema 走 .schema.json 侧文件）
  - 去 Tweet dataclass（用 dict 减一层）
  - SQLite 默认路径固定到项目根 `data/cache/x_user_cache.sqlite`（对齐 fetch_github cache 模式）
  - 加 `extract_fingerprints(text)`：从推文抽 arxiv_id / owner/repo / 通用 URL，用于 dup_check 跨 source 去重
  - 加 `format_tweet(tweet)`：返回 stdout-friendly markdown（对齐 fetch_hf/rss/github 风格）

下划线前缀提示本模块是 skill 私有，describe_script / load_skill 不会暴露给 LLM。
"""

import asyncio
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

import httpx


_SKILL_ROOT = Path(__file__).resolve().parent.parent
_PROJECT_ROOT = _SKILL_ROOT.parent.parent
_DEFAULT_DB = _PROJECT_ROOT / "data" / "cache" / "x_user_cache.sqlite"

X_API_BASE = os.getenv("X_API_BASE", "https://api.x.com/2")
X_BEARER_TOKEN = os.getenv("X_BEARER_TOKEN", "")
SQLITE_PATH = Path(os.getenv("X_TOOL_DB", str(_DEFAULT_DB)))


# ============================================================
# SQLite cache
# ============================================================

@contextmanager
def _get_db(path: Path = SQLITE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_cache (
                user_id TEXT PRIMARY KEY,
                username TEXT,
                name TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


class UserCache:
    """按 username / user_id 双向缓存 X 用户元信息，避免 resolve 重复调用。"""

    def upsert(self, users: Iterable[dict]) -> None:
        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        with _get_db() as conn:
            for u in users:
                conn.execute(
                    """
                    INSERT INTO user_cache (user_id, username, name, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id) DO UPDATE SET
                        username=excluded.username,
                        name=excluded.name,
                        updated_at=excluded.updated_at
                    """,
                    (u["id"], u.get("username"), u.get("name"), now),
                )

    def lookup_by_usernames(self, usernames: list[str]) -> dict[str, dict]:
        """返回 {username_lower: {user_id, username, name}}。"""
        if not usernames:
            return {}
        placeholders = ",".join("?" for _ in usernames)
        with _get_db() as conn:
            rows = conn.execute(
                f"SELECT user_id, username, name FROM user_cache "
                f"WHERE LOWER(username) IN ({placeholders})",
                tuple(u.lower() for u in usernames),
            ).fetchall()
        return {row["username"].lower(): dict(row) for row in rows if row["username"]}

    def lookup_by_ids(self, user_ids: list[str]) -> dict[str, dict]:
        if not user_ids:
            return {}
        placeholders = ",".join("?" for _ in user_ids)
        with _get_db() as conn:
            rows = conn.execute(
                f"SELECT user_id, username, name FROM user_cache "
                f"WHERE user_id IN ({placeholders})",
                tuple(user_ids),
            ).fetchall()
        return {row["user_id"]: dict(row) for row in rows}


# ============================================================
# X API client
# ============================================================

class XApiError(RuntimeError):
    pass


class XApiClient:
    def __init__(self, bearer_token: str, base_url: str = X_API_BASE, timeout: float = 20.0):
        if not bearer_token:
            raise ValueError("X_BEARER_TOKEN is required (env or explicit)")
        self._token = bearer_token
        self._base = base_url.rstrip("/")
        self._timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._token}",
            "Content-Type": "application/json",
        }

    async def _get(self, path: str, params: dict) -> dict:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.get(f"{self._base}{path}", headers=self._headers(), params=params)
        if resp.status_code >= 400:
            raise XApiError(f"X API {resp.status_code}: {resp.text[:300]}")
        return resp.json()

    async def resolve_users(
        self,
        user_ids: Optional[list[str]] = None,
        usernames: Optional[list[str]] = None,
    ) -> list[dict]:
        result: list[dict] = []
        fields = "username,name,description,public_metrics,verified"

        if user_ids:
            payload = await self._get("/users", {"ids": ",".join(user_ids), "user.fields": fields})
            result.extend(payload.get("data", []) or [])
        if usernames:
            payload = await self._get(
                "/users/by",
                {"usernames": ",".join(usernames), "user.fields": fields},
            )
            result.extend(payload.get("data", []) or [])
        return result

    async def get_user_tweets(
        self,
        user_id: str,
        since_id: Optional[str] = None,
        max_results: int = 10,
        include_replies: bool = False,
        include_retweets: bool = False,
    ) -> dict:
        params: dict[str, Any] = {
            "max_results": max(5, min(max_results, 100)),
            "tweet.fields": "created_at,public_metrics,conversation_id,referenced_tweets,lang",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        excludes = []
        if not include_replies:
            excludes.append("replies")
        if not include_retweets:
            excludes.append("retweets")
        if excludes:
            params["exclude"] = ",".join(excludes)
        if since_id:
            params["since_id"] = since_id
        return await self._get(f"/users/{user_id}/tweets", params)

    async def search_recent_tweets(
        self,
        query: str,
        max_results: int = 20,
        start_time: Optional[str] = None,
    ) -> dict:
        params: dict[str, Any] = {
            "query": query,
            "max_results": max(10, min(max_results, 100)),
            "tweet.fields": "created_at,public_metrics,conversation_id,referenced_tweets,lang",
            "expansions": "author_id",
            "user.fields": "username,name",
        }
        if start_time:
            params["start_time"] = start_time
        return await self._get("/tweets/search/recent", params)


# ============================================================
# Fingerprint 抽取（tweet text → dup_check 硬键）
# ============================================================

_ARXIV_RE = re.compile(r"\b(?:arxiv\.org/abs/)?(\d{4}\.\d{4,5})(?:v\d+)?\b", re.IGNORECASE)
_GITHUB_RE = re.compile(r"github\.com/([\w.-]+/[\w.-]+?)(?:[/?#\s]|$)", re.IGNORECASE)


def extract_fingerprints(text: str) -> list[str]:
    """从推文 text 抽取可能的跨 source fingerprint（arxiv_id / owner/repo）。

    返回去重后的列表（插入序保留），供 LLM 调 dup_check 时附加到 tweet 自身 URL 之外
    作为 linked fingerprints 使用。
    """
    if not text:
        return []
    seen: set[str] = set()
    out: list[str] = []
    for match in _ARXIV_RE.findall(text):
        if match not in seen:
            seen.add(match)
            out.append(match)
    for match in _GITHUB_RE.findall(text):
        # 排除 github.com/<user> 只到用户级的情况（必须是 owner/repo 两段）
        if "/" in match and not match.endswith("/"):
            key = match.rstrip(".,;)")
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


# ============================================================
# Tweet 解析 + markdown 格式化
# ============================================================

def parse_tweets(payload: dict) -> list[dict]:
    """payload → list of normalized tweet dict（含 author 元信息 + url + linked fingerprints）。"""
    raw_tweets = payload.get("data") or []
    includes = payload.get("includes") or {}
    users = {u["id"]: u for u in includes.get("users") or []}

    # 缓存 resolved authors（跨 script 复用）
    if users:
        UserCache().upsert(users.values())

    out: list[dict] = []
    for t in raw_tweets:
        author = users.get(t.get("author_id", ""), {})
        username = author.get("username") or ""
        text = " ".join((t.get("text") or "").split())
        tweet_url = f"https://x.com/{username}/status/{t['id']}" if username else ""
        out.append(
            {
                "id": t["id"],
                "text": text,
                "author_id": t.get("author_id"),
                "username": username,
                "name": author.get("name") or "",
                "created_at": t.get("created_at"),
                "public_metrics": t.get("public_metrics") or {},
                "lang": t.get("lang"),
                "url": tweet_url,
                "linked_fingerprints": extract_fingerprints(text),
            }
        )
    return out


def format_tweet(tweet: dict) -> str:
    """单条 tweet → markdown 行。"""
    handle = f"@{tweet['username']}" if tweet.get("username") else f"user:{tweet.get('author_id', '?')}"
    name = tweet.get("name") or ""
    byline = f"{handle}（{name}）" if name else handle
    created = (tweet.get("created_at") or "")[:19]
    metrics = tweet.get("public_metrics") or {}
    likes = metrics.get("like_count", 0)
    rts = metrics.get("retweet_count", 0)
    text = tweet.get("text", "")
    text_trunc = text[:500] + ("..." if len(text) > 500 else "")
    url = tweet.get("url") or ""
    linked = tweet.get("linked_fingerprints") or []

    lines = [
        f"- **{byline}** · {created} · likes {likes} / rts {rts}",
        f"  {text_trunc}",
    ]
    if url:
        lines.append(f"  链接: {url}")
    if linked:
        lines.append(f"  linked_fingerprints: {', '.join(linked)}")
    return "\n".join(lines)


# ============================================================
# 上层 API：供 fetch_x_*.py 调用
# ============================================================

async def fetch_users_updates(
    client: XApiClient,
    user_ids: list[str],
    since_id: Optional[str] = None,
    max_results_per_user: int = 10,
    include_replies: bool = False,
    include_retweets: bool = False,
) -> tuple[list[dict], list[dict]]:
    """并发拉多个 user 的 tweets。返回 (tweets, errors)。"""

    async def fetch_one(uid: str):
        try:
            payload = await client.get_user_tweets(
                user_id=uid,
                since_id=since_id,
                max_results=max_results_per_user,
                include_replies=include_replies,
                include_retweets=include_retweets,
            )
            return uid, parse_tweets(payload), None
        except Exception as e:
            return uid, [], str(e)

    results = await asyncio.gather(*(fetch_one(uid) for uid in user_ids))

    all_tweets: list[dict] = []
    errors: list[dict] = []
    seen_ids: set[str] = set()
    for uid, tweets, err in results:
        if err:
            errors.append({"user_id": uid, "error": err})
            continue
        for tw in tweets:
            if tw["id"] in seen_ids:
                continue
            seen_ids.add(tw["id"])
            all_tweets.append(tw)

    all_tweets.sort(key=lambda t: (t.get("created_at") or ""), reverse=True)
    return all_tweets, errors
