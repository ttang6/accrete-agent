"""Minimal MCP client examples.

Default: connect to the YouTube Transcript MCP server over stdio.

Usage from project root:

    .venv/Scripts/python.exe src/nanoagent/mcp/client.py

Local arxiv stdio example:

    .venv/Scripts/python.exe src/nanoagent/mcp/client.py --arxiv
"""

import argparse
import asyncio
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import AnyUrl


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARXIV_STORAGE = PROJECT_ROOT / "data" / "arxiv_papers"


async def list_capabilities(session: ClientSession) -> None:
    await session.initialize()

    tools = await session.list_tools()
    print("Tools:")
    for tool in tools.tools:
        print(f"- {tool.name}: {tool.description or ''}")

    try:
        resources = await session.list_resources()
    except Exception as exc:
        print(f"\nResources: unavailable ({type(exc).__name__}: {exc})")
    else:
        print("\nResources:")
        for resource in resources.resources:
            print(f"- {resource.uri}: {resource.name or ''}")

    try:
        templates = await session.list_resource_templates()
    except Exception as exc:
        print(f"\nResource templates: unavailable ({type(exc).__name__}: {exc})")
    else:
        print("\nResource templates:")
        for template in templates.resourceTemplates:
            print(f"- {template.uriTemplate}: {template.name or ''}")


async def read_resource(session: ClientSession, uri: str, max_chars: int) -> None:
    result = await session.read_resource(AnyUrl(uri))
    print(f"\nResource: {uri}")
    for index, content in enumerate(result.contents, 1):
        mime = content.mimeType or "unknown"
        print(f"\n[{index}] {mime} {content.uri}")
        if hasattr(content, "text"):
            text = content.text
            if len(text) > max_chars:
                text = text[:max_chars] + f"\n...(truncated, {len(content.text)} chars total)"
            print(text)
        elif hasattr(content, "blob"):
            blob = content.blob
            shown = blob[:max_chars]
            suffix = f"\n...(truncated, {len(blob)} chars total)" if len(blob) > max_chars else ""
            print(shown + suffix)
        else:
            print(content)


async def inspect_session(
    session: ClientSession,
    *,
    read_resource_uri: str | None,
    max_resource_chars: int,
) -> None:
    await list_capabilities(session)
    if read_resource_uri:
        await read_resource(session, read_resource_uri, max_resource_chars)


async def run_youtube_transcript(read_resource_uri: str | None, max_resource_chars: int) -> None:
    params = StdioServerParameters(
        command="npx.cmd",
        args=["@kimtaeyoon83/mcp-server-youtube-transcript"],
        env=None,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await inspect_session(
                session,
                read_resource_uri=read_resource_uri,
                max_resource_chars=max_resource_chars,
            )


async def run_arxiv(read_resource_uri: str | None, max_resource_chars: int) -> None:
    ARXIV_STORAGE.mkdir(parents=True, exist_ok=True)
    params = StdioServerParameters(
        command="uv",
        args=[
            "tool",
            "run",
            "arxiv-mcp-server",
            "--storage-path",
            str(ARXIV_STORAGE),
        ],
        env=None,
    )
    async with stdio_client(params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await inspect_session(
                session,
                read_resource_uri=read_resource_uri,
                max_resource_chars=max_resource_chars,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal nanoagent MCP client")
    parser.add_argument(
        "--arxiv",
        action="store_true",
        help="Connect to local arxiv-mcp-server over stdio instead of YouTube transcript MCP.",
    )
    parser.add_argument(
        "--read-resource",
        metavar="URI",
        help="Read one MCP resource URI after listing capabilities.",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=4000,
        help="Maximum characters to print per resource content item.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.arxiv:
        asyncio.run(run_arxiv(args.read_resource, args.max_resource_chars))
    else:
        asyncio.run(run_youtube_transcript(args.read_resource, args.max_resource_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
