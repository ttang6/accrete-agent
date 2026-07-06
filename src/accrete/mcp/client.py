"""Minimal MCP client（FastMCP）—— 主动发现模式。

正常 MCP client 的形态：从配置（项目根 `mcp_servers.json`，缺失则用内置示例）读 server，
用 `fastmcp.Client` 连接、自动发现它暴露的 tools / resources。fastmcp 的 Client 直接吃
标准 MCPConfig（`{"mcpServers": {...}}`），传输 / 生命周期 / 多 server 都它管——比官方
SDK 的 stdio_client + ClientSession 手动嵌套干净不少。

用法（项目根）：

    .venv/Scripts/python.exe src/accrete/mcp/client.py                  # 发现配置/默认里所有 server
    .venv/Scripts/python.exe src/accrete/mcp/client.py --config my.json
    .venv/Scripts/python.exe src/accrete/mcp/client.py --read-resource <uri>
"""

import argparse
import asyncio
import json
from pathlib import Path

from fastmcp import Client


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ARXIV_STORAGE = PROJECT_ROOT / "data" / "arxiv_papers"
DEFAULT_CONFIG = PROJECT_ROOT / "mcp_servers.json"


def _builtin_config() -> dict:
    """无配置文件时的回退示例（标准 MCPConfig 形态）。"""
    ARXIV_STORAGE.mkdir(parents=True, exist_ok=True)
    return {
        "mcpServers": {
            "arxiv": {
                "command": "uv",
                "args": ["tool", "run", "arxiv-mcp-server", "--storage-path", str(ARXIV_STORAGE)],
            },
        }
    }


def load_config(config_path: Path) -> dict:
    """有配置文件就读它（标准 MCPConfig），没有就回退到内置示例。"""
    if config_path.is_file():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return _builtin_config()


async def discover(config: dict, read_resource_uri: str | None, max_chars: int) -> None:
    async with Client(config) as client:
        tools = await client.list_tools()
        print(f"Tools ({len(tools)}):")
        for tool in tools:
            print(f"- {tool.name}: {tool.description or ''}")

        try:
            resources = await client.list_resources()
        except Exception as exc:
            print(f"\nResources: unavailable ({type(exc).__name__}: {exc})")
        else:
            print(f"\nResources ({len(resources)}):")
            for resource in resources:
                print(f"- {resource.uri}: {resource.name or ''}")

        if read_resource_uri:
            print(f"\nResource: {read_resource_uri}")
            for item in await client.read_resource(read_resource_uri):
                text = getattr(item, "text", None)
                if text is None:
                    print(item)
                    continue
                if len(text) > max_chars:
                    text = text[:max_chars] + f"\n...(truncated, {len(item.text)} chars total)"
                print(text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="accrete MCP client（FastMCP，主动发现）")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"MCPConfig 文件（默认 {DEFAULT_CONFIG.name}，缺失则用内置示例）。",
    )
    parser.add_argument(
        "--read-resource",
        metavar="URI",
        help="列完能力后读一个 resource URI。",
    )
    parser.add_argument(
        "--max-resource-chars",
        type=int,
        default=4000,
        help="每个 resource 内容最多打印多少字符。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    asyncio.run(discover(config, args.read_resource, args.max_resource_chars))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
