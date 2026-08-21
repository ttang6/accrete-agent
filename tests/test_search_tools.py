"""网络搜索工具的契约测试：用假 Tavily 客户端验证参数校验与结果格式化。"""

from typing import Any

import pytest

from infra.core.types import ToolResult
from infra.runtime.tools import WebFetchTool, WebSearchTool
from infra.runtime.tools import search as search_module


class FakeClient:
    """模拟 Tavily 客户端的 search / extract 响应。"""

    def __init__(
        self,
        search_response: dict[str, Any] | None = None,
        extract_response: dict[str, Any] | None = None,
        search_error: Exception | None = None,
        extract_error: Exception | None = None,
    ) -> None:
        self.search_response = search_response
        self.extract_response = extract_response
        self.search_error = search_error
        self.extract_error = extract_error

    def search(self, **kwargs: Any) -> dict[str, Any]:
        if self.search_error is not None:
            raise self.search_error
        return self.search_response or {"results": []}

    def extract(self, urls: list[str]) -> dict[str, Any]:
        if self.extract_error is not None:
            raise self.extract_error
        return self.extract_response or {"results": [], "failed_results": []}


@pytest.fixture
def fake_client(monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    client = FakeClient()
    monkeypatch.setattr(search_module, "_CLIENT", client)
    return client


def test_web_search_formats_results(fake_client: FakeClient):
    fake_client.search_response = {
        "results": [
            {"title": "Python 文档", "url": "https://docs.python.org", "content": "官方文档"},
            {"title": "教程", "url": "https://example.com", "content": "入门教程"},
        ]
    }

    result = WebSearchTool().execute({"query": "python"})

    assert not result.is_error
    assert "共 2 条结果" in result.content
    assert "Python 文档" in result.content
    assert "https://docs.python.org" in result.content
    assert "web_fetch" in result.content


def test_web_search_empty_results(fake_client: FakeClient):
    result = WebSearchTool().execute({"query": "不存在的关键词"})

    assert not result.is_error
    assert "没有找到相关结果" in result.content


def test_web_search_rejects_bad_arguments():
    result = WebSearchTool().execute({"query": "  "})

    assert result.is_error
    assert result.error_type == "schema_error"


def test_web_search_api_error_becomes_exec_error(fake_client: FakeClient):
    fake_client.search_error = RuntimeError("网络超时")

    result = WebSearchTool().execute({"query": "python"})

    assert result.is_error
    assert result.error_type == "exec_error"
    assert "网络超时" in result.content


def test_web_search_without_api_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(search_module, "_CLIENT", None)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    result = WebSearchTool().execute({"query": "python"})

    assert result.is_error
    assert result.error_type == "exec_error"
    assert "TAVILY_API_KEY" in result.content


def test_web_fetch_formats_results_and_failures(fake_client: FakeClient):
    fake_client.extract_response = {
        "results": [
            {"title": "长文", "url": "https://a.com", "raw_content": "x" * 1000, "content": ""},
            {"title": "短文", "url": "https://b.com", "raw_content": "hello"},
        ],
        "failed_results": [{"url": "https://c.com", "error": "404"}],
    }

    result = WebFetchTool().execute({"urls": ["https://a.com", "https://b.com", "https://c.com"], "max_content_length": 500})

    assert not result.is_error
    assert "# 长文" in result.content
    assert "内容过长已截断" in result.content
    assert "# 抓取失败" in result.content
    assert "404" in result.content


def test_web_fetch_rejects_bad_arguments():
    result = WebFetchTool().execute({"urls": []})

    assert result.is_error
    assert result.error_type == "schema_error"


def test_web_tools_declare_network_permission_group():
    assert {WebSearchTool().permission_group, WebFetchTool().permission_group} == {"network"}
