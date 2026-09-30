import os
from functools import lru_cache
from typing import TypedDict

from tavily import TavilyClient

import core.config  # noqa: F401  (.env 로딩)


class WebResult(TypedDict):
    title: str
    url: str
    content: str


@lru_cache
def _client() -> TavilyClient:
    return TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


def web_search(query: str, max_results: int = 5) -> list[WebResult]:
    """Tavily 웹검색. evidence 에는 결과의 url 을 함께 남길 것."""
    res = _client().search(query, max_results=max_results)
    return [
        {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in res.get("results", [])
    ]
