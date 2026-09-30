import os
import re
from urllib.parse import urlsplit


def profile_url(url: str | None) -> str | None:
    parsed = urlsplit(url or "")
    host = (parsed.hostname or "").lower()
    if parsed.scheme in {"http", "https"} and (host == "linkedin.com" or host.endswith(".linkedin.com")):
        match = re.fullmatch(r"/in/([^/]+)/?", parsed.path)
        if match:
            return "https://www.linkedin.com/in/" + match[1]
    return None


def web_search(query: str, max_results: int) -> list[dict]:
    from tools.web_search import web_search as shared_search

    return shared_search(query, max_results=max_results)


def read_pages(urls: list[str], timeout: int) -> dict:
    import core.config  # noqa: F401 - 프로젝트 .env 로딩
    from tavily import TavilyClient

    client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
    try:
        advanced = client.extract(urls=urls, extract_depth="advanced", format="markdown", timeout=timeout)
    except Exception:
        advanced = {"results": [], "failed_results": [{"url": url} for url in urls]}

    good = {row.get("url") for row in advanced.get("results", []) if row.get("raw_content", "").strip()}
    failed = {row.get("url") for row in advanced.get("failed_results", [])}
    retry = [url for url in urls if url not in good or url in failed]
    if retry:
        basic = client.extract(urls=retry, extract_depth="basic", format="markdown", timeout=timeout)
        by_url = {row.get("url"): row for row in advanced.get("results", [])}
        by_url.update({
            row.get("url"): row
            for row in basic.get("results", [])
            if row.get("raw_content", "").strip()
        })
        advanced["results"] = list(by_url.values())
        advanced["failed_results"] = basic.get("failed_results", [])
    return advanced
