import hashlib
import logging
import re
from urllib.parse import urlsplit

from .analysis_helpers import profile_sections
from .web_research import profile_url

LOG = logging.getLogger(__name__)


def build_sources(rows, scope: str, limit: int) -> list[dict]:
    sources = []
    for row in rows:
        url = row.get("url", "")
        body = row.get("raw_content") or row.get("content") or ""
        if urlsplit(url).scheme not in {"http", "https"} or not isinstance(body, str) or not body.strip():
            continue
        body = body[:limit]
        source_id = "S-" + hashlib.sha256((scope + url + body).encode()).hexdigest()[:12]
        sources.append({
            "source_id": source_id,
            "url": url,
            "title": row.get("title") or url,
            "content": body,
            "scope": scope,
        })
    return sources


def collect_pages(urls, page_reader, timeout: int, snippet_chars: int):
    """페이지 전체를 청크로 보존하고 URL별 수집 상태와 메타데이터를 반환한다."""
    urls = list(dict.fromkeys(urls))
    if not urls:
        return [], {}, {}

    statuses = {url: "unavailable" for url in urls}
    try:
        response = page_reader(urls, timeout=timeout)
    except Exception:
        LOG.exception("페이지 본문 추출 실패")
        return [], statuses, {}

    sources, metadata = [], {}
    for row in response.get("results", []):
        returned_url = row.get("url", "")
        target = next((
            url for url in urls
            if url == returned_url
            or (profile_url(url) and profile_url(url) == profile_url(returned_url))
        ), None)
        body = row.get("raw_content") or ""
        if not target or not isinstance(body, str) or not body.strip():
            continue
        if len(body) < 300 and re.search(r"sign in|log in|access denied|로그인", body, re.I):
            statuses[target] = "access_limited"
            continue

        statuses[target] = "extractor_returned_text"
        title = row.get("title") or target
        chunk_count = (len(body) + snippet_chars - 1) // snippet_chars
        metadata[target] = {"characters": len(body), "chunks": chunk_count}
        if profile_url(target):
            metadata[target].update(source="web_extract", sections=profile_sections(body))

        for offset in range(0, len(body), snippet_chars):
            chunk = body[offset:offset + snippet_chars]
            source_id = "S-" + hashlib.sha256(
                ("page_extract" + target + str(offset) + chunk).encode()
            ).hexdigest()[:12]
            sources.append({
                "source_id": source_id,
                "url": target,
                "title": title,
                "content": chunk,
                "scope": "page_extract",
                "page_offset": offset,
                "page_chars": len(body),
            })
    return sources, statuses, metadata
