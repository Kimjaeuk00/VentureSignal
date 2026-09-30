import json
from functools import lru_cache

from core.config import SOURCES_PATH


@lru_cache
def load_sources() -> dict[str, dict]:
    """S001 → {publisher, title, url}. python -m rag.ingest 실행 후 사용 가능."""
    return json.loads(SOURCES_PATH.read_text(encoding="utf-8"))


def format_source(source_id: str) -> str:
    s = load_sources().get(source_id)
    if not s:
        return f"[{source_id}] (출처 목록에 없음)"
    return f"[{source_id}] {s['publisher']}. {s['title']}. {s['url']}"
