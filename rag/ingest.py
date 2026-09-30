"""
조사자료 PDF → Qdrant 인덱스 + 출처 목록(sources.json) 생성.

    python -m rag.ingest

PDF 는 페이지마다 첫 줄 헤더로 유형이 구분되고, 한 페이지가 400~800 토큰 수준이라
MVP 에서는 페이지 단위로 청크를 만든다. (PDF 137쪽 권장: 이후 의미 단위 분할로 조정)
"""

import json
import re
import uuid

import pymupdf
from qdrant_client import models

from core.config import (
    EMBEDDING_DIM,
    QDRANT_COLLECTION,
    RESEARCH_PDF,
    SOURCES_PATH,
)
from rag.embeddings import embed
from rag.store import get_client

FOOTER_RE = re.compile(r"^(AI CHIP RESEARCH\s*\|.*|\d+ / \d+)$")
SOURCE_ID_RE = re.compile(r"\bS\d{3}\b")

# 헤더(첫 줄) → page_type
PAGE_TYPES = {
    "01 • COMPANY / DISCOVERY & TECHNOLOGY": "company_tech",
    "01 • COMPANY / MARKET & VALIDATION": "company_market",
    "02 • TECHNOLOGY": "tech_topic",
    "03 • MARKET": "market_topic",
    "00 • KOREA INDEX": "company_index",
    "00 • GLOBAL INDEX": "company_index",
    "05 • SOURCE REGISTER": "source_register",
}


def _clean_lines(text: str) -> list[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip() and not FOOTER_RE.match(ln.strip())]


def parse_pages(pdf_path=RESEARCH_PDF) -> list[dict]:
    doc = pymupdf.open(pdf_path)
    pages = []
    regions: dict[str, str] = {}

    for i, page in enumerate(doc):
        lines = _clean_lines(page.get_text())
        header = lines[0] if lines else ""
        page_type = PAGE_TYPES.get(header, "meta")
        meta = {
            "page": i + 1,
            "page_type": page_type,
            "title": lines[1] if len(lines) > 1 else "",
            "company_id": None,
            "company_name": None,
            "region": None,
            "category": None,
            "topic_id": None,
        }

        if page_type in ("company_tech", "company_market"):
            # 2번째 줄: 기업명 (market 페이지는 "기업명 | 시장성과 확인 과제")
            # 3번째 줄: "C03 | 국내 | AI 반도체 IP"
            meta["company_name"] = lines[1].split("|")[0].strip()
            parts = [p.strip() for p in lines[2].split("|")]
            meta["company_id"] = parts[0]
            if page_type == "company_tech":
                meta["region"], meta["category"] = parts[1], parts[2]
                regions[parts[0]] = (parts[1], parts[2])
            elif parts[0] in regions:
                meta["region"], meta["category"] = regions[parts[0]]
        elif page_type in ("tech_topic", "market_topic"):
            meta["topic_id"] = lines[2].split("|")[0].strip()

        body = "\n".join(lines)
        meta["source_ids"] = sorted(set(SOURCE_ID_RE.findall(body)))
        meta["text"] = body
        pages.append(meta)

    return pages


def parse_sources(pages: list[dict]) -> dict[str, dict]:
    """출처 목록 페이지(141~150쪽)에서 S001 → {publisher, title, url} 추출."""
    text = "\n".join(p["text"] for p in pages if p["page_type"] == "source_register")
    sources = {}
    for block in re.split(r"\n(?=S\d{3} \| )", text):
        m = re.match(r"(S\d{3}) \| (.+)", block)
        if not m:
            continue
        rest = block.splitlines()[1:]
        url_start = next((k for k, ln in enumerate(rest) if ln.startswith("http")), None)
        title = " ".join(rest[:url_start] if url_start is not None else rest)
        url = "".join(rest[url_start:]) if url_start is not None else ""
        sources[m.group(1)] = {"publisher": m.group(2).strip(), "title": title.strip(), "url": url}
    return sources


def build_index(pages: list[dict]) -> int:
    client = get_client()
    if client.collection_exists(QDRANT_COLLECTION):
        client.delete_collection(QDRANT_COLLECTION)
    client.create_collection(
        QDRANT_COLLECTION,
        vectors_config={"dense": models.VectorParams(size=EMBEDDING_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )

    chunks = [p for p in pages if p["page_type"] not in ("source_register",)]
    vectors = embed([c["text"] for c in chunks])
    client.upsert(
        QDRANT_COLLECTION,
        points=[
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"page-{c['page']}")),
                vector={"dense": dense, "sparse": sparse},
                payload=c,
            )
            for c, (dense, sparse) in zip(chunks, vectors)
        ],
    )
    return len(chunks)


def main():
    pages = parse_pages()
    sources = parse_sources(pages)
    SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    SOURCES_PATH.write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"sources: {len(sources)} → {SOURCES_PATH}")

    n = build_index(pages)
    print(f"indexed pages: {n} → collection '{QDRANT_COLLECTION}'")


if __name__ == "__main__":
    main()
