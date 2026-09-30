"""
에이전트가 사용하는 검색 API.

page_type 값
- company_index  : 기업 색인 (4~6쪽)
- company_tech   : 기업별 사업·제품·기술 근거 페이지
- company_market : 기업별 시장성·한계·확인 과제 페이지
- tech_topic     : 기술 구조·성능 비교 15개 주제 (T01~T15)
- market_topic   : 수요·시장·상용화 15개 주제 (T16~T30)
- meta           : 표지·방법론·RAG 설계 페이지
"""

from typing import Optional, TypedDict

from qdrant_client import models

from core.config import QDRANT_COLLECTION
from rag.embeddings import embed
from rag.store import get_client


class RetrievedChunk(TypedDict):
    text: str
    score: float
    page: int
    page_type: str
    company_id: Optional[str]
    company_name: Optional[str]
    region: Optional[str]
    category: Optional[str]
    topic_id: Optional[str]
    source_ids: list[str]


def _build_filter(page_types: Optional[list[str]], company_id: Optional[str]) -> Optional[models.Filter]:
    must = []
    if page_types:
        must.append(models.FieldCondition(key="page_type", match=models.MatchAny(any=page_types)))
    if company_id:
        must.append(models.FieldCondition(key="company_id", match=models.MatchValue(value=company_id)))
    return models.Filter(must=must) if must else None


def _to_chunk(point, score: float = 0.0) -> RetrievedChunk:
    p = point.payload
    return {
        "text": p["text"],
        "score": score,
        "page": p["page"],
        "page_type": p["page_type"],
        "company_id": p.get("company_id"),
        "company_name": p.get("company_name"),
        "region": p.get("region"),
        "category": p.get("category"),
        "topic_id": p.get("topic_id"),
        "source_ids": p.get("source_ids", []),
    }


def search(
    query: str,
    top_k: int = 5,
    page_types: Optional[list[str]] = None,
    company_id: Optional[str] = None,
) -> list[RetrievedChunk]:
    """bge-m3 dense + sparse 하이브리드 검색 (RRF 결합)."""
    dense, sparse = embed([query])[0]
    flt = _build_filter(page_types, company_id)
    res = get_client().query_points(
        QDRANT_COLLECTION,
        prefetch=[
            models.Prefetch(query=dense, using="dense", limit=top_k * 4, filter=flt),
            models.Prefetch(query=sparse, using="sparse", limit=top_k * 4, filter=flt),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_k,
        with_payload=True,
    )
    return [_to_chunk(pt, pt.score) for pt in res.points]


def get_company_pages(company_id: str) -> list[RetrievedChunk]:
    """company_id 의 기업 페이지 2장(기술 / 시장성)을 검색 없이 그대로 가져온다."""
    points, _ = get_client().scroll(
        QDRANT_COLLECTION,
        scroll_filter=_build_filter(["company_tech", "company_market"], company_id),
        limit=10,
        with_payload=True,
    )
    return sorted((_to_chunk(pt) for pt in points), key=lambda c: c["page"])
