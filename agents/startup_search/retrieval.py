"""
하이브리드 검색 — bge-m3 dense + sparse 를 각각 조회해 RRF 로 결합하고 회사 단위로 돌려준다.

rag.retriever.search 와 일부 겹치지만(같은 prefetch + RRF), 그 함수에는 region 필터가 없고
rag/ 는 수정할 수 없어서 startup_search 안에 필터를 포함해 새로 작성했다.
클라이언트·임베딩은 rag.store.get_client / rag.embeddings.embed 를 그대로 재사용한다.
(Qdrant 로컬 모드는 한 경로에 프로세스 하나만 접근할 수 있으므로 get_client 로만 얻는다.)

조회 대상은 회사당 1장인 company_tech 페이지이므로 결과는 회사 단위로 유일하다.
상장·인수·category 같은 조건 필터는 여기서 하지 않고 ranking 에서 처리한다.
"""

from typing import Optional, TypedDict

from langsmith import traceable
from qdrant_client import models

from core.config import QDRANT_COLLECTION
from rag.embeddings import embed
from rag.store import get_client

from .schemas import SearchConditions

# 회사가 50개뿐이므로 기본은 전부 가져와 ranking 이 필터·정렬하게 한다
DEFAULT_TOP_N = 50
PAGE_TYPE = "company_tech"


class Hit(TypedDict):
    company_id: str
    company_name: str
    score: float  # RRF 결합 점수 (검색 내에서만 의미 있음, 정규화는 ranking 에서)
    rank: int  # 1부터


def _build_filter(region: Optional[str]) -> models.Filter:
    must = [models.FieldCondition(key="page_type", match=models.MatchValue(value=PAGE_TYPE))]
    if region:
        must.append(models.FieldCondition(key="region", match=models.MatchValue(value=region)))
    return models.Filter(must=must)


@traceable(name="search_companies", run_type="retriever")
def search_companies(
    conditions: SearchConditions,
    query: Optional[str] = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[Hit]:
    """
    검색어(기본 conditions.semantic_query)로 회사를 찾는다. region 은 payload 필터로 건다.
    dense 와 sparse 는 같은 필터로 각각 prefetch 한 뒤 RRF 로 결합한다.
    """
    text = (query or conditions.semantic_query).strip()
    if not text:
        return []

    dense, sparse = embed([text])[0]
    flt = _build_filter(conditions.region)

    prefetch = [models.Prefetch(query=dense, using="dense", limit=top_n, filter=flt)]
    if sparse.indices:  # 사전에 없는 단어뿐이면 sparse 벡터가 비어 조회할 수 없다
        prefetch.append(models.Prefetch(query=sparse, using="sparse", limit=top_n, filter=flt))

    res = get_client().query_points(
        QDRANT_COLLECTION,
        prefetch=prefetch,
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=top_n,
        with_payload=["company_id", "company_name"],
    )
    return [
        {
            "company_id": pt.payload["company_id"],
            "company_name": pt.payload["company_name"],
            "score": pt.score,
            "rank": i,
        }
        for i, pt in enumerate(res.points, start=1)
    ]
