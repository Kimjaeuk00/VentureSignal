"""
스타트업 탐색 (RAG)

입력: state["query"]
출력: {"candidates": list[Candidate]}  — 최대 TOP_K_CANDIDATES 개, retrieval_score 내림차순

흐름
1. query_parser.parse_query   질의에서 지역·분야·후기 성장·상장 제외 조건을 뽑는다 (LLM, 실패 시 규칙 폴백)
2. retrieval.search_companies bge-m3 dense+sparse 하이브리드 검색 (region 은 payload 필터)
3. ranking.rerank             하드 필터(제외 category·상장/인수·late_stage) + category·태그·용어 점수
4. company_id 중복을 제거하고 상위 min(3, n) 개를 Candidate 로 변환한다.

- 조건에 맞는 후보가 3곳 미만이면 채우지 않는다. 0개도 허용하며, 그러면 그래프가 바로 보고서로 간다.
- 상장·인수 회사는 질의가 비상장·독립을 요구할 때만 제외한다 (비교·인접 회사가 정답인 질의가 있다).
- candidates 외 State 키는 반환하지 않는다. 점수·투자 판단은 하지 않는다.
"""

import logging

from langsmith import traceable

from core.config import TOP_K_CANDIDATES
from core.state import Candidate, GraphState

from .catalog import get_catalog
from .query_parser import parse_query
from .ranking import rerank
from .retrieval import search_companies

logger = logging.getLogger(__name__)


def _to_candidate(scored: dict, catalog: dict) -> Candidate:
    profile = catalog[scored["company_id"]]
    return {
        "company_id": profile["company_id"],
        "company_name": profile["company_name"],
        "description": profile["description"],  # PDF "기업·사업 근거" 문단 (출처 번호 포함)
        "domain": profile["category"],
        "retrieval_score": scored["retrieval_score"],
    }


@traceable(name="startup_search", run_type="chain", tags=["startup_search"])
def run(state: GraphState) -> dict:
    query = (state.get("query") or "").strip()
    if not query:
        return {"candidates": []}

    try:
        catalog = get_catalog()
        conditions = parse_query(query)
        hits = search_companies(conditions)
        ranked = rerank(hits, conditions, catalog, query=query)
    except Exception:
        # 인덱스 미생성(python -m rag.ingest 필요)·Qdrant 오류 등. 그래프가 죽지 않게 후보 없음으로 넘기되 원인은 남긴다.
        logger.exception("스타트업 탐색 실패: 후보 없이 진행합니다 (query=%r)", query)
        return {"candidates": []}

    candidates: list[Candidate] = []
    seen: set[str] = set()
    for scored in ranked:  # company_id 기준 중복 제거 (검색은 회사 단위라 보통 겹치지 않지만 계약으로 보장한다)
        if scored["company_id"] in seen:
            continue
        seen.add(scored["company_id"])
        candidates.append(_to_candidate(scored, catalog))
        if len(candidates) == TOP_K_CANDIDATES:
            break
    return {"candidates": candidates}
