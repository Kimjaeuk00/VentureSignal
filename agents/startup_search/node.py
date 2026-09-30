"""
스타트업 탐색 (RAG)

입력: state["query"]
출력: {"candidates": list[Candidate]}  — 최대 TOP_K_CANDIDATES 개

- 질의에서 도메인·투자 단계 등 조건을 추출하고 rag.retriever.search 로 후보 검색
  (page_types=["company_index", "company_tech"])
- company_id 기준 중복 제거. 조건에 맞는 후보가 3곳 미만이면 채우지 않는다.
- 상장·인수 기업은 비교 기업이지 투자 후보가 아님 (PDF 132쪽 참고)
"""

from core.config import TOP_K_CANDIDATES
from core.state import Candidate, GraphState


def run(state: GraphState) -> dict:
    # TODO: 구현 — 아래는 그래프 연결 확인용 스텁
    candidates: list[Candidate] = [
        {
            "company_id": f"STUB{i}",
            "company_name": f"Stub Company {i}",
            "description": "stub",
            "domain": "stub",
            "retrieval_score": 0.0,
        }
        for i in range(1, TOP_K_CANDIDATES + 1)
    ]
    return {"candidates": candidates}
