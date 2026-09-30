"""
시장성 평가 (RAG)

입력: state["current_candidate"]
출력: evaluation_update(cid, market=AnalysisResult)

- 기업 시장성 페이지(company_market) + 시장 공통 주제(market_topic) 검색
- 시장 성장만으로 기업 매출을 예측하지 않는다. 점수는 매기지 않는다.
"""

from core.state import GraphState, evaluation_update, make_analysis


def run(state: GraphState) -> dict:
    cid = state["current_candidate"]
    # TODO: 구현
    return evaluation_update(cid, market=make_analysis(summary="TODO: 시장성 평가"))
