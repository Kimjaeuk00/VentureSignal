"""
경쟁사 비교 (웹서치)

입력: state["current_candidate"], state["candidates"]
출력: evaluation_update(cid, competition=AnalysisResult)

- tools.web_search.web_search 로 피어 그룹 탐색, 경쟁 구도·차별성 분석
- details 에 보고서 4절 "경쟁 제품 비교" 표를 채울 수 있는 구조화 데이터를 담는다.
- 점수는 매기지 않는다.
"""

from core.state import GraphState, evaluation_update, make_analysis


def run(state: GraphState) -> dict:
    cid = state["current_candidate"]
    # TODO: 구현
    return evaluation_update(cid, competition=make_analysis(summary="TODO: 경쟁사 비교"))
