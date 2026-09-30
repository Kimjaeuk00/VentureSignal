"""
창업자 및 실적 (웹서치)

입력: state["current_candidate"], state["candidates"]
출력: evaluation_update(cid, team=..., traction=..., deal_terms=...)  — AnalysisResult 3개

- 창업팀 역량 / 고객·매출 실적 / 투자 이력·런웨이를 피어 그룹 대비로 분석
- 미확인 재무 항목은 추정하지 말고 "미확인"으로 남긴다. 점수는 매기지 않는다.
"""

from core.state import GraphState, evaluation_update, make_analysis


def run(state: GraphState) -> dict:
    cid = state["current_candidate"]
    # TODO: 구현
    return evaluation_update(
        cid,
        team=make_analysis(summary="TODO: 창업자"),
        traction=make_analysis(summary="TODO: 실적"),
        deal_terms=make_analysis(summary="TODO: 투자조건"),
    )
