"""
기술 요약 (RAG)

입력: state["current_candidate"]
출력: evaluation_update(cid, technology=AnalysisResult)

- rag.retriever.get_company_pages(cid) 로 기업 기술 페이지 확보
- rag.retriever.search(..., page_types=["tech_topic"]) 로 기술 공통 주제 보강
- 점수는 매기지 않는다. source_id 가 없는 성능 수치는 쓰지 않는다. (PDF 138쪽)
"""

from core.state import GraphState, evaluation_update, make_analysis


def run(state: GraphState) -> dict:
    cid = state["current_candidate"]
    # TODO: 구현
    return evaluation_update(cid, technology=make_analysis(summary="TODO: 기술 요약"))
