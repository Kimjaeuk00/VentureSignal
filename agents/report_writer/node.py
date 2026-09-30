"""
보고서 생성 (LLM)

입력: state 전체 (candidates, evaluations, selected_candidate)
출력: {"report": str}  — 마크다운. 다른 State 키는 반환하지 않는다. None 도 반환하지 않는다(main.py 가 문자열을 파일에 쓴다).

후보를 한 곳씩 평가하다 INVEST 가 나오면 즉시, 계속 HOLD 로 후보를 소진하면 이 노드가 실행된다.
- selected_candidate 가 있으면 → 그 후보 1곳의 투자 검토 보고서 (E절 8개 섹션)
- 없고 평가된 후보가 있으면   → 평가한 후보 전부의 보류 보고서 (보류 사유)
- 평가된 후보가 없으면        → 보고서 없이 안내 문자열 (정상 흐름은 아니다)

표는 State 값을 옮겨 채우고(collect → sections), LLM 은 문장만 쓴다(llm_writer). 값이 없으면 "미확인".
마크다운과 PDF 는 같은 Document 모델에서 만든다. PDF 는 outputs/ 에 저장되는 부수 효과이고(GraphState 에 경로 필드가 없다),
PDF 생성이 실패해도 마크다운 report 는 반환한다.
"""

import logging

from langsmith import traceable

from core.state import GraphState

from .collect import collect
from .llm_writer import llm_budget, write_hold_prose, write_invest_prose
from .references import SourceRegistry
from .render_md import render_md
from .render_pdf import write_pdf_fit
from .sections import build_hold_document, build_invest_document

logger = logging.getLogger(__name__)

NO_CANDIDATE_NOTICE = "평가된 후보가 없어 보고서를 작성하지 않았습니다."


@traceable(name="report_writer", run_type="chain", tags=["report_writer"])
def run(state: GraphState) -> dict:
    try:
        ctx = collect(state)
        if ctx.kind == "none":
            return {"report": NO_CANDIDATE_NOTICE}

        registry = SourceRegistry(titles=ctx.url_titles, accessed=ctx.as_of)
        with llm_budget():  # 문장 작성과 분량 요약을 합쳐 LLM 호출은 보고서 한 건에 최대 LLM_CALL_LIMIT(3)회
            if ctx.kind == "invest":
                document = build_invest_document(ctx, write_invest_prose(ctx), registry)
            else:
                document = build_hold_document(ctx, write_hold_prose(ctx), registry)
            try:
                # PDF 는 MAX_PAGES 쪽 이내: 넘으면 요약해 다시 만든다. 마크다운도 최종 PDF 와 같은 문서다.
                document, pdf_path, pages = write_pdf_fit(document)
                logger.info("PDF 저장: %s (%d쪽)", pdf_path, pages)
            except Exception:
                logger.exception("PDF 생성 실패: 마크다운 보고서만 반환합니다")
        return {"report": render_md(document)}
    except Exception as exc:
        # 마지막 노드에서 예외가 나면 그래프가 그때까지의 결과를 모두 잃는다. 원인은 로그에 남기고 문자열로 끝낸다.
        logger.exception("보고서 생성 실패")
        return {"report": f"보고서를 작성하지 못했습니다 ({type(exc).__name__}). 원인은 로그를 확인하세요."}
