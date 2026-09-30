"""
기술 요약 (RAG + 웹 보강)

입력: state["current_candidate"]
출력: evaluation_update(cid, technology=AnalysisResult)

- rag.retriever.get_company_pages(cid) 로 기업 기술 페이지(company_tech) 확보
- rag.retriever.search(..., page_types=["tech_topic"]) 로 기술 공통 주제 보강
  (tech_topic 페이지는 company_id 가 없으므로 기업 필터를 걸지 않는다)
- LLM 1차: PDF 만으로 요약 → 정해진 검색어(WEB_QUERIES)로 웹서치 → LLM 2차: 미확인 항목 웹 보강
  웹 결과가 없거나 웹서치가 실패하면 1차 요약을 그대로 쓴다. (details["web_search"] 에 기록)
- 점수는 매기지 않는다. 출처(source_id 또는 웹 URL)가 없는 성능 수치는 쓰지 않는다. (PDF 138쪽)
- 기업 기술 페이지가 없거나 LLM 이 확인 가능한 출처를 대지 못하면
  details["status"] = "insufficient_evidence" 로 근거 부족을 반환한다.
"""

import re
from datetime import date

from prompts.tech_summary import HUMAN_PROMPT, SYSTEM_PROMPT, WEB_PROMPT
from agents.tech_summary.schemas import TechSummaryOutput
from core.llm import get_llm
from core.state import GraphState, evaluation_update, make_analysis
from rag.retriever import get_company_pages, search
from core.sources import SourceBook
from tools.web_search import web_search

TECH_TOPIC_TOP_K = 3

# 보고서 '5. 기술력' 표의 행 (details 키 → 표 항목명)
FINDING_LABELS = {
    "core_technology": "핵심 기술",
    "product_stage": "제품 완성도",
    "software_environment": "사용 환경",
    "validation_limits": "검증·한계",
}
EXCLUDED = "출처 확인 불가로 제외"

# 웹 보강용 정해진 검색어. {target} = "기업명 제품명". PDF 에서 반복적으로 미확인이던 항목을 겨냥한다.
WEB_QUERIES = [
    "{target} mass production customers",  # 제품 완성도·양산·고객
    "{target} SDK software support",  # 사용 환경
    "{target} benchmark performance power efficiency",  # 독립 실측·성능
]
WEB_MAX_RESULTS = 3

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
THOUSANDS_RE = re.compile(r"(?<=\d),(?=\d{3})")


def run(state: GraphState) -> dict:
    cid = state["current_candidate"]

    tech_pages = [p for p in get_company_pages(cid) if p["page_type"] == "company_tech"]
    if not tech_pages:
        return _insufficient(cid, "기업 기술 페이지를 찾지 못함")

    topic_pages = search(tech_pages[0]["text"], top_k=TECH_TOPIC_TOP_K, page_types=["tech_topic"])
    pages = tech_pages + topic_pages
    company_name = tech_pages[0]["company_name"]
    context = _format_context(pages)
    llm = get_llm().with_structured_output(TechSummaryOutput)

    # 1차: 조사자료(PDF)만으로 요약
    out: TechSummaryOutput = llm.invoke(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT.format(company_name=company_name, company_id=cid, context=context)),
        ]
    )

    # 웹 검색 결과는 LLM 에게 URL 대신 출처 번호(U1…)로 보여 준다. LLM 이 URL 을 베끼다 틀리는 사고를 없앤다.
    book = SourceBook(canonicalize=True, rag_ids={sid for p in pages for sid in p["source_ids"]})

    # 2차: 정해진 검색어로 웹서치해 미확인 항목 보강. 결과가 없거나 실패하면 1차 요약을 그대로 쓴다.
    web_results, web_log = _web_search(company_name, out.product, book)
    if web_results:
        out = llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                (
                    "human",
                    WEB_PROMPT.format(
                        company_name=company_name,
                        company_id=cid,
                        draft=out.model_dump_json(indent=2),
                        context=context,
                        web_context=_format_web(web_results),
                    ),
                ),
            ]
        )
        _restore_sources(out, book)  # 출처 번호(U1…) → 표준 URL. 없는 번호는 뺀다.

    # LLM 이 지어낸 출처를 막는다: 실제로 가져온 페이지의 source_id 와 웹 결과 URL 만 남긴다.
    allowed = {sid for p in pages for sid in p["source_ids"]} | book.urls()

    def keep(ids: list[str]) -> list[str]:
        return [sid for sid in dict.fromkeys(ids) if sid in allowed]

    # 코드가 버린 내용도 unverified 에 남겨, 무엇이 빠졌는지 다음 노드가 알 수 있게 한다.
    unverified = list(out.unverified)
    if book.unknown:
        unverified.append(f"제공되지 않은 출처 표기 {len(book.unknown)}건을 제거함 — 해당 근거는 확인 필요")

    # 웹 수치는 인용한 검색 결과 본문에 그 값이 실제로 있어야 인정한다.
    # LLM 은 검색 결과의 본문만 보므로, 본문에 없는 값은 그 출처에서 나온 것이 아니다.
    web_numbers = {r.url: _numbers(f"{r.title} {r.content}") for r in web_results}

    performance = []
    for item in out.performance:
        ids = keep(item.source_ids)
        checked = [sid for sid in ids if sid not in web_numbers or _numbers(item.value) <= web_numbers[sid]]
        if checked:
            value = item.value.removesuffix(item.unit).strip() if item.unit else item.value  # "100ns"+"ns" 중복 제거
            performance.append({**item.model_dump(), "value": value, "source_ids": checked})
        elif ids:
            unverified.append(f"성능 수치 '{item.metric}': 웹 본문에서 값 확인 불가로 제외")
        else:  # 출처 없는 수치는 버리고, 지어냈을 수 있는 값 대신 지표 이름만 기록한다.
            unverified.append(f"성능 수치 '{item.metric}': {EXCLUDED}")

    # 보고서 '5. 기술력' 표의 행. 출처가 없는 내용은 추측으로 보고 '미확인'으로 바꾼다.
    findings = {}
    for name, label in FINDING_LABELS.items():
        finding = getattr(out, name)
        ids = keep(finding.source_ids)
        content = finding.content if ids else "미확인"
        # 웹 URL 을 인용했는데 본문에 (웹) 표시가 없으면 붙인다. LLM 이 지시를 놓치는 경우가 있다.
        if any(sid.startswith("http") for sid in ids) and "(웹" not in content:
            content += " (웹 출처 포함)"
        findings[name] = {"content": content, "source_ids": ids}
        if not ids and finding.content != "미확인":
            unverified.append(f"{label}: {EXCLUDED}")

    cited = [sid for row in [*performance, *findings.values()] for sid in row["source_ids"]]
    evidence = keep(out.evidence + cited)
    if not evidence:
        return _insufficient(cid, "요약에 확인 가능한 출처가 없음")

    update = evaluation_update(
        cid,
        technology=make_analysis(
            summary=out.summary,
            strengths=out.strengths,
            risks=out.risks,
            evidence=evidence,
            details={
                "status": "ok",
                "product": _clean_product(out.product),
                **findings,
                "performance": performance,
                "unverified": unverified,
                "pages": [p["page"] for p in pages],
                "web_search": web_log,
            },
        ),
    )
    update["sources"] = book.registry(used=evidence)  # 웹 출처 제목 (보고서 REFERENCE 용)
    return update


def _web_search(company_name: str, product: str, book: SourceBook) -> tuple[list, dict]:
    """정해진 검색어로 웹서치. 웹은 보강용이라 실패해도 예외를 올리지 않고 이유를 기록한다."""
    product = _clean_product(product)
    target = company_name if product in ("", "미확인") else f"{company_name} {product}"
    queries = [q.format(target=target) for q in WEB_QUERIES]
    log = {"searched_at": date.today().isoformat(), "queries": queries}  # 보고서 기준일·REFERENCE 용
    try:
        found = [r for q in queries for r in web_search(q, max_results=WEB_MAX_RESULTS)]
    except Exception as e:  # 키 없음·네트워크 오류 등
        return [], {"status": "skipped", "reason": f"{type(e).__name__}: {e}", **log}
    results = book.add(found)  # 같은 페이지는 한 번만, 표준 URL 과 출처 번호(U1…)를 붙인다
    return results, {"status": "ok" if results else "no_results", **log, "urls": [r.url for r in results]}


def _clean_product(product: str) -> str:
    """LLM 이 제품명 뒤에 붙인 설명·출처 표기 "(회사 공개, [S010])" 등을 잘라낸다."""
    return re.split(r"[(\[（]", product)[0].strip()


def _numbers(text: str) -> set[str]:
    """문자열 속 숫자 집합. 천 단위 쉼표는 없앤다 ("1,000" → "1000")."""
    return set(NUMBER_RE.findall(THOUSANDS_RE.sub("", text)))


def _format_web(results: list) -> str:
    return "\n\n".join(f"[웹 | {r.title} | 출처: {r.label}]\n{r.content}" for r in results)


def _restore_sources(out: TechSummaryOutput, book: SourceBook) -> None:
    """LLM 이 쓴 웹 출처 번호를 표준 URL 로 되돌린다(본문 인용 포함). 제공되지 않은 번호·URL 은 뺀다."""
    out.evidence = book.resolve(out.evidence)
    out.summary = book.restore(out.summary)
    out.strengths = [book.restore(t) for t in out.strengths]
    out.risks = [book.restore(t) for t in out.risks]
    out.unverified = [book.restore(t) for t in out.unverified]
    for name in FINDING_LABELS:
        finding = getattr(out, name)
        finding.content = book.restore(finding.content)
        finding.source_ids = book.resolve(finding.source_ids)
    for item in out.performance:
        item.condition = book.restore(item.condition)
        item.source_ids = book.resolve(item.source_ids)


def _format_context(pages: list[dict]) -> str:
    blocks = []
    for p in pages:
        label = f"{p['page_type']} {p['topic_id']}" if p["topic_id"] else p["page_type"]
        sources = ", ".join(p["source_ids"]) or "없음"
        blocks.append(f"[{p['page']}쪽 | {label} | 출처: {sources}]\n{p['text']}")
    return "\n\n".join(blocks)


def _insufficient(cid: str, reason: str) -> dict:
    return evaluation_update(
        cid,
        technology=make_analysis(
            summary=f"근거 부족: {reason}. 추측하지 않고 기술 요약을 비워 둠.",
            risks=[f"기술 근거 부족 ({reason}) — 추가 조사 필요"],
            details={"status": "insufficient_evidence", "reason": reason, "unverified": [f"기술 요약 전체: {reason}"]},
        ),
    )
