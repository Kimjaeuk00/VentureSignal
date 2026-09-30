"""
시장성 평가 (RAG + 필요 시 웹 검색)

입력:
- state["current_candidate"]
- state["candidates"]
- state["query"]

출력:
- evaluation_update(cid, market=AnalysisResult)

기업 시장성 자료와 시장 공통 자료로 먼저 분석한다.
정보가 부족하면 시장 규모·성장 또는 고객 수요·도입을 검색한다.
점수와 투자 여부는 판단하지 않는다.
"""

import json
import os
import re

from core.llm import get_llm
from core.sources import SourceBook
from core.state import GraphState, evaluation_update, make_analysis
from rag.retriever import get_company_pages, search
from tools.web_search import web_search

from .prompts import MARKET_SYSTEM_PROMPT, MARKET_USER_PROMPT
from .schemas import MarketAnalysis, MarketSearchQuery


# 기업 1곳당 검색 목적별 최대 1회, 전체 최대 2회
MAX_WEB_QUERIES = 2
WEB_RESULTS_PER_QUERY = 3


def _to_json(value) -> str:
    """자료를 프롬프트에 넣을 문자열로 변환한다."""
    return json.dumps(value, ensure_ascii=False, indent=2)


def _analyze(
    model,
    inputs: dict,
    stage: str,
    web_context: dict,
) -> MarketAnalysis:
    """자료를 전달하고 MarketAnalysis 형식으로 분석 결과를 받는다."""
    user_prompt = MARKET_USER_PROMPT.format(
        **inputs,
        analysis_stage=stage,
        web_context=_to_json(web_context),
    )

    result = model.invoke([
        ("system", MARKET_SYSTEM_PROMPT),
        ("human", user_prompt),
    ])

    if not isinstance(result, MarketAnalysis):
        raise TypeError(
            "시장성 분석 결과가 MarketAnalysis 형식이 아닙니다."
        )

    return result


_CITATION_PATTERN = re.compile(
    r"""https?://[^\s<>"'\[\]{}]+|\bS\d{3}\b"""
)

# 허용되지 않은 출처를 LLM이 인용했을 때 다시 요청하는 최대 횟수
MAX_EVIDENCE_RETRIES = 1


def _normalize_source(source: str, allowed_sources: set[str]) -> str:
    source = source.strip()

    # "[S006]" → "S006"
    match = re.fullmatch(r"\[(S\d{3})\]", source)
    if match:
        source = match.group(1)

    # 실제 제공된 URL은 그대로 유지한다.
    if source in allowed_sources:
        return source

    # URL 뒤에 문장부호가 붙은 경우:
    # 문장부호를 제거한 값이 실제 제공된 출처와 일치할 때만 보정한다.
    if source.startswith(("https://", "http://")):
        candidate = source

        while candidate and candidate[-1] in ".,;:!?)]。，；":
            candidate = candidate[:-1]

            if candidate in allowed_sources:
                return candidate

    return source


def _strip_unknown_sources(
    result: MarketAnalysis,
    allowed_sources: set[str],
) -> int:
    """허용되지 않은 출처를 본문과 evidence에서 제거한다. 제거한 수를 반환한다."""
    removed = 0

    def clean_text(text: str) -> str:
        nonlocal removed

        def replace(match: re.Match) -> str:
            nonlocal removed
            token = match.group(0)
            if _normalize_source(token, allowed_sources) in allowed_sources:
                return token
            removed += 1
            return "(출처 미확인)"

        return _CITATION_PATTERN.sub(replace, text)

    details = result.details
    result.summary = clean_text(result.summary)
    result.strengths = [clean_text(t) for t in result.strengths]
    result.risks = [clean_text(t) for t in result.risks]
    details.target_customers = [clean_text(t) for t in details.target_customers]
    details.market_size = clean_text(details.market_size)
    details.growth_drivers = [clean_text(t) for t in details.growth_drivers]
    details.business_model = clean_text(details.business_model)
    details.adoption_barriers = [clean_text(t) for t in details.adoption_barriers]
    details.missing_information = [
        clean_text(t) for t in details.missing_information
    ]

    kept = [
        _normalize_source(source, allowed_sources)
        for source in result.evidence
    ]
    kept = [source for source in kept if source in allowed_sources]
    removed += len(result.evidence) - len(kept)
    result.evidence = kept
    return removed


def _restore_sources(result: MarketAnalysis, book: SourceBook) -> None:
    """LLM 이 쓴 웹 출처 번호(U1…)를 표준 URL 로 되돌린다. 제공되지 않은 번호·URL 은 근거와 본문에서 뺀다."""
    details = result.details
    result.evidence = book.resolve(result.evidence)
    result.summary = book.restore(result.summary)
    result.strengths = [book.restore(t) for t in result.strengths]
    result.risks = [book.restore(t) for t in result.risks]
    details.target_customers = [book.restore(t) for t in details.target_customers]
    details.market_size = book.restore(details.market_size)
    details.growth_drivers = [book.restore(t) for t in details.growth_drivers]
    details.business_model = book.restore(details.business_model)
    details.adoption_barriers = [book.restore(t) for t in details.adoption_barriers]
    details.missing_information = [book.restore(t) for t in details.missing_information]


def _analyze_checked(
    model,
    inputs: dict,
    stage: str,
    web_context: dict,
    allowed_sources: set[str],
    book: SourceBook,
) -> MarketAnalysis:
    """분석하고 출처를 검증한다.

    웹 출처는 LLM 이 번호(U1…)로 쓰고 여기서 URL 로 되돌린다(LLM 이 URL 을 베끼다 틀리는 사고를 없앤다).
    허용되지 않은 출처가 남으면 오류를 알려 주고 다시 요청한다.
    그래도 남으면 그 출처만 제거하고 미확인 사항에 기록한다.
    한 후보의 출처 오류로 그래프 전체가 멈추지 않게 한다.
    """
    allowed_sources = allowed_sources | book.urls()
    context = web_context
    last_error: ValueError | None = None
    dropped: list[str] = []

    for _ in range(MAX_EVIDENCE_RETRIES + 1):
        before = len(book.unknown)
        result = _analyze(model, inputs, stage, context)
        _restore_sources(result, book)
        dropped = book.unknown[before:]  # 이번 응답에서 목록에 없어 뺀 출처 표기

        try:
            if dropped:
                raise ValueError(
                    "제공되지 않은 출처 표기가 있습니다: "
                    f"{', '.join(dict.fromkeys(dropped))}"
                )

            _validate_evidence(result, allowed_sources)
            return result
        except ValueError as exc:
            last_error = exc
            context = {
                **web_context,
                "previous_error": (
                    f"{exc} 제공된 출처 ID(S006)와 웹 label(U1 …)만 인용하고 "
                    "evidence에도 그 값만 넣는다."
                ),
            }

    removed = _strip_unknown_sources(result, allowed_sources) + len(dropped)

    try:
        _validate_evidence(result, allowed_sources)
    except ValueError:
        # 남은 evidence가 본문 인용과 맞지 않으면 비운다.
        result.evidence = []

    result.details.missing_information.append(
        "분석 결과에 제공되지 않은 출처 "
        f"{removed}건이 있어 제거함. 해당 근거는 확인 필요. "
        f"({last_error})"
    )
    return result


def _validate_evidence(
    result: MarketAnalysis,
    allowed_sources: set[str],
) -> None:
    """출처 유효성을 검사하고 본문 인용과 evidence를 맞춘다."""

    def normalize(source: str) -> str:
        return _normalize_source(source, allowed_sources)

    # 1. LLM이 반환한 evidence 자체의 유효성 검사
    declared_sources = [
        normalize(source)
        for source in result.evidence
    ]

    unknown_declared = set(declared_sources) - allowed_sources

    if unknown_declared:
        raise ValueError(
            "evidence에 제공되지 않은 출처가 있습니다: "
            f"{sorted(unknown_declared)}"
        )

    # 2. 분석 본문만 모은다.
    # 검색 계획이나 실행 로그는 인용 본문에 포함하지 않는다.
    details = result.details

    body_parts = [
        result.summary,
        *result.strengths,
        *result.risks,
        *details.target_customers,
        details.market_size,
        *details.growth_drivers,
        details.business_model,
        *details.adoption_barriers,
        *details.missing_information,
    ]

    body = "\n".join(body_parts)

    # URL을 먼저 인식하여 URL 내부의 S006 같은 문자열을
    # 별도 RAG 출처로 중복 추출하지 않는다.
    cited_sources = [
        normalize(match.group(0))
        for match in _CITATION_PATTERN.finditer(body)
    ]

    # 등장 순서를 유지하며 중복 제거
    cited_sources = list(dict.fromkeys(cited_sources))

    # 3. 본문에만 숨어 있는 잘못된 출처도 검사
    unknown_cited = set(cited_sources) - allowed_sources

    if unknown_cited:
        raise ValueError(
            "본문에 제공되지 않은 출처가 있습니다: "
            f"{sorted(unknown_cited)}"
        )

    # 출처 목록만 있고 본문 인용이 전혀 없다면 확인 필요
    if declared_sources and not cited_sources:
        raise ValueError(
            "evidence에는 출처가 있지만 본문에는 인용이 없습니다."
        )

    # 4. 본문에서 실제 인용한 출처로 목록을 맞춘다.
    # - 본문에 인용했지만 evidence에서 빠진 출처 추가
    # - 본문에서 인용하지 않은 출처 제외
    result.evidence = cited_sources


def _validate_search_plans(
    plans: list[MarketSearchQuery],
) -> None:
    """검색 횟수, 목적 중복, 빈 검색어를 검사한다."""
    if len(plans) > MAX_WEB_QUERIES:
        raise ValueError("웹 검색 계획이 허용 횟수를 초과했습니다.")

    purposes = [plan.purpose for plan in plans]

    if len(purposes) != len(set(purposes)):
        raise ValueError(
            "웹 검색 계획에 같은 목적이 중복되었습니다."
        )

    for plan in plans:
        plan.query = plan.query.strip()

        if not plan.query:
            raise ValueError("웹 검색어가 비어 있습니다.")


def run(state: GraphState) -> dict:
    # 1. 현재 분석할 기업 확인
    cid = state["current_candidate"]

    if not cid:
        raise ValueError("current_candidate가 비어 있습니다.")

    candidate = next(
        (
            item
            for item in state["candidates"]
            if item["company_id"] == cid
        ),
        None,
    )

    if candidate is None:
        raise ValueError(
            f"후보 목록에서 company_id={cid}인 기업을 찾지 못했습니다."
        )

    # 2. 해당 기업의 시장성 페이지 조회
    company_chunks = [
        chunk
        for chunk in get_company_pages(cid)
        if chunk["page_type"] == "company_market"
    ]

    # 3. 기업의 사업 영역에 맞는 시장 공통 자료 검색
    market_query = (
        f"{candidate['domain']} "
        f"{candidate['description']} "
        "시장 규모 성장 수요 구매 고객 사업 모델 도입 장벽"
    )

    market_chunks = search(
        market_query,
        top_k=3,
        page_types=["market_topic"],
    )

    # 시장 공통 자료에는 company_id 필터를 넣지 않는다.
    rag_chunks = company_chunks + market_chunks

    allowed_sources = {
        source_id
        for chunk in rag_chunks
        for source_id in chunk["source_ids"]
    }

    inputs = {
        "query": state["query"],
        "candidate_info": _to_json(candidate),
        "company_context": _to_json(company_chunks),
        "market_context": _to_json(market_chunks),
    }

    # 4. RAG 초기 분석
    # 이 호출에서 분석 결과와 필요한 웹 검색 계획을 함께 받는다.
    model = get_llm().with_structured_output(MarketAnalysis)

    book = SourceBook(canonicalize=True, rag_ids=set(allowed_sources))  # 웹 검색 결과 ↔ label(U1…)

    initial = _analyze_checked(
        model=model,
        inputs=inputs,
        stage="RAG 초기 분석",
        web_context={
            "status": "아직 웹 검색을 실행하지 않음",
        },
        allowed_sources=allowed_sources,
        book=book,
    )

    # 미확인 사항의 빈 문자열과 중복 제거
    missing = list(dict.fromkeys(
        question.strip()
        for question in initial.details.missing_information
        if question.strip()
    ))

    search_plans = list(initial.details.web_search_queries)

    # 기업별 자료가 전혀 없으면 고객·제품 확인 검색을 확보한다.
    if not company_chunks:
        question = (
            f"{candidate['company_name']}의 목표 고객과 "
            "제품의 수익 구조를 확인할 공식 자료가 필요함."
        )

        if question not in missing:
            missing.insert(0, question)

        has_customer_search = any(
            plan.purpose == "customer_adoption"
            for plan in search_plans
        )

        if not has_customer_search:
            search_plans.append(
                MarketSearchQuery(
                    purpose="customer_adoption",
                    query=(
                        f"{candidate['company_name']} "
                        "제품 목표 고객 도입 사례 공식"
                    ),
                )
            )

    _validate_search_plans(search_plans)

    initial.details.missing_information = missing

    final = initial
    search_log = []

    # 5. 검색 계획이 있을 때만 웹 검색
    if search_plans:
        if not os.getenv("TAVILY_API_KEY", "").strip():
            raise RuntimeError(
                "웹 보완 검색이 필요하지만 TAVILY_API_KEY가 없습니다. "
                ".env에 설정한 뒤 프로그램을 다시 실행하세요."
            )

        for plan in search_plans:
            # 시장 검색에는 기업명이 없을 수 있으므로
            # 기업명을 일괄 추가하지 않고 계획의 검색어를 사용한다.
            search_query = plan.query

            try:
                results = web_search(
                    search_query,
                    max_results=WEB_RESULTS_PER_QUERY,
                )
            except Exception as exc:
                # 검색 실패를 '검색 결과 없음'과 구분한다.
                # 민감한 정보 노출을 피하기 위해 오류 종류만 기록한다.
                search_log.append({
                    "purpose": plan.purpose,
                    "query": search_query,
                    "status": "failed",
                    "error_type": type(exc).__name__,
                })
                continue

            usable_results = [
                item
                for item in results
                if item.get("url", "").startswith(
                    ("https://", "http://")
                )
                and item.get("content", "").strip()
            ]

            # 같은 페이지의 자료는 한 번만 등록하고 label(U1…)을 붙인다.
            shown = book.add(usable_results)

            # success는 자료 반환 성공이며 사실 검증 완료가 아니다.
            search_log.append({
                "purpose": plan.purpose,
                "query": search_query,
                "status": (
                    "success" if usable_results else "no_results"
                ),
                "urls": [source.url for source in shown],
            })

        # 6. 웹 자료가 확보되면 RAG와 함께 최종 분석
        if book.sources:
            # LLM 에게는 URL 대신 label 을 보여 준다.
            prompt_log = [
                {**entry, "urls": [book.label_of(url) for url in entry["urls"]]}
                if "urls" in entry else entry
                for entry in search_log
            ]

            final = _analyze_checked(
                model=model,
                inputs=inputs,
                stage="웹 보완 후 최종 분석",
                web_context={
                    "initial_missing_information": missing,
                    "search_plan": [
                        plan.model_dump()
                        for plan in search_plans
                    ],
                    "search_log": prompt_log,
                    "results": [
                        {"label": source.label, "title": source.title, "content": source.content}
                        for source in book.sources
                    ],
                    "note": (
                        "검색 결과에 제공된 내용만 사용한다. "
                        "검색 성공은 정보 확인 완료를 뜻하지 않는다. "
                        "기존 자료나 새 자료로 실제 확인된 항목만 "
                        "missing_information에서 제거한다. "
                        "추가 검색 계획은 만들지 않고 "
                        "web_search_queries는 빈 목록으로 반환한다."
                    ),
                },
                allowed_sources=allowed_sources,
                book=book,
            )

        else:
            # 검색 실패 또는 결과 없음:
            # 초기 분석과 미확인 사항을 그대로 유지한다.
            final.details.missing_information.append(
                "웹 보완 검색에서 활용 가능한 근거를 확보하지 못함. "
                "검색 기록을 확인하고 추가 조사 필요."
            )

    # 7. 공통 State 형식으로 변환
    data = final.model_dump()

    # 이번 실행에서 검색은 종료했다.
    # 수행한 계획은 executed_search_plan에 별도로 보관한다.
    data["details"]["web_search_queries"] = []
    data["details"]["initial_missing_information"] = missing
    data["details"]["executed_search_plan"] = [
        plan.model_dump()
        for plan in search_plans
    ]
    data["details"]["web_search_log"] = search_log

    # 현재 기업의 market 필드만 반환한다. 웹 출처의 제목은 sources 로 함께 남긴다(보고서 REFERENCE 용).
    update = evaluation_update(
        cid,
        market=make_analysis(**data),
    )
    update["sources"] = book.registry(used=data["evidence"])
    return update