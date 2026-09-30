"""
경쟁사 비교 (웹서치)

입력: state["current_candidate"], state["candidates"]
출력: evaluation_update(cid, competition=AnalysisResult)

- tools.web_search.web_search 로 피어 그룹 탐색, 경쟁 구도·차별성 분석
- details 에 보고서 4절 "경쟁 제품 비교" 표를 채울 수 있는 구조화 데이터를 담는다.
- 점수는 매기지 않는다.
"""

import json

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from pydantic import BaseModel, Field
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import LLM_MODEL, LLM_PROVIDER
from core.sources import SourceBook
from core.state import GraphState, evaluation_update, make_analysis
from prompts.competitor_compare import COMPETITION_INSTRUCTIONS
from tools.web_search import web_search


class PeerProduct(BaseModel):
    company_name: str = ""
    product_name: str = ""
    customer_and_use: str = ""
    performance_and_power: str = ""
    price_and_adoption: str = ""
    strengths_and_weaknesses: str = ""
    source_url: str = ""


class CompetitionDetails(BaseModel):
    peer_products: list[PeerProduct] = Field(default_factory=list)
    comparison_conditions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    critical_risks: list[str] = Field(default_factory=list)


class CompetitionResult(BaseModel):
    summary: str
    strengths: list[str]
    risks: list[str]
    evidence: list[str]
    details: CompetitionDetails


MAX_SUBMIT_FAILURES = 2  # 검증에 실패한 제출을 돌려보내 다시 쓰게 하는 최대 횟수. 그 뒤에는 잘못된 출처만 빼고 받아들인다


def _check(result: dict, book: SourceBook) -> str:
    """제출 결과의 문제를 한 줄로. 없으면 빈 문자열."""
    problems = []
    cited = [*result["evidence"], *(peer["source_url"] for peer in result["details"]["peer_products"] if peer["source_url"])]
    known = set(book.resolve(cited))
    unknown = [ref for ref in cited if ref.strip() and not book.resolve([ref])]
    book.unknown.clear()  # 검사용 호출이 남긴 기록은 지운다 (최종 변환에서 다시 기록)
    if unknown:
        problems.append(f"검색 결과에 없는 출처 번호입니다: {', '.join(dict.fromkeys(unknown))}")
    if (result["strengths"] or result["details"]["peer_products"]) and not known:
        problems.append("경쟁 우위나 경쟁 제품을 제시하려면 근거가 필요합니다")
    return "; ".join(problems)


def _finalize(result: dict, book: SourceBook, problems: str) -> dict:
    """LLM 이 쓴 출처 번호를 표준 URL 로 바꾼다(본문 인용 포함). 검색 결과에 없는 출처는 뺀다."""
    book.unknown.clear()
    text = book.restore
    details = result["details"]
    for peer in details["peer_products"]:
        for key in ("company_name", "product_name", "customer_and_use", "performance_and_power",
                    "price_and_adoption", "strengths_and_weaknesses"):
            peer[key] = text(peer[key])
        urls = book.resolve([peer["source_url"]]) if peer["source_url"] else []
        peer["source_url"] = urls[0] if urls else ""
    for key in ("comparison_conditions", "missing_information", "critical_risks"):
        details[key] = [text(item) for item in details[key]]
    result["summary"] = text(result["summary"])
    result["strengths"] = [text(item) for item in result["strengths"]]
    result["risks"] = [text(item) for item in result["risks"]]
    result["evidence"] = book.resolve(result["evidence"])
    for peer in details["peer_products"]:  # 표에 쓴 출처는 근거에도 있어야 보고서 REFERENCE 로 이어진다
        if peer["source_url"] and peer["source_url"] not in result["evidence"]:
            result["evidence"].append(peer["source_url"])
    if book.unknown or problems:
        removed = len(book.unknown)
        details["missing_information"].append(
            f"제공되지 않은 출처 표기 {removed}건을 제거했다. 해당 근거는 확인이 필요하다." if removed else f"출처 검증 문제가 남았다: {problems}")
    return result


def _insufficient(cid: str, reason: str) -> dict:
    """검증을 통과한 결과를 끝내 내지 못한 경우 — 그래프를 멈추지 않고 근거 부족으로 넘긴다."""
    return evaluation_update(cid, competition=make_analysis(
        summary=f"경쟁사 비교를 완료하지 못했다: {reason}",
        details={"status": "insufficient_evidence", "reason": reason, "missing_information": [f"경쟁사 비교 전체: {reason}"]},
    ))


def run(state: GraphState) -> dict:
    cid = state["current_candidate"]
    if not isinstance(cid, str) or not cid:
        raise ValueError("current_candidate에 기업 ID가 필요합니다.")

    company = next(
        (item for item in state["candidates"] if item["company_id"] == cid),
        None,
    )
    if company is None:
        raise ValueError(f"후보 목록에서 {cid}를 찾지 못했습니다.")

    book = SourceBook(canonicalize=True)  # 검색 결과 ↔ 출처 번호(U1…) — LLM 은 URL 을 쓰지 않는다
    search_calls = 0
    submit_failures = 0
    accepted_result: dict | None = None

    @tool
    def competitor_web_search(query: str) -> str:
        """경쟁 제품과 경쟁사 자료를 웹에서 검색한다."""
        nonlocal search_calls
        if search_calls >= 3:
            return "검색 한도에 도달했습니다. 확보한 출처로 결과를 작성하세요."
        search_calls += 1
        try:
            found = web_search(query, max_results=5)
        except Exception as exc:
            return f"검색 연결 오류 ({type(exc).__name__}). 다른 검색어로 다시 시도하세요."
        shown = book.add(found)
        if not shown:
            return "검색 결과가 없습니다. 다른 검색어로 다시 시도하세요."
        return "\n\n".join(book.render(item) for item in shown)

    @tool
    def submit_competition_result(result: CompetitionResult) -> str:
        """경쟁 제품 비교 결과를 제출하고 검색 출처를 검증한다."""
        nonlocal accepted_result, submit_failures
        candidate_result = CompetitionResult.model_validate(result).model_dump()
        problems = _check(candidate_result, book)
        if problems and submit_failures < MAX_SUBMIT_FAILURES:
            submit_failures += 1
            labels = ", ".join(item.label for item in book.sources) or "없음(먼저 검색하세요)"
            return f"검증 실패: {problems}. 사용할 수 있는 출처 번호: {labels}. 결과를 수정해 다시 제출하세요. 이 과정은 summary에 쓰지 마세요."
        accepted_result = _finalize(candidate_result, book, problems)
        return "검증 통과. 분석이 완료되었습니다."

    model = init_chat_model(
        model=LLM_MODEL,
        model_provider=LLM_PROVIDER,
        use_responses_api=True,
        reasoning_effort="low",
        max_tokens=4000,
        timeout=30,
    )
    agent = create_agent(
        model=model,
        tools=[competitor_web_search, submit_competition_result],
        system_prompt=COMPETITION_INSTRUCTIONS,
    )
    agent.invoke(
        {"messages": [{
            "role": "user",
            "content": f"다음 기업의 경쟁 제품을 비교하세요: {json.dumps(company, ensure_ascii=False)}",
        }]},
        config={"recursion_limit": 24},
    )

    if accepted_result is None:
        return _insufficient(cid, "에이전트가 분석 결과를 제출하지 않았다")

    update = evaluation_update(cid, competition=make_analysis(**accepted_result))
    update["sources"] = book.registry(used=accepted_result["evidence"])  # 출처 제목 (보고서 REFERENCE 용)
    return update


"""테스트용 데이터 1"""
# test_state: GraphState = {
#     "query": "SK하이닉스 HBM 경쟁 제품 비교",
#     "candidates": [
#         {
#             "company_id": "SK_HYNIX",
#             "company_name": "SK하이닉스",
#             "description": (
#                 "DRAM, HBM, NAND 플래시 등 메모리 반도체를 개발·생산하며 AI 서버용 고대역폭 메모리를 공급하는 기업"
#             ),
#             "domain": "HBM 메모리 반도체",
#             "retrieval_score": 0.0,
#         }
#     ],
#     "candidate_index": 1,
#     "current_candidate": "SK_HYNIX",
#     "evaluations": {},
#     "selected_candidate": None,
#     "report": None,
# }
"""테스트용 데이터 2"""
# test_state: GraphState = {
#     "query": "싸이닉솔루션 시스템반도체 디자인하우스 경쟁사 비교",
#     "candidates": [
#         {
#             "company_id": "SYNIC_SOLUTION",
#             "company_name": "싸이닉솔루션",
#             "description": (
#                 "팹리스 고객에게 반도체 설계 및 파운드리 연계 서비스를 제공하고, "
#                 "ASIC 개발·양산과 시스템반도체 디자인 솔루션 사업을 수행하는 기업"
#             ),
#             "domain": "시스템반도체 디자인하우스 및 ASIC",
#             "retrieval_score": 0.0,
#         }
#     ],
#     "candidate_index": 1,
#     "current_candidate": "SYNIC_SOLUTION",
#     "evaluations": {},
#     "selected_candidate": None,
#     "report": None,
# }
"""테스트용 데이터 3"""
# test_state: GraphState = {
#     "query": "DEEPX 엣지 AI 반도체 경쟁 제품 비교",
#     "candidates": [
#         {
#             "company_id": "DEEPX",
#             "company_name": "DEEPX",
#             "description": (
#                 "엣지 기기에서 AI 추론을 수행하는 저전력 NPU를 개발하는 기업. "
#                 "DX-M1 등 AI 가속 칩과 모듈을 제공한다."
#             ),
#             "domain": "엣지 AI 추론용 NPU 및 AI 가속기",
#             "retrieval_score": 0.0,
#         }
#     ],
#     "candidate_index": 1,
#     "current_candidate": "DEEPX",
#     "evaluations": {},
#     "selected_candidate": None,
#     "report": None,
# }
"""테스트용 데이터 4"""
# test_state: GraphState = {
#     "query": "Panmnesia CXL 메모리 확장 솔루션 경쟁 제품 비교",
#     "candidates": [
#         {
#             "company_id": "PANMNESIA",
#             "company_name": "Panmnesia",
#             "description": (
#                 "데이터센터와 AI 인프라의 메모리 확장·공유를 위한 "
#                 "CXL 컨트롤러 IP, 스위치 칩 및 연결 솔루션을 개발하는 기업"
#             ),
#             "domain": "CXL 기반 메모리 확장 및 데이터센터 인터커넥트",
#             "retrieval_score": 0.0,
#         }
#     ],
#     "candidate_index": 1,
#     "current_candidate": "PANMNESIA",
#     "evaluations": {},
#     "selected_candidate": None,
#     "report": None,
# }
# print(json.dumps(run(test_state), ensure_ascii=False, indent=2))
