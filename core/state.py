from typing import TypedDict, Optional, Annotated, Literal, Any, NotRequired

# =========================
# 0. evaluations 병합 함수
# 같은 기업의 분석 결과를 덮어쓰지 않고 누적
# =========================


def merge_evaluations(
    old: dict[str, "CandidateEvaluation"], new: dict[str, "CandidateEvaluation"]
) -> dict[str, "CandidateEvaluation"]:

    result = {**old}

    for company_id, new_evaluation in new.items():

        if company_id not in result:
            result[company_id] = new_evaluation

        else:
            result[company_id] = {**result[company_id], **new_evaluation}

    return result


# =========================
# 0-1. sources 병합 함수
# 웹 출처(URL)의 메타데이터. 같은 URL 은 같은 페이지라 값이 같으므로 비어 있지 않은 값을 살려 합친다.
# =========================


class SourceInfo(TypedDict, total=False):
    title: str
    retrieved_at: str


def merge_sources(old: dict[str, "SourceInfo"], new: dict[str, "SourceInfo"]) -> dict[str, "SourceInfo"]:
    result = {url: dict(info) for url, info in old.items()}

    for url, info in new.items():
        merged = result.setdefault(url, {})
        for key, value in info.items():
            if value or key not in merged:
                merged[key] = value

    return result


# =========================
# 1. 후보 기업 정보
# 스타트업 탐색 Agent의 출력
# =========================


class Candidate(TypedDict):
    company_id: str
    company_name: str
    description: str
    domain: str
    retrieval_score: float


# =========================
# 2. Agent 공통 분석 결과
#
# Agent는 점수를 매기지 않고
# 사실 / 근거 / 강점 / 위험요소만 수집
#
# details에는 Agent별 고유 구조화 데이터 저장
# =========================


class AnalysisResult(TypedDict):
    summary: str
    strengths: list[str]
    risks: list[str]
    evidence: list[str]

    # Agent별 구조화 데이터
    details: dict[str, Any]


# =========================
# 3. Scorecard 결과
#
# 모든 분석이 끝난 뒤
# 투자 판단 Node에서 한 번만 점수 계산
# =========================


class ScorecardResult(TypedDict):
    # 미산정 항목은 None: 0점이나 기본점수로 채우지 않고 제외한 뒤 채점된 항목의 비중으로 total_score 를 환산한다
    team: Optional[float]
    market: Optional[float]
    technology: Optional[float]
    competition: Optional[float]
    traction: Optional[float]
    deal_terms: Optional[float]
    total_score: float
    coverage: NotRequired[float]  # 채점된 항목의 비중 합 (1.0 이면 여섯 항목 모두 채점)
    unscored: NotRequired[list[str]]  # 미산정 항목


# =========================
# 4. 후보 1개에 대한 전체 평가 결과
# =========================


class CandidateEvaluation(TypedDict, total=False):

    # 각 Agent의 조사 결과
    technology: AnalysisResult
    market: AnalysisResult
    competition: AnalysisResult

    team: AnalysisResult
    traction: AnalysisResult
    deal_terms: AnalysisResult

    # 모든 조사가 끝난 뒤 생성
    scorecard: ScorecardResult

    # 최종 판단
    decision: Literal["INVEST", "HOLD"]


# =========================
# 5. LangGraph 전체 State
# =========================


class GraphState(TypedDict):

    # 사용자 최초 질의
    query: str

    # 스타트업 탐색 Agent가 찾은 Top-K 후보
    candidates: list[Candidate]

    # 현재 몇 번째 후보를 평가 중인지
    candidate_index: int

    # 현재 분석 중인 기업 ID
    current_candidate: Optional[str]

    # 후보별 전체 분석 결과 누적
    # key = company_id
    evaluations: Annotated[dict[str, CandidateEvaluation], merge_evaluations]

    # 웹 출처 메타데이터 {표준 URL: {title, retrieved_at}}
    # 분석 Agent 가 evidence 에 남긴 URL 의 제목을 모아 보고서 REFERENCE 가 쓴다 (없어도 동작한다)
    sources: Annotated[dict[str, SourceInfo], merge_sources]

    # 최종 투자 후보
    selected_candidate: Optional[str]

    # 최종 보고서
    report: Optional[str]


# =========================
# 6. 노드 작성용 헬퍼
# =========================


def initial_state(query: str) -> GraphState:
    return {
        "query": query,
        "candidates": [],
        "candidate_index": 0,
        "current_candidate": None,
        "evaluations": {},
        "sources": {},
        "selected_candidate": None,
        "report": None,
    }


def make_analysis(
    summary: str = "",
    strengths: Optional[list[str]] = None,
    risks: Optional[list[str]] = None,
    evidence: Optional[list[str]] = None,
    details: Optional[dict[str, Any]] = None,
) -> AnalysisResult:
    return {
        "summary": summary,
        "strengths": strengths or [],
        "risks": risks or [],
        "evidence": evidence or [],
        "details": details or {},
    }


def evaluation_update(company_id: str, **fields: Any) -> dict:
    """
    분석 노드의 반환값을 만든다. 자기 담당 필드만 넘길 것.

    예) return evaluation_update(cid, technology=make_analysis(...))
    merge_evaluations 가 기존 결과와 병합하므로 다른 노드의 필드는 유지된다.
    """
    return {"evaluations": {company_id: fields}}
