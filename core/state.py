import uuid
from datetime import datetime
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
# 0-2. 제어 필드 병합 함수 (Supervisor 패턴)
# node_status / rework_count 처럼 키 단위로 부분 갱신되는 dict 를 덮어쓰지 않고 합친다.
# Supervisor 는 한 번에 한 노드만 실행하므로 지금은 동시 쓰기가 없지만, 병렬 확장 시 덮어쓰기를 막는 안전장치다.
# =========================


def merge_dict(old: dict, new: dict) -> dict:
    return {**old, **new}


# =========================
# 0-3. 제어 메타데이터용 타입
# =========================

# "{company_id}:{agent}" 단위의 진행 상태
NodeStatus = Literal["pending", "done", "failed", "needs_rework", "excluded"]


class Decision(TypedDict):
    # Supervisor 의 마지막 라우팅 결정 1건. 결정 로그 본문은 State 가 아니라 외부 로그(trace_id 로 연결)에 둔다.
    next: str
    reason: str


class ReworkNote(TypedDict):
    # 재작업을 요청할 때 Supervisor 가 남기는 지시. 재작업되는 에이전트가 읽는다 (core/rework.py)
    missing: list[str]  # 근거가 부족하다고 평가한 항목
    queries: list[str]  # 그 항목을 찾기 위해 Supervisor 가 정한 웹 검색어


class QualityVerdict(TypedDict):
    # 보고서 품질 평가 결과. 미달이면 feedback 이 재작업·재작성 판단의 입력이 된다.
    groundedness: bool  # 주장이 [번호] 출처로 추적되는가
    neutrality: bool  # 근거 없는 단정·후보 간 우열 서술이 없는가
    bias_control: bool  # 단일 출처·유리한 근거 편중이 없는가
    passed: bool
    feedback: list[str]


# TODO: 설계 문서에 미정 — 팀 합의 후 확정 (docs/State-Schema-설계.md 6절)
# Supervisor 진입 횟수 상한(안전망). 시작 1회, 후보 1곳 평가에 6회, 보고서 단계에 2회(작성 후·평가 후)가 들고,
# 재작업·재시도·보고서 재작성이 1회씩 더해진다. 후보 3곳 모두 HOLD면 21회, 후보마다 재작업 1회면 24회이고,
# 후보 3곳·에이전트 4개가 모두 재작업되고 품질도 계속 미달인 최악은 약 35회다(MAX_REWORK=1, 스텁 측정).
# 이 값은 정상 흐름에서는 닿지 않는 안전망이다. MAX_REWORK 를 올리면 다시 계산해야 한다.
# recursion_limit 은 이 값에서 계산한다(audit.recursion_limit).
DEFAULT_MAX_STEPS = 50


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
    # 설계 원칙 (docs/State-Schema-설계.md):
    # - State = "작업 결과(페이로드)" + "조정·종료·재개에 필요한 최소 제어 메타"만 담는다.
    # - 결정 로그·트레이스 본문은 State 에 넣지 않는다. 외부 로거/LangSmith 로 보내고 trace_id 로 State 와 잇는다.
    # - 각 노드는 자기 필드만 반환한다. 제어 필드는 Supervisor 가 쓴다.

    # ── 작업 페이로드 ───────────────────────────

    # 사용자 최초 질의
    query: str

    # 스타트업 탐색 Agent가 찾은 Top-K 후보
    candidates: list[Candidate]

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

    # 보고서 품질 평가 결과 (quality_check 노드)
    quality_verdict: Optional[QualityVerdict]

    # ── 제어 메타데이터 (조정·종료·재개에 필요한 최소치) ──

    # 현재 몇 번째 후보를 평가 중인지
    candidate_index: int

    # 현재 분석 중인 기업 ID
    current_candidate: Optional[str]

    # 외부 로그/트레이스와 State 를 잇는 상관 키 (실행마다 생성)
    trace_id: str

    # 실행 시작 시각(ISO, 로컬 시간). 이 실행이 만드는 보고서 파일명(md·PDF)에 같은 시각을 써서 서로 짝이 보이게 하고,
    # 보고서를 다시 써도 같은 이름으로 덮어쓰게 한다.
    started_at: str

    # Supervisor 진입 횟수와 상한 (종료 가드)
    step_count: int
    max_steps: int

    # Supervisor 가 정한 다음 노드 이름 (add_conditional_edges 의 분기 키)
    next_agent: Optional[str]

    # 재개·재시도 판단용 최소 상태. key = "{company_id}:{agent}"
    node_status: Annotated[dict[str, NodeStatus], merge_dict]

    # 에이전트별 재작업 횟수 (이력 리스트 대신 카운터만 둬 체크포인트 증식을 막는다). key = "{company_id}:{agent}"
    rework_count: Annotated[dict[str, int], merge_dict]

    # 재작업을 요청할 때 Supervisor 가 남기는 "부족한 항목"과 검색어 (최신 지시 한 건만 덮어쓴다, 이력은 쌓지 않는다).
    # 재작업되는 에이전트가 검색어·프롬프트에 반영한다. key = "{company_id}:{agent}"
    rework_notes: Annotated[dict[str, ReworkNote], merge_dict]

    # 마지막 노드 오류와 마지막 라우팅 결정 1건
    last_error: Optional[str]
    last_decision: Optional[Decision]

    # 보고서 재작성 횟수 (종료 가드)
    quality_attempts: int


# =========================
# 6. 노드 작성용 헬퍼
# =========================


def initial_state(query: str, max_steps: int = DEFAULT_MAX_STEPS) -> GraphState:
    return {
        "query": query,
        "candidates": [],
        "evaluations": {},
        "sources": {},
        "selected_candidate": None,
        "report": None,
        "quality_verdict": None,
        "candidate_index": 0,
        "current_candidate": None,
        "trace_id": uuid.uuid4().hex,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "step_count": 0,
        "max_steps": max_steps,
        "next_agent": None,
        "node_status": {},
        "rework_count": {},
        "rework_notes": {},
        "last_error": None,
        "last_decision": None,
        "quality_attempts": 0,
    }


def agent_key(company_id: str, agent: str) -> str:
    """node_status / rework_count 의 키. 후보와 에이전트를 함께 식별한다."""
    return f"{company_id}:{agent}"


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
