"""
Supervisor 정책 — 근거 충분성 규칙과 상수 (LLM 호출 없음, 결정적)

분석 에이전트(기술·시장성·경쟁·창업자/실적)는 점수를 매기지 않고 AnalysisResult 만 남긴다.
Supervisor 는 이 결과가 투자 판단으로 넘어갈 만큼 충분한지를 여기서 판정한다.

- 분석 노드가 스스로 "근거 부족"을 표시한 경우(details["status"] 가 INSUFFICIENT_STATUSES 중 하나)
- 출처(evidence)가 MIN_EVIDENCE 개 미만인 경우
→ insufficient. 재작업 상한(MAX_REWORK) 안에서만 다시 요청한다.

투자 판단 노드의 insufficient_evidence(필수 항목 미산정 HOLD)는 점수화 이후의 판정이다.
여기서는 분석 직후의 입력 충분성만 본다.
"""

from typing import Literal, Optional

from core.state import AnalysisResult

# 노드 이름 (graph/builder.py 의 상수와 같아야 한다. tests 에서 일치를 확인한다)
CANDIDATE_SELECT = "candidate_select"
TECH_SUMMARY = "tech_summary"
MARKET_EVAL = "market_eval"
COMPETITOR_COMPARE = "competitor_compare"
FOUNDER_TRACTION = "founder_traction"
INVESTMENT_DECISION = "investment_decision"
REPORT_WRITER = "report_writer"
QUALITY_CHECK = "quality_check"
# 그래프 종료. Supervisor 가 next_agent 에 이 값을 쓰면 add_conditional_edges 가 END 로 보낸다 (langgraph.graph.END 와 같은 값)
END_TARGET = "__end__"

# 보고서 단계의 진행 표시. node_status 에 쓴다 (후보와 무관해서 company_id 대신 "report" 를 쓴다).
# "pending" = 그 노드로 보냈고 아직 돌아오지 않았다 → Supervisor 가 다시 진입했다면 돌아온 것이다.
REPORT_STAGE_KEYS = {REPORT_WRITER: f"report:{REPORT_WRITER}", QUALITY_CHECK: f"report:{QUALITY_CHECK}"}

# 분석 에이전트 → 그 에이전트가 evaluations[cid] 에 남기는 필드.
# 필드가 있으면 그 에이전트가 한 번은 실행된 것이다 (분석 노드는 node_status 를 직접 쓰지 않는다).
# 한 에이전트가 필드를 여럿 쓰면(창업자/실적) 첫 필드가 완료 표지이고, 충분성은 모든 필드를 본다.
# 이 표는 에이전트와 결과 필드의 대응이다. 실행 순서는 LLM Supervisor 가 정한다(llm_policy.py, prompts/supervisor.py).
# 딕셔너리 순서는 LLM 을 쓰지 못할 때의 규칙 기반 대체 경로(node._decide)에서만 쓰인다.
ANALYSIS_AGENTS: dict[str, tuple[str, ...]] = {
    TECH_SUMMARY: ("technology",),
    MARKET_EVAL: ("market",),
    COMPETITOR_COMPARE: ("competition",),
    FOUNDER_TRACTION: ("team", "traction"),
}

# 설정값: 실행 측정(Supervisor 진입 횟수, 재작업 효과)을 보고 조정한다.
MAX_REWORK = 1  # 에이전트별 재작업 상한
MIN_EVIDENCE = 1  # 분석 결과 하나가 가져야 하는 최소 출처 수

# 분석 노드가 details["status"] 에 남기는 근거 부족 표시 (tech/market/competition: insufficient_evidence, 창업자/실적: no_sources, unverified)
INSUFFICIENT_STATUSES = {"insufficient_evidence", "no_sources", "unverified"}

Sufficiency = Literal["ok", "insufficient"]


def assess_sufficiency(analysis: Optional[AnalysisResult]) -> tuple[Sufficiency, str]:
    """분석 결과 1개의 근거 충분성과 사유(한 줄). 결과가 없으면 insufficient."""
    if not analysis:
        return "insufficient", "분석 결과가 없다"

    details = analysis.get("details") or {}
    if details.get("status") in INSUFFICIENT_STATUSES:
        return "insufficient", f"분석 노드가 근거 부족을 표시했다({details['status']}): {details.get('reason', '사유 없음')}"[:200]

    count = len([e for e in analysis.get("evidence") or [] if isinstance(e, str) and e.strip()])
    if count < MIN_EVIDENCE:
        return "insufficient", f"출처 {count}건 (최소 {MIN_EVIDENCE}건)"

    return "ok", ""


def assess_agent(evaluation: dict, agent: str) -> tuple[Sufficiency, str]:
    """에이전트가 남긴 모든 필드를 보고 충분성을 판정한다. 하나라도 부족하면 insufficient."""
    for field in ANALYSIS_AGENTS[agent]:
        verdict, reason = assess_sufficiency(evaluation.get(field))
        if verdict == "insufficient":
            return verdict, f"{field}: {reason}"
    return "ok", ""
