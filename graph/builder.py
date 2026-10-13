"""
그래프 조립. 노드 추가·순서 변경 시에만 수정하고, 에이전트 로직은 agents/ 에서 작업한다.
"""

from typing import Callable, Optional

from langgraph.graph import END, START, StateGraph

from agents import (
    candidate_select,
    competitor_compare,
    founder_traction,
    investment_decision,
    market_eval,
    quality_check,
    report_writer,
    startup_search,
    supervisor,
    tech_summary,
)
from agents.supervisor.guard import guarded
from core.state import GraphState

# 노드 이름
STARTUP_SEARCH = "startup_search"
SUPERVISOR = "supervisor"
CANDIDATE_SELECT = "candidate_select"
TECH_SUMMARY = "tech_summary"
MARKET_EVAL = "market_eval"
COMPETITOR_COMPARE = "competitor_compare"
FOUNDER_TRACTION = "founder_traction"
INVESTMENT_DECISION = "investment_decision"
REPORT_WRITER = "report_writer"
QUALITY_CHECK = "quality_check"

DEFAULT_NODES: dict[str, Callable] = {
    STARTUP_SEARCH: startup_search.run,
    CANDIDATE_SELECT: candidate_select.run,
    TECH_SUMMARY: tech_summary.run,
    MARKET_EVAL: market_eval.run,
    COMPETITOR_COMPARE: competitor_compare.run,
    FOUNDER_TRACTION: founder_traction.run,
    INVESTMENT_DECISION: investment_decision.run,
    REPORT_WRITER: report_writer.run,
    QUALITY_CHECK: quality_check.run,
}

# Supervisor 가 라우팅하는 하위 노드. 보고서 작성과 품질 평가도 포함한다 (모두 Supervisor 로만 돌아온다).
# 실행 순서는 여기서 정하지 않는다 (Supervisor 가 State 를 보고 정한다).
WORKERS = [
    CANDIDATE_SELECT,
    TECH_SUMMARY,
    MARKET_EVAL,
    COMPETITOR_COMPARE,
    FOUNDER_TRACTION,
    INVESTMENT_DECISION,
    REPORT_WRITER,
    QUALITY_CHECK,
]

# 예외가 나도 그래프가 멈추지 않도록 감싸는 노드 (실패는 Supervisor 가 재시도·제외로 처리한다.
# 품질 평가가 실패하면 이미 만들어진 보고서를 잃지 않도록 감싼다)
GUARDED = {TECH_SUMMARY, MARKET_EVAL, COMPETITOR_COMPARE, FOUNDER_TRACTION, QUALITY_CHECK}


def build_graph(overrides: Optional[dict[str, Callable]] = None):
    """overrides 로 특정 노드를 교체할 수 있다 (테스트·단독 디버깅용).

    허브 구조: 모든 하위 에이전트는 Supervisor 로만 돌아오고(에이전트 간 직접 edge 없음),
    Supervisor 가 next_agent 로 add_conditional_edges 분기를 한다.
    """
    nodes = {**DEFAULT_NODES, **(overrides or {})}

    g = StateGraph(GraphState)
    for name, fn in nodes.items():
        g.add_node(name, guarded(name, fn) if name in GUARDED else fn)
    g.add_node(SUPERVISOR, supervisor.run)

    g.add_edge(START, STARTUP_SEARCH)
    g.add_edge(STARTUP_SEARCH, SUPERVISOR)
    for worker in WORKERS:
        g.add_edge(worker, SUPERVISOR)
    g.add_conditional_edges(SUPERVISOR, supervisor.route, [*WORKERS, END])

    return g.compile()
