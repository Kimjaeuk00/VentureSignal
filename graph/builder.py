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
    report_writer,
    startup_search,
    tech_summary,
)
from core.state import GraphState

# 노드 이름
STARTUP_SEARCH = "startup_search"
CANDIDATE_SELECT = "candidate_select"
TECH_SUMMARY = "tech_summary"
MARKET_EVAL = "market_eval"
COMPETITOR_COMPARE = "competitor_compare"
FOUNDER_TRACTION = "founder_traction"
INVESTMENT_DECISION = "investment_decision"
REPORT_WRITER = "report_writer"

DEFAULT_NODES: dict[str, Callable] = {
    STARTUP_SEARCH: startup_search.run,
    CANDIDATE_SELECT: candidate_select.run,
    TECH_SUMMARY: tech_summary.run,
    MARKET_EVAL: market_eval.run,
    COMPETITOR_COMPARE: competitor_compare.run,
    FOUNDER_TRACTION: founder_traction.run,
    INVESTMENT_DECISION: investment_decision.run,
    REPORT_WRITER: report_writer.run,
}

# 후보 분석 구간 (순차 실행)
ANALYSIS_SEQUENCE = [TECH_SUMMARY, MARKET_EVAL, COMPETITOR_COMPARE, FOUNDER_TRACTION]


def route_after_search(state: GraphState) -> str:
    return CANDIDATE_SELECT if state["candidates"] else REPORT_WRITER


def route_after_decision(state: GraphState) -> str:
    cid = state["current_candidate"]
    decision = state["evaluations"].get(cid, {}).get("decision")
    if decision == "INVEST":
        return REPORT_WRITER
    if state["candidate_index"] < len(state["candidates"]):
        return CANDIDATE_SELECT
    return REPORT_WRITER


def build_graph(overrides: Optional[dict[str, Callable]] = None):
    """overrides 로 특정 노드를 교체할 수 있다 (테스트·단독 디버깅용)."""
    nodes = {**DEFAULT_NODES, **(overrides or {})}

    g = StateGraph(GraphState)
    for name, fn in nodes.items():
        g.add_node(name, fn)

    g.add_edge(START, STARTUP_SEARCH)
    g.add_conditional_edges(STARTUP_SEARCH, route_after_search, [CANDIDATE_SELECT, REPORT_WRITER])
    g.add_edge(CANDIDATE_SELECT, ANALYSIS_SEQUENCE[0])
    for a, b in zip(ANALYSIS_SEQUENCE, ANALYSIS_SEQUENCE[1:]):
        g.add_edge(a, b)
    g.add_edge(ANALYSIS_SEQUENCE[-1], INVESTMENT_DECISION)
    g.add_conditional_edges(INVESTMENT_DECISION, route_after_decision, [CANDIDATE_SELECT, REPORT_WRITER])
    g.add_edge(REPORT_WRITER, END)

    return g.compile()
