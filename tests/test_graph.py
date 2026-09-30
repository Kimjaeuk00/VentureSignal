from core.state import evaluation_update, initial_state
from graph.builder import INVESTMENT_DECISION, STARTUP_SEARCH, build_graph

CONFIG = {"recursion_limit": 50}


def _candidates(n):
    return [
        {"company_id": f"C{i}", "company_name": f"Co{i}", "description": "", "domain": "", "retrieval_score": 0.0}
        for i in range(n)
    ]


def _search(n):
    return lambda state: {"candidates": _candidates(n)}


def _decide_invest_on(target):
    def run(state):
        cid = state["current_candidate"]
        decision = "INVEST" if cid == target else "HOLD"
        update = evaluation_update(cid, decision=decision)
        if decision == "INVEST":
            update["selected_candidate"] = cid
        return update

    return run


def test_stub_graph_evaluates_all_candidates_then_reports():
    result = build_graph().invoke(initial_state("q"), config=CONFIG)

    assert len(result["evaluations"]) == len(result["candidates"])
    for ev in result["evaluations"].values():
        # 모든 분석 노드 결과가 병합되어 있어야 한다
        assert {"technology", "market", "competition", "team", "traction", "deal_terms", "scorecard"} <= ev.keys()
        assert ev["decision"] == "HOLD"
    assert result["selected_candidate"] is None
    assert result["report"]


def test_invest_stops_loop():
    graph = build_graph({STARTUP_SEARCH: _search(3), INVESTMENT_DECISION: _decide_invest_on("C1")})
    result = graph.invoke(initial_state("q"), config=CONFIG)

    assert result["selected_candidate"] == "C1"
    assert set(result["evaluations"]) == {"C0", "C1"}  # C2 는 평가하지 않음


def test_no_candidates_goes_straight_to_report():
    result = build_graph({STARTUP_SEARCH: _search(0)}).invoke(initial_state("q"), config=CONFIG)
    assert result["evaluations"] == {}
    assert result["report"]
