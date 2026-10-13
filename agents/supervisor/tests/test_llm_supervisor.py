"""LLM Supervisor: 다음 에이전트 선택·보고 평가는 LLM 이 하고, 코드는 종료·안전 불변 조건만 검사한다 (가짜 LLM, 실제 호출 없음).

실행: pytest agents/supervisor/tests
"""

from typing import get_args

from agents.supervisor import llm_policy, policy
from agents.supervisor.node import run
from agents.supervisor.schemas import AgentReview, NextAgent, SupervisorDecision
from core.state import agent_key, initial_state, make_analysis


def _good():
    return make_analysis(summary="요약", strengths=["강점"], risks=["위험"], evidence=["S001"])


def _state(evaluations=None, last=None, **over):
    state = initial_state("q")
    state["candidates"] = [
        {"company_id": "C0", "company_name": "Co0", "description": "설명", "domain": "NPU", "retrieval_score": 0.0},
        {"company_id": "C1", "company_name": "Co1", "description": "", "domain": "", "retrieval_score": 0.0},
    ]
    state["current_candidate"], state["candidate_index"] = "C0", 1
    state["evaluations"] = {"C0": evaluations or {}}
    state["next_agent"] = last
    state.update(over)
    return state


def _d(next_agent, review=None, reason="사유"):
    return SupervisorDecision(review=review, next_agent=next_agent, reason=reason)


def _all_done():
    return {"technology": _good(), "market": _good(), "competition": _good(), "team": _good(), "traction": _good()}


def test_schema_choices_match_node_names():
    assert set(get_args(NextAgent)) == {*policy.ANALYSIS_AGENTS, policy.INVESTMENT_DECISION}


def test_llm_chooses_the_order_not_the_code(llm_on):
    fake = llm_on([_d("competitor_compare")])
    update = run(_state())
    assert update["next_agent"] == "competitor_compare"  # 코드의 기본 순서(tech_summary 먼저)와 다르다
    assert len(fake.calls) == 1


def test_prompt_carries_agent_status_and_rework_limit(llm_on):
    fake = llm_on([_d("market_eval")])
    run(_state(evaluations={"technology": _good()}, last="tech_summary"))
    system, human = fake.calls[0][0][1], fake.calls[0][1][1]
    assert f"{policy.MAX_REWORK}회까지" in system
    assert "tech_summary: 완료" in human and "market_eval: 아직 실행하지 않음" in human
    assert "[방금 돌아온 에이전트] tech_summary" in human


def test_insufficient_review_with_rework_counts_and_marks_needs_rework(llm_on):
    review = AgentReview(agent="tech_summary", sufficient=False, missing=["양산 고객"])
    llm_on([_d("tech_summary", review, "다시 조사")])
    update = run(_state(evaluations={"technology": _good()}, last="tech_summary"))

    key = agent_key("C0", "tech_summary")
    assert update["next_agent"] == "tech_summary"
    assert update["rework_count"][key] == 1 and update["node_status"][key] == "needs_rework"
    assert "부족: 양산 고객" in update["last_decision"]["reason"]


def test_insufficient_review_without_rework_marks_excluded(llm_on):
    review = AgentReview(agent="tech_summary", sufficient=False, missing=["x"])
    llm_on([_d("market_eval", review)])
    update = run(_state(evaluations={"technology": _good()}, last="tech_summary"))
    assert update["node_status"][agent_key("C0", "tech_summary")] == "excluded"
    assert "rework_count" not in update


def test_sufficient_review_marks_done(llm_on):
    llm_on([_d("market_eval", AgentReview(agent="tech_summary", sufficient=True))])
    update = run(_state(evaluations={"technology": _good()}, last="tech_summary"))
    assert update["node_status"][agent_key("C0", "tech_summary")] == "done"


def test_investment_decision_before_analysis_is_rejected_then_asked_again(llm_on):
    fake = llm_on([_d("investment_decision"), _d("tech_summary")])
    update = run(_state())
    assert update["next_agent"] == "tech_summary"
    assert len(fake.calls) == 2 and "이전 응답의 문제" in fake.calls[1][1][1]


def test_rework_over_the_limit_is_rejected(llm_on):
    key = agent_key("C0", "tech_summary")
    fake = llm_on([_d("tech_summary"), _d("market_eval")])
    update = run(_state(evaluations={"technology": _good()}, last="tech_summary", rework_count={key: policy.MAX_REWORK}))
    assert update["next_agent"] == "market_eval"
    assert "재작업 상한" in fake.calls[1][1][1]


def test_excluded_agent_cannot_be_dispatched_again(llm_on):
    key = agent_key("C0", "tech_summary")
    llm_on([_d("tech_summary"), _d("market_eval")])
    update = run(_state(evaluations={"technology": _good()}, node_status={key: "excluded"}))
    assert update["next_agent"] == "market_eval"


def test_investment_decision_allowed_once_all_analyses_exist(llm_on):
    llm_on([_d("investment_decision", AgentReview(agent="founder_traction", sufficient=True))])
    assert run(_state(evaluations=_all_done(), last="founder_traction"))["next_agent"] == "investment_decision"


def test_repeated_violations_fall_back_to_rules(llm_on):
    fake = llm_on([_d("investment_decision"), _d("investment_decision")])
    update = run(_state())
    assert update["next_agent"] in policy.ANALYSIS_AGENTS  # 규칙 기반이 아직 실행하지 않은 에이전트를 고른다
    assert "규칙 기반으로 대신함" in update["last_decision"]["reason"]
    assert len(fake.calls) == llm_policy.MAX_ASK  # LLM 호출은 상한을 넘지 않는다


def test_llm_failure_falls_back_to_rules(llm_on):
    llm_on([RuntimeError("llm down")])
    update = run(_state())
    assert update["next_agent"] in policy.ANALYSIS_AGENTS
    assert "LLM 호출 실패" in update["last_decision"]["reason"]


def test_failed_agent_retry_is_rule_based_without_llm(llm_on):
    fake = llm_on([])
    key = agent_key("C0", "market_eval")
    update = run(_state(evaluations={"technology": _good()}, node_status={key: "failed"}, last_error="boom"))
    assert update["next_agent"] == "market_eval" and "재시도" in update["last_decision"]["reason"]
    assert fake.calls == []


def test_after_a_decision_transitions_are_rule_based_without_llm(llm_on):
    fake = llm_on([])
    update = run(_state(evaluations={**_all_done(), "decision": "HOLD"}))
    assert update["next_agent"] == policy.CANDIDATE_SELECT
    assert fake.calls == []


def test_disabled_flag_uses_rules_only(monkeypatch, llm_on):
    fake = llm_on([_d("competitor_compare")])
    monkeypatch.setattr(llm_policy, "ENABLED", False)
    run(_state())
    assert fake.calls == []


def test_whole_graph_follows_llm_chosen_order_and_terminates(llm_on):
    """가짜 LLM 이 '아직 실행하지 않은 마지막 에이전트'를 고르면 기본 순서와 반대로 실행된다. 끝까지 돌고 종료한다."""
    from agents.supervisor.tests.test_graph_hub import CONFIG, Harness
    from graph.builder import build_graph

    order = ["tech_summary", "market_eval", "competitor_compare", "founder_traction"]

    def policy_fn(messages):
        human = messages[1][1]
        pending = [a for a in order if f"- {a}: 아직 실행하지 않음" in human]
        return _d(pending[-1] if pending else "investment_decision")

    llm_on(policy_fn)
    h = Harness(candidates=1)
    result = build_graph(h.nodes()).invoke(initial_state("q"), config=CONFIG)

    ran = [name for name, _ in h.calls if name in order]
    assert ran == list(reversed(order))
    assert result["report"] == "보고서"


def test_failed_rework_is_shown_to_the_llm_as_old_result(llm_on):
    """재작업 실행이 예외로 실패하면(결과는 이전 것 그대로) 그 사실을 LLM 이 알아야 개선된 것으로 오해하지 않는다."""
    fake = llm_on([_d("market_eval")])
    key = agent_key("C0", "tech_summary")
    state = _state(evaluations={"technology": _good()}, last="tech_summary",
                   node_status={key: "failed"}, last_error="tech_summary: ValidationError: boom", rework_count={key: 1})
    run(state)
    human = fake.calls[0][1][1]
    assert "직전 재작업 실행이 실패해" in human and "ValidationError" in human


def test_no_failure_warning_when_nothing_failed(llm_on):
    fake = llm_on([_d("market_eval")])
    run(_state(evaluations={"technology": _good()}, last="tech_summary"))
    assert "직전 재작업 실행이 실패해" not in fake.calls[0][1][1]


def test_rework_request_leaves_the_missing_items_and_queries_for_the_agent(llm_on):
    review = AgentReview(agent="tech_summary", sufficient=False, missing=["양산 고객", "수율"])
    decision = SupervisorDecision(review=review, next_agent="tech_summary", reason="다시 조사",
                                  search_queries=["DX-M1 mass production customers", "DX-M1 수율"])
    llm_on([decision])
    update = run(_state(evaluations={"technology": _good()}, last="tech_summary"))
    assert update["rework_notes"][agent_key("C0", "tech_summary")] == {
        "missing": ["양산 고객", "수율"], "queries": ["DX-M1 mass production customers", "DX-M1 수율"]}


def test_rework_without_queries_still_passes_the_missing_items(llm_on):
    llm_on([_d("tech_summary", AgentReview(agent="tech_summary", sufficient=False, missing=["수율"]))])
    update = run(_state(evaluations={"technology": _good()}, last="tech_summary"))
    assert update["rework_notes"][agent_key("C0", "tech_summary")] == {"missing": ["수율"], "queries": []}


def test_prompt_asks_for_search_queries_on_rework(llm_on):
    fake = llm_on([_d("market_eval")])
    run(_state())
    assert "search_queries" in fake.calls[0][0][1]


def test_no_notes_when_the_next_agent_is_not_a_rework(llm_on):
    llm_on([_d("market_eval", AgentReview(agent="tech_summary", sufficient=False, missing=["x"]))])
    update = run(_state(evaluations={"technology": _good()}, last="tech_summary"))
    assert "rework_notes" not in update
