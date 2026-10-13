"""Supervisor 라우팅: State 만 보고 다음 노드를 정한다 (LLM·웹검색 호출 없음).

실행: pytest agents/supervisor/tests
"""

from agents.supervisor import policy
from agents.supervisor.node import route, run
from core.state import agent_key, initial_state, make_analysis


def _candidates(n):
    return [
        {"company_id": f"C{i}", "company_name": f"Co{i}", "description": "", "domain": "", "retrieval_score": 0.0}
        for i in range(n)
    ]


def _good(**details):
    return make_analysis(summary="s", evidence=["S001"], details=details)


def _all_analyses(**over):
    base = {
        "technology": _good(),
        "market": _good(),
        "competition": _good(),
        "team": _good(),
        "traction": _good(),
    }
    return {**base, **over}


def _state(candidates=1, cid="C0", index=1, evaluations=None, **over):
    state = initial_state("q")
    state["candidates"] = _candidates(candidates)
    state["current_candidate"] = cid
    state["candidate_index"] = index
    state["evaluations"] = {cid: evaluations} if evaluations is not None and cid else {}
    state.update(over)
    return state


def _apply(state, update):
    """노드 반환값을 State 에 반영한다 (reducer 가 하는 일을 흉내)."""
    for key, value in update.items():
        if key in ("node_status", "rework_count"):
            state[key] = {**state[key], **value}
        else:
            state[key] = value
    return state


def test_node_names_match_graph_builder():
    from graph import builder

    assert policy.CANDIDATE_SELECT == builder.CANDIDATE_SELECT
    assert policy.TECH_SUMMARY == builder.TECH_SUMMARY
    assert policy.MARKET_EVAL == builder.MARKET_EVAL
    assert policy.COMPETITOR_COMPARE == builder.COMPETITOR_COMPARE
    assert policy.FOUNDER_TRACTION == builder.FOUNDER_TRACTION
    assert policy.INVESTMENT_DECISION == builder.INVESTMENT_DECISION
    assert policy.REPORT_WRITER == builder.REPORT_WRITER
    assert policy.QUALITY_CHECK == builder.QUALITY_CHECK
    assert policy.END_TARGET == builder.END


def test_no_candidates_goes_to_report():
    update = run(_state(candidates=0, cid=None, index=0))
    assert update["next_agent"] == policy.REPORT_WRITER


def test_first_step_selects_candidate():
    update = run(_state(cid=None, index=0, evaluations=None))
    assert update["next_agent"] == policy.CANDIDATE_SELECT


def test_runs_unexecuted_analysis_agent_then_decision():
    state = _state(evaluations={"technology": _good()})
    assert run(state)["next_agent"] in policy.ANALYSIS_AGENTS

    state = _state(evaluations=_all_analyses())
    update = run(state)
    assert update["next_agent"] == policy.INVESTMENT_DECISION
    assert update["node_status"][agent_key("C0", policy.TECH_SUMMARY)] == "done"


def test_insufficient_evidence_requests_rework():
    bad = make_analysis(summary="s", details={"status": "insufficient_evidence", "reason": "r"})
    update = run(_state(evaluations=_all_analyses(market=bad)))

    key = agent_key("C0", policy.MARKET_EVAL)
    assert update["next_agent"] == policy.MARKET_EVAL
    assert update["rework_count"][key] == 1
    assert update["node_status"][key] == "needs_rework"
    assert "근거 부족" in update["last_decision"]["reason"]


def test_empty_evidence_is_insufficient():
    update = run(_state(evaluations=_all_analyses(competition=make_analysis(summary="s"))))
    assert update["next_agent"] == policy.COMPETITOR_COMPARE


def test_rework_limit_proceeds_without_further_rework():
    bad = make_analysis(summary="s")
    key = agent_key("C0", policy.COMPETITOR_COMPARE)
    state = _state(evaluations=_all_analyses(competition=bad), rework_count={key: policy.MAX_REWORK})

    update = run(state)

    assert update["next_agent"] == policy.INVESTMENT_DECISION
    assert update["node_status"][key] == "excluded"
    assert "rework_count" not in update


def test_max_steps_forces_report():
    state = _state(evaluations={}, step_count=3, max_steps=3)
    update = run(state)
    assert update["next_agent"] == policy.REPORT_WRITER
    assert "max_steps" in update["last_decision"]["reason"]


def test_invest_goes_to_report_even_if_candidates_remain():
    state = _state(candidates=3, index=1, evaluations={**_all_analyses(), "decision": "INVEST"})
    assert run(state)["next_agent"] == policy.REPORT_WRITER


def test_hold_with_remaining_candidates_selects_next():
    state = _state(candidates=3, index=1, evaluations={**_all_analyses(), "decision": "HOLD"})
    assert run(state)["next_agent"] == policy.CANDIDATE_SELECT


def test_hold_when_exhausted_goes_to_report():
    state = _state(candidates=3, index=3, cid="C2", evaluations={**_all_analyses(), "decision": "HOLD"})
    assert run(state)["next_agent"] == policy.REPORT_WRITER


def test_returns_only_control_fields_and_increments_step():
    state = _state(evaluations={})
    update = run(state)
    assert set(update) <= {"step_count", "next_agent", "last_decision", "node_status", "rework_count"}
    assert update["step_count"] == state["step_count"] + 1
    assert route(_apply(state, update)) == update["next_agent"]


def test_decision_is_logged_with_trace_id(caplog):
    state = _state(evaluations={})
    with caplog.at_level("INFO", logger="venturesignal.decision"):
        run(state)
    assert state["trace_id"] in caplog.text


def test_simulated_loop_terminates_and_reworks_once():
    """에이전트를 흉내 내며 Supervisor 를 반복 호출해 종료·재작업 횟수를 확인한다."""
    state = _state(candidates=1, cid=None, index=0, evaluations=None)
    calls: dict[str, int] = {}
    fields = {
        policy.TECH_SUMMARY: {"technology": _good()},
        policy.MARKET_EVAL: {"market": _good()},
        policy.COMPETITOR_COMPARE: {"competition": make_analysis(summary="증거 없음")},  # 항상 부족
        policy.FOUNDER_TRACTION: {"team": _good(), "traction": _good()},
    }

    for _ in range(state["max_steps"]):
        _apply(state, run(state))
        target = route(state)
        calls[target] = calls.get(target, 0) + 1
        if target == policy.REPORT_WRITER:
            break
        if target == policy.CANDIDATE_SELECT:
            state["current_candidate"], state["candidate_index"] = "C0", 1
            state["evaluations"] = {"C0": {}}
        elif target == policy.INVESTMENT_DECISION:
            state["evaluations"]["C0"]["decision"] = "HOLD"
        else:
            state["evaluations"]["C0"].update(fields[target])

    assert route(state) == policy.REPORT_WRITER
    assert calls[policy.COMPETITOR_COMPARE] == 1 + policy.MAX_REWORK  # 첫 실행 + 재작업
    assert calls[policy.TECH_SUMMARY] == 1


# --------------------------------------------------------------------------- 보고서 단계 (report_writer·quality_check 도 Supervisor 를 거친다)

WRITER_KEY = policy.REPORT_STAGE_KEYS[policy.REPORT_WRITER]
CHECK_KEY = policy.REPORT_STAGE_KEYS[policy.QUALITY_CHECK]


def _verdict(passed, feedback=()):
    return {"groundedness": passed, "neutrality": True, "bias_control": True, "passed": passed, "feedback": list(feedback)}


def test_dispatching_report_writer_marks_it_pending():
    update = run(_state(candidates=0, cid=None, index=0))
    assert update["next_agent"] == policy.REPORT_WRITER
    assert update["node_status"][WRITER_KEY] == "pending"


def test_report_written_goes_to_quality_check_via_supervisor():
    state = _state(report="보고서", node_status={WRITER_KEY: "pending"})
    update = run(state)
    assert update["next_agent"] == policy.QUALITY_CHECK
    assert update["node_status"][WRITER_KEY] == "done" and update["node_status"][CHECK_KEY] == "pending"


def test_quality_passed_ends():
    state = _state(report="보고서", node_status={CHECK_KEY: "pending"}, quality_verdict=_verdict(True), quality_attempts=1)
    update = run(state)
    assert update["next_agent"] == policy.END_TARGET
    assert update["node_status"][CHECK_KEY] == "done"


def test_quality_failed_rewrites_when_under_limit():
    from agents.quality_check.policy import MAX_REVISIONS

    state = _state(report="보고서", node_status={CHECK_KEY: "pending"},
                   quality_verdict=_verdict(False, ["[Groundedness] 인용 누락"]), quality_attempts=MAX_REVISIONS)
    update = run(state)
    assert update["next_agent"] == policy.REPORT_WRITER
    assert update["node_status"][WRITER_KEY] == "pending"  # 다시 보냈으니 돌아올 때까지 pending
    assert "인용 누락" in update["last_decision"]["reason"]


def test_quality_failed_at_revision_limit_ends():
    from agents.quality_check.policy import MAX_REVISIONS

    state = _state(report="보고서", node_status={CHECK_KEY: "pending"},
                   quality_verdict=_verdict(False, ["x"]), quality_attempts=MAX_REVISIONS + 1)
    assert run(state)["next_agent"] == policy.END_TARGET


def test_missing_verdict_after_quality_check_ends_instead_of_looping():
    state = _state(report="보고서", node_status={CHECK_KEY: "pending"}, quality_verdict=None)
    assert run(state)["next_agent"] == policy.END_TARGET


def test_max_steps_does_not_force_report_again_once_report_stage_started():
    # 보고서 단계가 시작된 뒤에는 상한을 넘겨도 단계 진행(평가 → 종료)을 따른다
    state = _state(report="보고서", node_status={WRITER_KEY: "done", CHECK_KEY: "pending"},
                   quality_verdict=_verdict(True), quality_attempts=1, step_count=99, max_steps=3)
    assert run(state)["next_agent"] == policy.END_TARGET
