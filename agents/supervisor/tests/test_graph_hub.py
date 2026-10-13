"""허브 구조 그래프: 모든 노드를 스텁으로 바꿔 라우팅·재작업·실패 처리·종료를 확인한다 (LLM·웹검색 호출 없음).

실행: pytest agents/supervisor/tests
"""

from graph.builder import (
    COMPETITOR_COMPARE,
    FOUNDER_TRACTION,
    INVESTMENT_DECISION,
    MARKET_EVAL,
    REPORT_WRITER,
    STARTUP_SEARCH,
    SUPERVISOR,
    TECH_SUMMARY,
    WORKERS,
    QUALITY_CHECK,
    build_graph,
)
from core.state import evaluation_update, initial_state, make_analysis

CONFIG = {"recursion_limit": 100}


def _good(**details):
    return make_analysis(summary="s", evidence=["S001"], details=details)


def _candidates(n):
    return [
        {"company_id": f"C{i}", "company_name": f"Co{i}", "description": "", "domain": "", "retrieval_score": 0.0}
        for i in range(n)
    ]


class Harness:
    """스텁 노드를 만들고 호출 순서를 기록한다."""

    def __init__(self, candidates=3, invest_on=None, bad_competition_for=(), fail_agent=None):
        self.calls: list[tuple[str, str]] = []
        self.n = candidates
        self.invest_on = invest_on
        self.bad_competition_for = set(bad_competition_for)
        self.fail_agent = fail_agent  # (agent, 실패 횟수)
        self.failures = 0

    def _record(self, name, state):
        self.calls.append((name, state["current_candidate"]))

    def nodes(self):
        def search(state):
            return {"candidates": _candidates(self.n)}

        def select(state):
            self._record("candidate_select", state)
            i = state["candidate_index"]
            return {"current_candidate": state["candidates"][i]["company_id"], "candidate_index": i + 1}

        def worker(name, **fields):
            def run(state):
                self._record(name, state)
                if self.fail_agent and self.fail_agent[0] == name and self.failures < self.fail_agent[1]:
                    self.failures += 1
                    raise RuntimeError("boom")
                cid = state["current_candidate"]
                made = {key: (value(cid) if callable(value) else value) for key, value in fields.items()}
                return evaluation_update(cid, **made)

            return run

        def competition(cid):
            return make_analysis(summary="증거 없음") if cid in self.bad_competition_for else _good()

        def decide(state):
            self._record("investment_decision", state)
            cid = state["current_candidate"]
            decision = "INVEST" if cid == self.invest_on else "HOLD"
            update = evaluation_update(cid, decision=decision)
            if decision == "INVEST":
                update["selected_candidate"] = cid
            return update

        def report(state):
            self._record("report_writer", state)
            return {"report": "보고서"}

        return {
            STARTUP_SEARCH: search,
            "candidate_select": select,
            TECH_SUMMARY: worker(TECH_SUMMARY, technology=lambda c: _good()),
            MARKET_EVAL: worker(MARKET_EVAL, market=lambda c: _good()),
            COMPETITOR_COMPARE: worker(COMPETITOR_COMPARE, competition=competition),
            FOUNDER_TRACTION: worker(FOUNDER_TRACTION, team=lambda c: _good(), traction=lambda c: _good()),
            INVESTMENT_DECISION: decide,
            REPORT_WRITER: report,
        }

    def run(self, **state_over):
        state = initial_state("q")
        state.update(state_over)
        return build_graph(self.nodes()).invoke(state, config=CONFIG)

    def count(self, name, cid=None):
        return sum(1 for n, c in self.calls if n == name and (cid is None or c == cid))


def test_report_writer_and_quality_check_are_workers_too():
    """보고서 작성과 품질 평가도 하위 노드다: 서로 직접 연결하지 않고 Supervisor 를 거친다."""
    assert REPORT_WRITER in WORKERS and QUALITY_CHECK in WORKERS
    edges = {(e.source, e.target) for e in build_graph().get_graph().edges}
    assert (REPORT_WRITER, QUALITY_CHECK) not in edges
    assert (REPORT_WRITER, SUPERVISOR) in edges and (QUALITY_CHECK, SUPERVISOR) in edges
    assert (QUALITY_CHECK, "__end__") not in edges  # 종료도 Supervisor 가 정한다
    assert (SUPERVISOR, "__end__") in edges


def test_workers_only_return_to_supervisor():
    graph = build_graph().get_graph()
    for edge in graph.edges:
        if edge.source in WORKERS:
            assert edge.target == SUPERVISOR
        if edge.source == SUPERVISOR:
            assert edge.conditional  # 모든 분기는 Supervisor 의 add_conditional_edges 를 거친다
    assert not any(e.source in WORKERS and e.target in WORKERS for e in graph.edges)


def test_invest_stops_loop_and_skips_remaining_candidates():
    h = Harness(candidates=3, invest_on="C1")
    result = h.run()

    assert result["selected_candidate"] == "C1"
    assert set(result["evaluations"]) == {"C0", "C1"}  # C2 는 평가하지 않음
    assert h.count("report_writer") == 1


def test_all_hold_exhausts_candidates_then_reports():
    h = Harness(candidates=3)
    result = h.run()

    assert result["selected_candidate"] is None
    assert set(result["evaluations"]) == {"C0", "C1", "C2"}
    assert all(ev["decision"] == "HOLD" for ev in result["evaluations"].values())
    assert result["report"] == "보고서"


def test_no_candidates_goes_straight_to_report():
    h = Harness(candidates=0)
    result = h.run()

    assert result["evaluations"] == {}
    assert h.calls == [("report_writer", None)]


def test_insufficient_evidence_triggers_one_rework_then_proceeds():
    h = Harness(candidates=1, bad_competition_for={"C0"})
    result = h.run()

    assert h.count(COMPETITOR_COMPARE, "C0") == 2  # 첫 실행 + 재작업 1회
    assert h.count(TECH_SUMMARY, "C0") == 1  # 충분한 에이전트는 다시 시키지 않는다
    assert h.count("investment_decision", "C0") == 1  # 상한 후에는 미확인으로 두고 진행
    assert result["rework_count"]["C0:competitor_compare"] == 1
    assert result["node_status"]["C0:competitor_compare"] == "excluded"
    assert result["report"] == "보고서"


def test_worker_exception_is_retried_then_graph_continues():
    h = Harness(candidates=1, fail_agent=(MARKET_EVAL, 1))
    result = h.run()

    assert h.count(MARKET_EVAL, "C0") == 2  # 실패 1회 + 재시도 성공
    assert "market" in result["evaluations"]["C0"]
    assert "boom" in result["last_error"]
    assert result["report"] == "보고서"


def test_worker_failing_forever_is_excluded_and_graph_terminates():
    h = Harness(candidates=1, fail_agent=(MARKET_EVAL, 99))
    result = h.run()

    assert h.count(MARKET_EVAL, "C0") == 2  # 첫 실행 + 재시도 상한
    assert "market" not in result["evaluations"]["C0"]
    assert result["node_status"]["C0:market_eval"] == "excluded"
    assert result["report"] == "보고서"


def test_max_steps_forces_report_and_terminates(caplog):
    h = Harness(candidates=3)
    with caplog.at_level("INFO", logger="venturesignal.decision"):
        result = h.run(max_steps=4)

    assert result["report"] == "보고서"
    # max_steps 를 넘긴 진입(5)에서 강제 이동하고, 보고서 단계에서 Supervisor 가 두 번 더 진입한다 (작성 후, 평가 후)
    assert result["step_count"] == 4 + 1 + 2
    assert "max_steps" in caplog.text
    assert h.count("report_writer") == 1


def test_trace_id_is_stable_across_run():
    state = initial_state("q")
    h = Harness(candidates=1)
    result = build_graph(h.nodes()).invoke(state, config=CONFIG)
    assert result["trace_id"] == state["trace_id"]


# --------------------------------------------------------------------------- 품질 평가 루프


def _quality_stub(pass_on_attempt):
    """pass_on_attempt 번째 평가에서 통과하는 가짜 quality_check. None 이면 계속 미달."""

    def run(state):
        attempt = state["quality_attempts"] + 1
        passed = pass_on_attempt is not None and attempt >= pass_on_attempt
        verdict = {
            "groundedness": passed,
            "neutrality": True,
            "bias_control": True,
            "passed": passed,
            "feedback": [] if passed else ["[Groundedness] 인용 누락"],
        }
        return {"quality_verdict": verdict, "quality_attempts": attempt}

    return run


def _run_with_quality(pass_on_attempt):
    h = Harness(candidates=1)
    nodes = {**h.nodes(), "quality_check": _quality_stub(pass_on_attempt)}
    state = initial_state("q")
    result = build_graph(nodes).invoke(state, config=CONFIG)
    return h, result


def test_quality_pass_ends_without_rewrite():
    h, result = _run_with_quality(pass_on_attempt=1)
    assert h.count("report_writer") == 1
    assert result["quality_verdict"]["passed"]


def test_quality_fail_rewrites_report_through_supervisor_then_passes():
    h, result = _run_with_quality(pass_on_attempt=2)
    assert h.count("report_writer") == 2  # 첫 작성 + 재작성 1회
    assert result["quality_attempts"] == 2
    assert result["quality_verdict"]["passed"]
    assert h.count(INVESTMENT_DECISION) == 1  # 재작성에서 분석을 다시 하지 않는다


def test_quality_always_failing_stops_at_revision_limit():
    from agents.quality_check.policy import MAX_REVISIONS

    h, result = _run_with_quality(pass_on_attempt=None)
    assert h.count("report_writer") == 1 + MAX_REVISIONS  # 무한 루프 없이 상한에서 끝난다
    assert result["quality_attempts"] == 1 + MAX_REVISIONS
    assert not result["quality_verdict"]["passed"]
    assert result["report"] == "보고서"  # 미달이어도 보고서는 남는다


def test_worst_case_finishes_within_computed_recursion_limit():
    """후보 3곳 모두 근거 부족 재작업 + 품질 평가 계속 미달이어도 run_config 의 recursion_limit 안에서 끝난다."""
    from agents.supervisor.audit import run_config

    h = Harness(candidates=3, bad_competition_for={"C0", "C1", "C2"})
    nodes = {**h.nodes(), "quality_check": _quality_stub(None)}
    state = initial_state("q")
    result = build_graph(nodes).invoke(state, config=run_config(state))  # GraphRecursionError 가 나면 실패

    assert result["report"] == "보고서"
    assert result["step_count"] <= state["max_steps"] + 1 + 1  # max_steps 를 넘기면 강제 이동하므로 크게 늘지 않는다
