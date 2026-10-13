"""품질 평가: 인용 누락·우열 단정·출처 편중을 잡고 정상 보고서는 통과시킨다 (LLM 호출 없음, Judge 는 대체).

실행: pytest agents/quality_check/tests
"""

import pytest

from agents.quality_check import node
from agents.quality_check.checks import check_bias_control, check_groundedness, check_neutrality, is_report
from agents.quality_check.schemas import JudgeIssue, JudgeOutput
from core.state import initial_state, make_analysis

GOOD = """# 투자 검토 보고서

## 1. SUMMARY

Alpha 는 보류이다. 근거가 일부 부족하다.

| 항목 | 점수 | 근거 | 출처 |
|---|---|---|---|
| 기술력 | 3 | 공개된 사양이 있다 | [1, 2] |
| 시장성 | 3 | 수요 전망이 있다 | [3] |
| 경쟁 우위 | 2 | 비교 자료가 부족하다 | [4] |

## 8. REFERENCE

[1] A(접속 2026-10-13). *제목*. a.com, https://a.com
[2] B(접속 2026-10-13). *제목*. b.com, https://b.com
[3] C(접속 2026-10-13). *제목*. c.com, https://c.com
[4] D(접속 2026-10-13). *제목*. d.com, https://d.com
"""


def _state(report, with_risks=True):
    state = initial_state("q")
    state["report"] = report
    state["candidates"] = [
        {"company_id": "C1", "company_name": "Alpha", "description": "", "domain": "", "retrieval_score": 0.0},
        {"company_id": "C2", "company_name": "Beta", "description": "", "domain": "", "retrieval_score": 0.0},
    ]
    state["evaluations"] = {
        "C1": {"technology": make_analysis(strengths=["강점"], risks=["위험"] if with_risks else [])},
        "C2": {},
    }
    return state


@pytest.fixture(autouse=True)
def no_judge(monkeypatch):
    """기본은 Judge 가 아무것도 지적하지 않는다. LLM 을 부르지 않는다."""
    monkeypatch.setattr(node, "judge", lambda state, report: {k: [] for k in node.ITEM_LABELS})


def test_good_report_passes():
    update = node.run(_state(GOOD))
    verdict = update["quality_verdict"]
    assert verdict["passed"], verdict["feedback"]
    assert update["quality_attempts"] == 1


def test_missing_citations_fail_groundedness():
    report = GOOD.replace("[1, 2]", "").replace("[3]", "").replace("[4]", "")
    verdict = node.run(_state(report))["quality_verdict"]
    assert not verdict["groundedness"] and not verdict["passed"]
    assert any("인용이 없다" in f for f in verdict["feedback"])


def test_dangling_citation_fails_groundedness():
    verdict = node.run(_state(GOOD.replace("| [3] |", "| [9] |")))["quality_verdict"]
    assert not verdict["groundedness"]
    assert any("[9]" in f for f in verdict["feedback"])


def test_ranking_sentence_between_candidates_fails_neutrality():
    report = GOOD.replace("근거가 일부 부족하다.", "Alpha 는 Beta 보다 우수하다.")
    verdict = node.run(_state(report))["quality_verdict"]
    assert not verdict["neutrality"] and not verdict["passed"]


def test_promotional_phrase_fails_neutrality():
    verdict = node.run(_state(GOOD.replace("보류이다.", "보류이다. 그래도 강력 추천한다.")))["quality_verdict"]
    assert not verdict["neutrality"]


def test_invest_or_hold_wording_alone_is_allowed():
    assert check_neutrality(GOOD.replace("보류이다.", "투자 검토로 판단한다."), ["Alpha", "Beta"]) == []


def test_single_source_dominance_fails_bias_control():
    report = GOOD.replace("[1, 2]", "[1]").replace("[3]", "[1]").replace("[4]", "[1]")
    verdict = node.run(_state(report))["quality_verdict"]
    assert not verdict["bias_control"] and not verdict["passed"]


def test_strengths_without_any_risk_fails_bias_control():
    verdict = node.run(_state(GOOD, with_risks=False))["quality_verdict"]
    assert not verdict["bias_control"]


def test_judge_issue_with_quote_in_report_fails_that_item(monkeypatch):
    class Fake:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return JudgeOutput(issues=[JudgeIssue(item="neutrality", quote="공개된 사양이 있다", reason="과장")])

    monkeypatch.undo()  # 기본 Judge 대체를 풀고 실제 judge() 에 가짜 LLM 을 꽂는다
    monkeypatch.setattr(node, "get_llm", lambda: Fake())
    verdict = node.run(_state(GOOD))["quality_verdict"]
    assert not verdict["neutrality"] and not verdict["passed"]


def test_judge_issue_with_quote_not_in_report_is_ignored(monkeypatch):
    class Fake:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            return JudgeOutput(issues=[JudgeIssue(item="neutrality", quote="보고서에 없는 문장", reason="추측")])

    monkeypatch.undo()
    monkeypatch.setattr(node, "get_llm", lambda: Fake())
    assert node.run(_state(GOOD))["quality_verdict"]["passed"]


def test_llm_failure_falls_back_to_deterministic_only(monkeypatch):
    monkeypatch.undo()

    def boom():
        raise RuntimeError("llm down")

    monkeypatch.setattr(node, "get_llm", boom)
    assert node.run(_state(GOOD))["quality_verdict"]["passed"]
    assert not node.run(_state(GOOD.replace("| [3] |", "| [9] |")))["quality_verdict"]["passed"]


def test_non_report_text_is_skipped_and_passes():
    for text in ["", "평가된 후보가 없어 보고서를 작성하지 않았습니다.", "보고서를 작성하지 못했습니다 (ValueError)."]:
        assert not is_report(text)
        verdict = node.run(_state(text))["quality_verdict"]
        assert verdict["passed"]


def test_attempts_increment():
    state = _state(GOOD)
    state["quality_attempts"] = 1
    assert node.run(state)["quality_attempts"] == 2


def test_check_functions_return_empty_for_good_report():
    assert check_groundedness(GOOD) == []
    assert check_bias_control(GOOD, {}) == []


# --------------------------------------------------------------------------- Judge 오탐 방지 (groundedness 는 수치로 검증한다)


def _fake_judge(monkeypatch, issues):
    class Fake:
        def with_structured_output(self, schema):
            return self

        def invoke(self, messages):
            self.messages = messages
            return JudgeOutput(issues=issues)

    fake = Fake()
    monkeypatch.undo()  # 기본 Judge 대체를 풀고 실제 judge() 에 가짜 LLM 을 꽂는다
    monkeypatch.setattr(node, "get_llm", lambda: fake)
    return fake


def _state_with_summary(report, summary):
    state = _state(report)
    state["evaluations"]["C1"]["technology"] = make_analysis(summary=summary, strengths=["강점"], risks=["위험"])
    return state


def test_groundedness_issue_is_dropped_when_numbers_are_in_the_analysis(monkeypatch):
    # 오탐: Judge 가 출처 본문을 못 보고 "출처가 뒷받침하지 않는다"고 지적했지만, 문장의 수치는 분석 결과에 있다
    report = GOOD.replace("공개된 사양이 있다", "보도된 PO 77건, 1,300만 달러 이상")
    _fake_judge(monkeypatch, [JudgeIssue(item="groundedness", quote="보도된 PO 77건, 1,300만 달러 이상", reason="출처 확인 불가")])
    verdict = node.run(_state_with_summary(report, "PO 77건과 1300만 달러 이상이 보도됐다"))["quality_verdict"]
    assert verdict["groundedness"] and verdict["passed"]


def test_groundedness_issue_is_kept_when_a_number_is_not_in_the_analysis(monkeypatch):
    # 진짜 문제: 문장의 수치(88건)가 분석 결과 어디에도 없다
    report = GOOD.replace("공개된 사양이 있다", "보도된 PO 88건")
    _fake_judge(monkeypatch, [JudgeIssue(item="groundedness", quote="보도된 PO 88건", reason="근거 없는 수치")])
    verdict = node.run(_state_with_summary(report, "PO 77건이 보도됐다"))["quality_verdict"]
    assert not verdict["groundedness"]
    assert any("88" in f for f in verdict["feedback"])


def test_groundedness_issue_with_only_citation_numbers_is_dropped(monkeypatch):
    _fake_judge(monkeypatch, [JudgeIssue(item="groundedness", quote="공개된 사양이 있다 | [1, 2]", reason="출처가 부족")])
    report = GOOD.replace("공개된 사양이 있다 | [1, 2]", "공개된 사양이 있다 | [1, 2]")
    assert node.run(_state_with_summary(report, "요약"))["quality_verdict"]["passed"]


def test_judge_receives_analysis_results_to_compare_against(monkeypatch):
    fake = _fake_judge(monkeypatch, [])
    node.run(_state_with_summary(GOOD, "DX-M1은 25 TOPS 이다"))
    human = fake.messages[1][1]
    assert "[분석 결과]" in human and "25 TOPS" in human


def test_numbers_helper_ignores_thousands_separator():
    assert node._numbers("1,300만 달러, 77건, 2.5W") == {"1300", "77", "2.5"}
