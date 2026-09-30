from types import SimpleNamespace

import pytest

from agents.founder_traction import node
from agents.founder_traction.agent import FounderPerformanceAgent
from agents.founder_traction.analysis_helpers import (
    build_verified_fact_summary,
    estimate_runway,
)
from agents.founder_traction.schemas import Fact
from agents.founder_traction.web_research import profile_url


def test_node_delegates_to_founder_performance(monkeypatch):
    state = {"current_candidate": "C03"}
    expected = {"evaluations": {"C03": {"team": {}}}}
    monkeypatch.setattr(node, "run_founder_traction", lambda received: expected if received is state else None)

    assert node.run(state) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (
            "https://www.linkedin.com/in/joo-young-kim-7b227955/?isSelfProfile=false",
            "https://www.linkedin.com/in/joo-young-kim-7b227955",
        ),
        ("https://kr.linkedin.com/in/example/", "https://www.linkedin.com/in/example"),
        ("https://www.linkedin.com/company/example", None),
        ("https://example.com/in/example", None),
    ],
)
def test_profile_url_accepts_only_linkedin_person_profiles(raw, expected):
    assert profile_url(raw) == expected


def test_estimate_runway_requires_matching_recent_cash_and_burn():
    facts = [
        SimpleNamespace(
            id="cash-1",
            subject="HyperAccel",
            category="cash",
            attributes={"value": "120", "unit": "KRW_100M", "as_of": "2026-09-01"},
        ),
        SimpleNamespace(
            id="burn-1",
            subject="HyperAccel",
            category="burn",
            attributes={
                "value": "10",
                "unit": "KRW_100M",
                "as_of": "2026-09-01",
                "period": "month",
            },
        ),
    ]

    assert estimate_runway(facts, "HyperAccel", "2026-09-30") == {
        "months": 12.0,
        "as_of": "2026-09-01",
        "formula": "cash / monthly net burn",
        "fact_ids": ["cash-1", "burn-1"],
        "note": "기준일 단순 추정. 이후 현금 변동과 향후 지출 미반영.",
    }


def test_profile_page_is_split_without_dropping_later_sections():
    body = "Experience\n" + ("A" * 40) + "\nEducation\n" + ("B" * 40)
    agent = FounderPerformanceAgent(
        llm=object(),
        page_reader=lambda urls, timeout: {
            "results": [{"url": urls[0], "title": "CEO", "raw_content": body}]
        },
    )
    agent.config = type(agent.config)(snippet_chars=30)

    docs, statuses = agent._pages(["https://www.linkedin.com/in/example"])

    assert "".join(doc["content"] for doc in docs) == body
    assert statuses["https://www.linkedin.com/in/example"] == "extractor_returned_text"
    assert agent.page_metadata["https://www.linkedin.com/in/example"]["chunks"] > 1


def test_fallback_summary_uses_all_target_facts_and_excludes_peers():
    facts = [
        Fact(id="career-1", subject="CEO", category="career", text="경력 사실", evidence=[]),
        Fact(id="education-1", subject="CEO", category="education", text="학력 사실", evidence=[]),
        Fact(id="expertise-1", subject="CEO", category="expertise", text="전문성 사실", evidence=[]),
        Fact(id="execution-1", subject="CEO", category="execution", text="실행 사실", evidence=[]),
        Fact(id="peer-1", subject="Peer CEO", category="career", text="피어 경력", evidence=[]),
    ]

    summary, fact_ids = build_verified_fact_summary(
        "team", facts, {"CEO"}, ["skill", "publication"]
    )

    assert fact_ids == ["career-1", "education-1", "expertise-1", "execution-1"]
    assert all(text in summary for text in ("경력 사실", "학력 사실", "전문성 사실", "실행 사실"))
    assert "피어 경력" not in summary
    assert "명시적으로 확인되는 기술 목록" in summary
    assert "동일인 검증을 통과한 논문 자료" in summary


def test_fallback_summary_removes_only_exact_duplicate_text():
    facts = [
        Fact(id="one", subject="Company", category="customer", text="협력 사실", evidence=[]),
        Fact(id="duplicate", subject="Company", category="commercialization", text="  협력   사실 ", evidence=[]),
        Fact(id="different", subject="Company", category="milestone", text="별도 사실", evidence=[]),
    ]

    summary, fact_ids = build_verified_fact_summary("traction", facts, {"Company"}, [])

    assert summary == "협력 사실 별도 사실"
    assert fact_ids == ["one", "different"]
