import pytest

from agents.investment_decision.scoring import build_scorecard, decide

KEYS = ("technology", "competition", "market", "team", "traction", "deal_terms")


def test_full_score_is_100():
    sc = build_scorecard({k: 5 for k in KEYS})
    assert sc["total_score"] == 100.0
    assert decide(sc) == "INVEST"


def test_weights_applied():
    # 기술(35%)만 5점 → 35점
    sc = build_scorecard({k: (5 if k == "technology" else 0) for k in KEYS})
    assert sc["total_score"] == 35.0
    assert decide(sc) == "HOLD"


def test_missing_or_out_of_range_rejected():
    with pytest.raises(ValueError):
        build_scorecard({"technology": 5})
    with pytest.raises(ValueError):
        build_scorecard({k: 6 for k in KEYS})
