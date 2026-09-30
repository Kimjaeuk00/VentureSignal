from core.config import INVEST_THRESHOLD, SCORE_MAX, SCORECARD_WEIGHTS
from core.state import ScorecardResult


def build_scorecard(scores: dict[str, float]) -> ScorecardResult:
    """항목별 0~SCORE_MAX 점수 → 가중합을 0~100 으로 환산한 total_score 포함 ScorecardResult."""
    missing = SCORECARD_WEIGHTS.keys() - scores.keys()
    if missing:
        raise ValueError(f"누락된 Scorecard 항목: {sorted(missing)}")
    for key, value in scores.items():
        if not 0 <= value <= SCORE_MAX:
            raise ValueError(f"{key} 점수 범위 오류: {value}")

    weighted = sum(scores[k] * w for k, w in SCORECARD_WEIGHTS.items())
    total = round(weighted / SCORE_MAX * 100, 1)
    return {**{k: float(scores[k]) for k in SCORECARD_WEIGHTS}, "total_score": total}


def decide(scorecard: ScorecardResult) -> str:
    return "INVEST" if scorecard["total_score"] >= INVEST_THRESHOLD else "HOLD"
