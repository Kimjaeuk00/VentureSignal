"""항목별 점수의 가중합과 하한값 비교."""

import math

from core.config import SCORE_MAX, SCORECARD_WEIGHTS
from core.state import ScorecardResult


def build_scorecard(scores: dict[str, float]) -> ScorecardResult:
    expected = set(SCORECARD_WEIGHTS)

    if set(scores) != expected:
        raise ValueError(
            "평가 항목이 일치하지 않습니다. "
            f"필요 항목: {sorted(expected)}"
        )

    normalized = {}

    for key, value in scores.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} 점수는 숫자여야 합니다.")

        value = float(value)

        if not math.isfinite(value) or not 0 <= value <= SCORE_MAX:
            raise ValueError(
                f"{key} 점수는 0~{SCORE_MAX} 범위여야 합니다."
            )

        normalized[key] = value

    weighted = sum(
        normalized[key] * weight
        for key, weight in SCORECARD_WEIGHTS.items()
    )

    total = round(weighted / SCORE_MAX * 100, 1)

    return {
        **normalized,
        "total_score": total,
    }


def decide(
    scorecard: ScorecardResult,
    lower_bound: float,
) -> str:
    """하한값을 엄격히 초과해야 INVEST. 같거나 작으면 HOLD."""
    if isinstance(lower_bound, bool) or not isinstance(
        lower_bound, (int, float)
    ):
        raise ValueError("하한값은 숫자여야 합니다.")

    if not math.isfinite(lower_bound) or not 0 <= lower_bound <= 100:
        raise ValueError("하한값은 0~100 범위여야 합니다.")

    return (
        "INVEST"
        if scorecard["total_score"] > lower_bound
        else "HOLD"
    )