"""항목별 점수의 가중합과 하한값 비교.

판정에 쓰는 점수는 핵심 점수(Core Score)다.
- 필수 5항목(제품/기술력, 경쟁 우위, 시장성, 창업자, 실적)이 모두 채점돼야 총점을 낸다. 하나라도 미산정이면 총점을 내지 않는다
  (근거 부족 → HOLD). 큰 항목이 빠진 채 재정규화한 총점은 의미가 없다.
- 투자조건(5%)은 참고 항목이다. 웹 조사로는 현금·소진액이 거의 확인되지 않아 후보를 가르는 정보가 못 되고, 어떤 후보에는 있고
  어떤 후보에는 없으면 후보 간 점수표가 달라진다. 그래서 채점되어 있어도 총점에서는 항상 제외하고 필수 5항목의 비중 합(95%)을
  100점으로 환산한다. 0점이나 기본점수로 채우지 않는다.
"""

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

# 총점 계산에 반드시 필요한 항목과 참고 항목 (참고 항목은 표시만 하고 총점에서 제외한다)
REQUIRED_CRITERIA = ("technology", "competition", "market", "team", "traction")
REFERENCE_CRITERIA = ("deal_terms",)


def build_core_scorecard(
    scores: dict[str, float | None],
) -> ScorecardResult | None:
    """필수 5항목의 가중합을 비중 합(95%)으로 나눠 100점 환산한 핵심 점수. 필수 항목이 하나라도 None 이면 None.

    반환값에는 항목별 점수(미산정은 None, 투자조건은 있으면 참고용으로 그대로), total_score, coverage(총점에 쓴 비중 합),
    unscored(미산정 항목)가 든다.
    """
    expected = set(SCORECARD_WEIGHTS)

    if set(scores) != expected:
        raise ValueError(
            "평가 항목이 일치하지 않습니다. "
            f"필요 항목: {sorted(expected)}"
        )

    scored = {}

    for key, value in scores.items():
        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} 점수는 숫자 또는 None 이어야 합니다.")

        value = float(value)

        if not math.isfinite(value) or not 0 <= value <= SCORE_MAX:
            raise ValueError(
                f"{key} 점수는 0~{SCORE_MAX} 범위여야 합니다."
            )

        scored[key] = value

    if any(key not in scored for key in REQUIRED_CRITERIA):
        return None

    coverage = sum(SCORECARD_WEIGHTS[key] for key in REQUIRED_CRITERIA)
    weighted = sum(scored[key] * SCORECARD_WEIGHTS[key] for key in REQUIRED_CRITERIA)

    return {
        **{key: scored.get(key) for key in SCORECARD_WEIGHTS},
        "total_score": round(weighted / coverage / SCORE_MAX * 100, 1),
        "coverage": round(coverage, 4),
        "unscored": [key for key in SCORECARD_WEIGHTS if key not in scored],
    }
