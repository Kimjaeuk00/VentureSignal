"""팀에서 확정한 기준 기업 점수와 하한값.

- 검색이나 LLM을 호출하지 않는다.
- 기준 기업을 다시 평가하지 않는다.
- 확정한 항목별 점수로 핵심 점수를 계산하고, 그 평균을 고정 하한값과 대조한다.
- 기준을 변경할 때는 팀 검토 후 버전을 올린다.

하한값 규칙 (v3, 2026-09-30 확정): 산정된 기준 기업 핵심 점수의 **평균**(소수 첫째 자리 반올림). 설계 문서의 최저값 방식과 다르다.
핵심 점수는 후보와 같은 규칙이다: 필수 5항목의 가중합 ÷ 95% × 100, 투자조건은 참고라 제외.
산정 과정과 근거는 docs/기준기업_3종_점수산정.md 에 있다.
"""

import statistics
from datetime import date

from core.config import SCORE_MAX, SCORECARD_WEIGHTS
from .scoring import REQUIRED_CRITERIA, build_core_scorecard


# v1: 최저값 방식(기준 점수 미확정) → v3: 기준 기업 3곳 핵심 점수(투자조건 제외)의 평균
BASELINE_VERSION = "v3"

# 팀에서 기준 점수를 확정한 날짜.
CONFIRMED_AT: str | None = "2026-09-30"

# 기준 기업 3곳 핵심 점수의 평균 (싸이닉솔루션 57.9, 그린리소스 60.0, Ambiq 60.0 → 59.3).
FIXED_LOWER_BOUND: float | None = 59.3

# 기준 점수를 산정할 때 사용한 가중치와 점수 범위.
BASELINE_WEIGHTS = {
    "technology": 0.35,
    "competition": 0.25,
    "market": 0.15,
    "team": 0.10,
    "traction": 0.10,
    "deal_terms": 0.05,
}

BASELINE_SCORE_MAX = 5

# 기술력·시장성·경쟁 우위는 상장일 기준 자료(PDF)를, 창업자·실적·경쟁사 비교 등 웹 검색분은 산정 시점(2026-09-30)의 정보를 썼다.
EVALUATION_BASIS = "listing_date_pdf_plus_web_at_scoring"

# as_of: 평가 기준일(상장일). scores: 채점 3회의 항목별 중앙값(0~5 정수, 미산정은 None). 투자조건은 참고 항목이라 총점에 쓰지 않는다.
REFERENCE_COMPANIES = [
    {
        "company_id": "BENCH_SCINIC",
        "company_name": "싸이닉솔루션",
        "as_of": "2025-07-07",
        "scores": {"technology": 3, "competition": 3, "market": 3, "team": 2, "traction": 3, "deal_terms": None},
    },
    {
        "company_id": "BENCH_GREEN",
        "company_name": "그린리소스",
        "as_of": "2023-11-24",
        "scores": {"technology": 3, "competition": 3, "market": 3, "team": 3, "traction": 3, "deal_terms": None},
    },
    {
        "company_id": "BENCH_AMBIQ",
        "company_name": "Ambiq",
        "as_of": "2025-07-30",
        "scores": {"technology": 3, "competition": 3, "market": 3, "team": 3, "traction": 3, "deal_terms": None},
    },
]


def _validate_date(value: str | None, label: str) -> str:
    """확정일과 평가 기준일이 올바른 날짜인지 확인한다."""
    if not isinstance(value, str):
        raise ValueError(f"{label}을 YYYY-MM-DD 형식으로 입력하세요.")

    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(
            f"{label}은 YYYY-MM-DD 형식이어야 합니다."
        ) from exc

    if parsed.isoformat() != value:
        raise ValueError(
            f"{label}은 YYYY-MM-DD 형식이어야 합니다."
        )

    return value


def load_fixed_baseline() -> dict:
    """확정된 점수를 검증해서 반환한다. 외부 API는 호출하지 않는다."""
    confirmed_at = _validate_date(CONFIRMED_AT, "기준 확정일")

    if not BASELINE_VERSION.strip():
        raise ValueError("기준 버전이 필요합니다.")

    # 다른 가중치로 계산한 후보 점수와 비교하지 않는다.
    if (
        SCORECARD_WEIGHTS != BASELINE_WEIGHTS
        or SCORE_MAX != BASELINE_SCORE_MAX
    ):
        raise ValueError(
            "현재 채점 가중치 또는 점수 범위가 고정 기준과 다릅니다. "
            "팀에서 평가 기준과 기준 버전을 확인하세요."
        )

    expected_ids = {
        "BENCH_SCINIC",
        "BENCH_GREEN",
        "BENCH_AMBIQ",
    }

    actual_ids = [
        company["company_id"]
        for company in REFERENCE_COMPANIES
    ]

    if (
        len(actual_ids) != len(expected_ids)
        or set(actual_ids) != expected_ids
    ):
        raise ValueError("기준 기업 3곳이 중복 없이 필요합니다.")

    companies = []

    for company in REFERENCE_COMPANIES:
        name = company["company_name"]
        as_of = _validate_date(
            company["as_of"],
            f"{name} 평가 기준일",
        )

        scores = company["scores"]

        if set(scores) != set(BASELINE_WEIGHTS):
            raise ValueError(f"{name}의 평가 항목이 일치하지 않습니다.")

        if all(value is None for value in scores.values()):
            raise ValueError(
                f"{name}의 기준 점수가 아직 확정되지 않았습니다."
            )

        # 현재 채점 스키마와 동일하게 항목별 정수 점수를 사용한다.
        # 후보와 같은 핵심 점수 규칙을 쓴다: 필수 5항목만 100점 환산하고 투자조건은 참고(미산정 None 허용).
        for key, value in scores.items():
            if value is not None and type(value) is not int:
                raise ValueError(
                    f"{name}.{key}는 0~5 범위의 정수 또는 None(미산정)이어야 합니다."
                )

        scorecard = build_core_scorecard(scores)

        if scorecard is None:
            missing = [key for key in REQUIRED_CRITERIA if scores[key] is None]
            raise ValueError(
                f"{name}의 필수 항목 점수가 없어 핵심 점수를 산정할 수 없습니다: "
                f"{', '.join(missing)}."
            )

        companies.append({
            "company_id": company["company_id"],
            "company_name": name,
            "as_of": as_of,
            "scorecard": scorecard,
        })

    if (
        isinstance(FIXED_LOWER_BOUND, bool)
        or not isinstance(FIXED_LOWER_BOUND, (int, float))
    ):
        raise ValueError("확정된 하한값을 입력하세요.")

    calculated_mean = round(
        statistics.mean(
            company["scorecard"]["total_score"]
            for company in companies
        ),
        1,
    )

    if FIXED_LOWER_BOUND != calculated_mean:
        raise ValueError(
            f"고정 하한값 {FIXED_LOWER_BOUND}과 "
            f"기준 기업 평균 총점 {calculated_mean}이 다릅니다."
        )

    return {
        "version": BASELINE_VERSION,
        "confirmed_at": confirmed_at,
        "evaluation_basis": EVALUATION_BASIS,
        "weights": BASELINE_WEIGHTS.copy(),
        "score_max": BASELINE_SCORE_MAX,
        "comparison": ">",
        "lower_bound_method": "mean",
        "lower_bound": float(FIXED_LOWER_BOUND),
        "companies": companies,
    }
