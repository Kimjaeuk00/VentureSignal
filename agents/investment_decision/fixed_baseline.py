"""팀에서 확정한 기준 기업 점수.

- 검색이나 LLM을 호출하지 않는다.
- 기준 기업을 다시 평가하지 않는다.
- 확정한 항목별 점수로 총점을 계산하고 고정 하한값과 대조한다.
- 기준을 변경할 때는 팀 검토 후 버전을 올린다.
"""

from datetime import date

from core.config import SCORE_MAX, SCORECARD_WEIGHTS
from .scoring import REQUIRED_CRITERIA, build_core_scorecard


BASELINE_VERSION = "v1"

# 팀에서 기준 점수를 확정한 날짜. 예: "2026-09-30"
CONFIRMED_AT: str | None = None

# 기준 기업 5곳의 확정 총점 중 최솟값을 입력한다.
FIXED_LOWER_BOUND: float | None = None

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

# 기준 기업은 상장 시점을 기준으로 평가한다.
EVALUATION_BASIS = "listing_date"

# None을 실제 평가 결과로 교체한다.
# as_of에는 해당 평가에 실제 적용한 기준일을 입력한다.
# scores에는 0~5 범위의 확정된 항목별 점수를 입력한다.
REFERENCE_COMPANIES = [
    {
        "company_id": "BENCH_SCINIC",
        "company_name": "싸이닉솔루션",
        "as_of": None,
        "scores": {
            "technology": None,
            "competition": None,
            "market": None,
            "team": None,
            "traction": None,
            "deal_terms": None,
        },
    },
    {
        "company_id": "BENCH_GREEN",
        "company_name": "그린리소스",
        "as_of": None,
        "scores": {
            "technology": None,
            "competition": None,
            "market": None,
            "team": None,
            "traction": None,
            "deal_terms": None,
        },
    },
    {
        "company_id": "C40",
        "company_name": "Blaize",
        "as_of": None,
        "scores": {
            "technology": None,
            "competition": None,
            "market": None,
            "team": None,
            "traction": None,
            "deal_terms": None,
        },
    },
    {
        "company_id": "BENCH_AMBIQ",
        "company_name": "Ambiq",
        "as_of": None,
        "scores": {
            "technology": None,
            "competition": None,
            "market": None,
            "team": None,
            "traction": None,
            "deal_terms": None,
        },
    },
    {
        "company_id": "C49",
        "company_name": "Moore Threads",
        "as_of": None,
        "scores": {
            "technology": None,
            "competition": None,
            "market": None,
            "team": None,
            "traction": None,
            "deal_terms": None,
        },
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
        "C40",
        "BENCH_AMBIQ",
        "C49",
    }

    actual_ids = [
        company["company_id"]
        for company in REFERENCE_COMPANIES
    ]

    if (
        len(actual_ids) != len(expected_ids)
        or set(actual_ids) != expected_ids
    ):
        raise ValueError("기준 기업 5곳이 중복 없이 필요합니다.")

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
                f"{', '.join(missing)}"
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

    calculated_minimum = min(
        company["scorecard"]["total_score"]
        for company in companies
    )

    if FIXED_LOWER_BOUND != calculated_minimum:
        raise ValueError(
            f"고정 하한값 {FIXED_LOWER_BOUND}과 "
            f"기준 기업 최저 총점 {calculated_minimum}이 다릅니다."
        )

    return {
        "version": BASELINE_VERSION,
        "confirmed_at": confirmed_at,
        "evaluation_basis": EVALUATION_BASIS,
        "weights": BASELINE_WEIGHTS.copy(),
        "score_max": BASELINE_SCORE_MAX,
        "comparison": ">",
        "lower_bound": float(FIXED_LOWER_BOUND),
        "companies": companies,
    }