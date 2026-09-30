"""기준 기업 5곳의 최초 평가.

- 후보 탐색과 투자 판정 없이 기준 기업 5곳을 모두 분석한다.
- 후보 기업과 동일한 assess() 함수로 채점한다.
- 결과는 팀 검토용 JSON으로 저장한다.
- fixed_baseline.py의 확정 점수를 자동으로 변경하지 않는다.
"""

import argparse
import importlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from core.config import (
    LLM_MODEL,
    LLM_PROVIDER,
    ROOT_DIR,
    SCORE_MAX,
    SCORECARD_WEIGHTS,
)
from core.state import initial_state, merge_evaluations

from .scoring import build_core_scorecard


# 제공된 기준 기업 자료에 따른 날짜.
# 실제 평가 전에 팀에서 사용할 기준일과 일치하는지 확인한다.
REFERENCE_COMPANIES = [
    {
        "company_id": "BENCH_SCINIC",
        "company_name": "싸이닉솔루션",
        "description": "반도체 설계 및 양산 지원을 제공하는 디자인하우스",
        "domain": "반도체 디자인하우스",
        "as_of": "2025-07-07",
    },
    {
        "company_id": "BENCH_GREEN",
        "company_name": "그린리소스",
        "description": "반도체 장비 부품용 코팅 소재·장비·공정 기업",
        "domain": "반도체 장비 부품 코팅",
        "as_of": "2023-11-24",
    },
    {
        "company_id": "C40",
        "company_name": "Blaize",
        "description": "엣지 AI 프로세서와 소프트웨어 플랫폼 기업",
        "domain": "엣지 AI 프로세서",
        "as_of": "2025-01-14",
    },
    {
        "company_id": "BENCH_AMBIQ",
        "company_name": "Ambiq",
        "description": "초저전력 MCU와 엣지 AI 처리 솔루션 기업",
        "domain": "초저전력 MCU 및 엣지 AI",
        "as_of": "2025-07-30",
    },
    {
        "company_id": "C49",
        "company_name": "Moore Threads",
        "description": "GPU와 AI·HPC 소프트웨어 플랫폼 기업",
        "domain": "AI 및 HPC GPU",
        "as_of": "2025-12-05",
    },
]

ANALYSIS_NODES = [
    ("tech_summary", ("technology",)),
    ("market_eval", ("market",)),
    ("competitor_compare", ("competition",)),
    ("founder_traction", ("team", "traction", "deal_terms")),
]


def _load_analysis_nodes():
    """과거 기준일 분석을 지원하는 노드만 실행 대상으로 가져온다.

    담당자는 실제 검색·분석의 날짜 처리를 구현한 뒤,
    해당 node.py에 SUPPORTS_EVALUATION_AS_OF = True를 선언한다.
    선언만 추가하는 것은 구현 완료가 아니다.
    """
    nodes = []
    unsupported = []

    for name, fields in ANALYSIS_NODES:
        module = importlib.import_module(f"agents.{name}.node")

        if getattr(module, "SUPPORTS_EVALUATION_AS_OF", False) is not True:
            unsupported.append(name)

        nodes.append((name, module.run, fields))

    if unsupported:
        raise RuntimeError(
            "상장 시점 분석 지원이 확인되지 않은 에이전트: "
            + ", ".join(unsupported)
            + ". 최초 기준 평가를 시작하지 않았습니다."
        )

    return nodes


def _save(path: Path, result: dict) -> None:
    """현재 실행의 중간 결과를 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)

    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _apply_analysis(
    state: dict,
    patch: dict,
    fields: tuple[str, ...],
    as_of: str,
) -> None:
    """해당 기업의 분석과 평가 기준일을 확인하고 State에 병합한다."""
    cid = state["current_candidate"]
    updates = patch.get("evaluations", {})

    if set(updates) != {cid}:
        raise ValueError(
            f"분석 결과는 현재 기업 {cid}의 evaluations만 반환해야 합니다."
        )

    evaluation = updates[cid]

    for field in fields:
        analysis = evaluation.get(field)

        if not isinstance(analysis, dict):
            raise ValueError(f"{cid}.{field} 분석이 없습니다.")

        details = analysis.get("details")
        if not isinstance(details, dict):
            raise ValueError(f"{cid}.{field}.details가 없습니다.")

        # 이 값의 일치만으로 자료의 역사적 정확성이 보장되지는 않는다.
        # 검색·본문의 날짜 검토는 각 분석 에이전트가 수행해야 한다.
        if details.get("as_of") != as_of:
            raise ValueError(
                f"{cid}.{field}의 평가 기준일이 일치하지 않습니다. "
                f"필요한 기준일: {as_of}"
            )

    state["evaluations"] = merge_evaluations(
        state["evaluations"],
        updates,
    )


def _calculate_lower_bound(companies: list[dict]) -> float | None:
    """5곳 모두 채점됐을 때만 최저 총점을 반환한다."""
    expected_ids = {
        company["company_id"]
        for company in REFERENCE_COMPANIES
    }

    actual_ids = [
        company["company_id"]
        for company in companies
    ]

    if (
        len(actual_ids) != len(expected_ids)
        or set(actual_ids) != expected_ids
    ):
        return None

    totals = []

    for company in companies:
        if company["scorecard"] is None:
            return None

        scores = {
            key: company["assessment"][key]["score"]
            for key in SCORECARD_WEIGHTS
        }

        # 후보 채점과 같은 함수로 다시 계산한다(필수 5항목의 핵심 점수, 투자조건은 참고).
        recalculated = build_core_scorecard(scores)

        if recalculated != company["scorecard"]:
            raise ValueError(
                f"{company['company_name']}의 항목별 점수와 총점이 다릅니다."
            )

        totals.append(recalculated["total_score"])

    return min(totals)


def build_reference_scores() -> dict:
    """기준 기업을 최초 평가하여 팀 검토용 결과를 저장한다."""
    # 모든 노드의 지원 여부를 먼저 확인한다.
    # 지원하지 않으면 검색·채점 API 호출 전에 중단한다.
    nodes = _load_analysis_nodes()

    from .node import assess

    now = datetime.now(ZoneInfo("Asia/Seoul"))

    # 실행마다 다른 파일을 만들어 이전 평가 근거를 보존한다.
    output_path = (
        ROOT_DIR
        / "outputs"
        / "benchmark"
        / f"reference_{now.strftime('%Y%m%d_%H%M%S_%f')}.json"
    )

    result = {
        "status": "running",
        "evaluation_basis": "listing_date",
        "evaluated_at": now.isoformat(),
        "model": LLM_MODEL,
        "provider": LLM_PROVIDER,
        "weights": dict(SCORECARD_WEIGHTS),
        "score_max": SCORE_MAX,
        "comparison": ">",
        "proposed_lower_bound": None,
        "companies": [],
    }

    _save(output_path, result)

    try:
        for reference in REFERENCE_COMPANIES:
            cid = reference["company_id"]
            as_of = reference["as_of"]

            candidate = {
                key: reference[key]
                for key in (
                    "company_id",
                    "company_name",
                    "description",
                    "domain",
                )
            }
            candidate["retrieval_score"] = 0.0

            state = initial_state(
                f"{reference['company_name']}을 {as_of} 기준으로 분석한다. "
                "당시 공개되어 있던 근거만 사용하고, "
                "기준일 이후에 달성한 성과는 포함하지 않는다. "
                "당시의 전망과 실제 달성 실적을 구분한다."
            )

            state["candidates"] = [candidate]
            state["current_candidate"] = cid
            state["candidate_index"] = 1

            # 분석 에이전트와 합의할 입력 필드.
            state["evaluation_mode"] = "benchmark"
            state["evaluation_as_of"] = as_of

            print(
                f"기준 기업 분석 시작: "
                f"{reference['company_name']} / {as_of}"
            )

            for name, run_analysis, fields in nodes:
                print(f"  실행: {name}")

                patch = run_analysis(state)

                _apply_analysis(
                    state=state,
                    patch=patch,
                    fields=fields,
                    as_of=as_of,
                )

            # run()을 호출하지 않는다.
            # 고정 하한값 없이 공통 채점 함수만 실행한다.
            assessment_result = assess(state)

            company_result = {
                "company_id": cid,
                "company_name": reference["company_name"],
                "as_of": as_of,
                **assessment_result,
                "analyses": state["evaluations"][cid],
            }

            result["companies"].append(company_result)
            _save(output_path, result)

            scorecard = assessment_result["scorecard"]
            total = (
                scorecard["total_score"]
                if scorecard is not None
                else "미산정"
            )

            print(f"  총점: {total}")

        lower_bound = _calculate_lower_bound(result["companies"])

        result["proposed_lower_bound"] = lower_bound
        result["status"] = (
            "ready_for_review"
            if lower_bound is not None
            else "needs_more_evidence"
        )

        _save(output_path, result)

    except Exception as exc:
        result["status"] = "failed"
        # API 키 등 민감한 정보가 오류 메시지에 포함될 수 있어
        # 저장 파일에는 예외 종류만 남긴다.
        result["error_type"] = type(exc).__name__
        _save(output_path, result)
        print(f"중간 결과 저장 위치: {output_path}")
        raise

    print(f"검토용 결과 저장 위치: {output_path}")

    if result["proposed_lower_bound"] is None:
        print(
            "채점하지 못한 기업이 있어 하한값을 산정하지 않았습니다. "
            "assessment의 missing_information을 확인하세요."
        )
    else:
        print(
            f"검토용 하한값: "
            f"{result['proposed_lower_bound']:.1f}점"
        )
        print(
            "팀 검토 후 fixed_baseline.py에 확정 점수를 입력하세요. "
            "고정 기준은 자동으로 변경하지 않았습니다."
        )

    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build",
        action="store_true",
        help="기준 기업 5곳을 실제 API로 분석하여 검토용 결과 저장",
    )

    args = parser.parse_args()

    if args.build:
        build_reference_scores()
    else:
        from .fixed_baseline import load_fixed_baseline

        print(
            json.dumps(
                load_fixed_baseline(),
                ensure_ascii=False,
                indent=2,
            )
        )