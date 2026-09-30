"""
투자 평가 — 여섯 항목 채점과 가중합 계산

입력:
- current_candidate
- candidates
- evaluations[cid]의 여섯 분석 결과

출력:
- assessment: 항목별 원래 점수, 근거, 미확인 사항
- scorecard: 계산에 적용한 점수와 100점 기준 가중합 총점

처리:
- LLM이 각 항목을 0~5점으로 채점한다.
- Python이 출처와 출력 형식을 검증한다.
- 투자조건이 미산정이면 계산에 기본점수 2.5점을 적용한다.
- 원래 투자조건 분석의 미산정 상태와 미확인 사항은 보존한다.
- 다른 항목이 미산정이면 총점을 계산하지 않는다.
- 고정 하한값 비교와 INVEST/HOLD 판정은 수행하지 않는다.
"""

import json
import re
from copy import deepcopy
from datetime import date, datetime
from zoneinfo import ZoneInfo

from core.config import SCORECARD_WEIGHTS
from core.llm import get_llm
from core.state import GraphState, evaluation_update

from .prompts import INVESTMENT_SYSTEM_PROMPT, INVESTMENT_USER_PROMPT
from .schemas import InvestmentAssessment
from .scoring import build_scorecard


CRITERIA = tuple(SCORECARD_WEIGHTS)

DEFAULT_DEAL_TERMS_SCORE = 2.5

LABELS = {
    "technology": "제품·기술력",
    "competition": "경쟁 우위",
    "market": "시장성",
    "team": "창업자·팀",
    "traction": "실적",
    "deal_terms": "투자조건",
}

# 채점 입력에서 제외할 조사 과정 기록
PROCESS_FIELDS = {
    "initial_missing_information",
    "executed_search_plan",
    "web_search_log",
    "web_search_queries",
}

CITATION_PATTERN = re.compile(
    r"""https?://[^\s<>"'\[\]{}]+|\bS\d{3}\b"""
)


def _to_json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
    )


def _normalize_source(source: str) -> str:
    """RAG 출처의 대괄호만 정리한다. URL은 바꾸지 않는다."""
    source = source.strip()

    match = re.fullmatch(r"\[(S\d{3})\]", source)

    if match:
        return match.group(1)

    return source


def _prepare_analyses(evaluation: dict) -> dict:
    """여섯 분석의 입력 구조를 확인하고 채점용 자료를 만든다."""
    analyses = {}

    for key in CRITERIA:
        analysis = evaluation.get(key)

        if not isinstance(analysis, dict):
            raise ValueError(
                f"{LABELS[key]} 분석이 누락되었습니다. "
                "앞선 에이전트의 실행 결과를 확인하세요."
            )

        summary = analysis.get("summary")

        if not isinstance(summary, str) or not summary.strip():
            raise ValueError(
                f"{LABELS[key]} 분석의 summary가 비어 있습니다."
            )

        if summary.lstrip().upper().startswith(("TODO", "STUB")):
            raise ValueError(
                f"{LABELS[key]} 분석이 아직 미구현 결과입니다. "
                "기업의 투자 위험으로 처리하지 않습니다."
            )

        for field in ("strengths", "risks", "evidence"):
            values = analysis.get(field)

            if not isinstance(values, list) or not all(
                isinstance(value, str)
                for value in values
            ):
                raise ValueError(
                    f"{LABELS[key]}.{field}는 문자열 목록이어야 합니다."
                )

        details = analysis.get("details")

        if not isinstance(details, dict):
            raise ValueError(
                f"{LABELS[key]}.details는 딕셔너리여야 합니다."
            )

        analyses[key] = {
            "summary": summary,
            "strengths": analysis["strengths"],
            "risks": analysis["risks"],
            "evidence": analysis["evidence"],
            "details": {
                name: value
                for name, value in details.items()
                if name not in PROCESS_FIELDS
            },
        }

    return analyses


def _collect_sources(analyses: dict) -> set[str]:
    """입력 분석에 실제로 제공된 출처만 모은다."""
    sources = set()

    for analysis in analyses.values():
        for source in analysis["evidence"]:
            normalized = _normalize_source(source)

            if normalized:
                sources.add(normalized)

    peers = analyses["competition"]["details"].get(
        "peer_products",
        [],
    )

    if isinstance(peers, list):
        for peer in peers:
            if not isinstance(peer, dict):
                continue

            source = peer.get("source_url")

            if isinstance(source, str) and source.strip():
                sources.add(_normalize_source(source))

    return sources


def _normalize_body_citation(
    citation: str,
    allowed_sources: set[str],
) -> str:
    """문장 끝 구두점이 붙은 URL을 실제 출처와 대조한다."""
    citation = _normalize_source(citation)

    if citation in allowed_sources:
        return citation

    # 원본 URL에 구두점이 있는 경우에는 위에서 먼저 일치한다.
    trimmed = citation.rstrip(".,;:!?，。；：！？)]）")

    if trimmed in allowed_sources:
        return trimmed

    return citation


def _validate_assessment(
    assessment: InvestmentAssessment,
    allowed_sources: set[str],
) -> None:
    """LLM이 작성한 원래 점수와 출처를 검증한다."""
    for key in CRITERIA:
        item = getattr(assessment, key)

        item.rationale = item.rationale.strip()

        if not item.rationale:
            raise ValueError(
                f"{LABELS[key]}의 채점 이유가 비어 있습니다."
            )

        item.evidence = list(dict.fromkeys(
            _normalize_source(source)
            for source in item.evidence
            if source.strip()
        ))

        unknown = set(item.evidence) - allowed_sources

        if unknown:
            raise ValueError(
                f"{LABELS[key]}에 제공되지 않은 출처가 있습니다: "
                f"{sorted(unknown)}"
            )

        item.missing_information = list(dict.fromkeys(
            question.strip()
            for question in item.missing_information
            if question.strip()
        ))

        if item.score is not None and not item.evidence:
            raise ValueError(
                f"{LABELS[key]}에 점수는 있지만 출처가 없습니다."
            )

        if item.score is None and not item.missing_information:
            raise ValueError(
                f"{LABELS[key]}가 채점 불가인데 "
                "추가로 필요한 정보가 기록되지 않았습니다."
            )

        body = "\n".join([
            item.rationale,
            *item.missing_information,
        ])

        cited_sources = {
            _normalize_body_citation(citation, allowed_sources)
            for citation in CITATION_PATTERN.findall(body)
        }

        unknown_citations = cited_sources - allowed_sources

        if unknown_citations:
            raise ValueError(
                f"{LABELS[key]} 본문에 알 수 없는 출처가 있습니다: "
                f"{sorted(unknown_citations)}"
            )

        undeclared = cited_sources - set(item.evidence)

        if undeclared:
            raise ValueError(
                f"{LABELS[key]} 본문 출처가 evidence에서 누락되었습니다: "
                f"{sorted(undeclared)}"
            )


def _normalize_founder_sources(analyses: dict) -> dict:
    """창업자·실적 분석의 내부 출처 ID를 URL과 연결한다."""
    result = deepcopy(analyses)

    for section in ("team", "traction", "deal_terms"):
        analysis = result[section]
        details = analysis["details"]

        allowed_urls = {
            source.strip()
            for source in analysis["evidence"]
            if source.startswith(("https://", "http://"))
        }

        source_map = {}

        for source in details.get("sources", []):
            source_id = source.get("source_id")
            url = source.get("url")

            if (
                isinstance(source_id, str)
                and isinstance(url, str)
                and url in allowed_urls
            ):
                source_map[source_id] = url

        normalized_facts = []

        for fact in details.get("facts", []):
            # 동일인 여부가 확인되지 않은 논문은 제외한다.
            if fact.get("category") == "unverified_publication":
                continue

            converted = []

            for evidence in fact.get("evidence", []):
                source_id = evidence.get("source_id")
                url = source_map.get(source_id)

                if url is None:
                    raise ValueError(
                        f"{LABELS[section]} 사실의 출처를 URL로 "
                        f"연결할 수 없습니다: {source_id}"
                    )

                converted.append({
                    "url": url,
                    "quote": evidence.get("quote", ""),
                })

            normalized_facts.append({
                **fact,
                "evidence": converted,
            })

        details["facts"] = normalized_facts

    return result


def build_evaluation_result(assessment_data: dict) -> dict:
    """기본점수 정책을 적용하고 가중합을 계산한다.

    assessment에는 근거에 따른 원래 점수를 보존한다.
    scorecard에는 실제 계산에 적용한 점수를 넣는다.
    """
    assessment_data = deepcopy(assessment_data)

    scores = {
        key: assessment_data[key]["score"]
        for key in CRITERIA
    }

    # 실제 투자조건 점수가 있으면 그대로 사용한다.
    # 미산정일 때만 계산에 기본점수를 적용한다.
    if scores["deal_terms"] is None:
        scores["deal_terms"] = DEFAULT_DEAL_TERMS_SCORE

        note = (
            "[기본점수 적용] 투자조건의 핵심 정보가 부족하여 "
            f"가중합 계산에는 정책상 기본점수 "
            f"{DEFAULT_DEAL_TERMS_SCORE}/5점을 적용했습니다. "
            "실제 자금 여력이나 런웨이가 확인되었다는 뜻은 아닙니다."
        )

        rationale = assessment_data["deal_terms"]["rationale"]

        if note not in rationale:
            assessment_data["deal_terms"]["rationale"] = (
                f"{rationale}\n\n{note}"
            )

    # 투자조건 외 항목이 미산정이면 총점은 계산하지 않는다.
    if any(score is None for score in scores.values()):
        scorecard = None
    else:
        scorecard = build_scorecard(scores)

    return {
        "assessment": assessment_data,
        "scorecard": scorecard,
    }


def assess(state: GraphState) -> dict:
    """여섯 분석 결과를 채점하고 가중합을 계산한다."""
    cid = state.get("current_candidate")

    if not isinstance(cid, str) or not cid:
        raise ValueError("current_candidate에 기업 ID가 필요합니다.")

    candidate = next(
        (
            item
            for item in state.get("candidates", [])
            if item["company_id"] == cid
        ),
        None,
    )

    if candidate is None:
        raise ValueError(f"후보 목록에서 {cid}를 찾지 못했습니다.")

    mode = state.get("evaluation_mode", "candidate")

    if mode not in {"candidate", "benchmark"}:
        raise ValueError("알 수 없는 평가 모드입니다.")

    as_of = state.get("evaluation_as_of")

    if as_of is None:
        if mode == "benchmark":
            raise ValueError("기준 기업의 평가 기준일이 필요합니다.")

        as_of = datetime.now(
            ZoneInfo("Asia/Seoul")
        ).date().isoformat()

    if not isinstance(as_of, str):
        raise ValueError(
            "평가 기준일은 YYYY-MM-DD 문자열이어야 합니다."
        )

    try:
        parsed_date = date.fromisoformat(as_of)
    except ValueError as exc:
        raise ValueError(
            "평가 기준일은 YYYY-MM-DD 형식이어야 합니다."
        ) from exc

    if parsed_date.isoformat() != as_of:
        raise ValueError(
            "평가 기준일은 YYYY-MM-DD 형식이어야 합니다."
        )

    evaluation = state.get("evaluations", {}).get(cid, {})

    analyses = _prepare_analyses(evaluation)
    analyses = _normalize_founder_sources(analyses)

    if mode == "benchmark":
        for key, analysis in analyses.items():
            if analysis["details"].get("as_of") != as_of:
                raise ValueError(
                    f"{LABELS[key]} 분석의 기준일이 "
                    f"요청한 {as_of}와 다릅니다."
                )

    allowed_sources = _collect_sources(analyses)

    candidate_info = {
        key: candidate[key]
        for key in (
            "company_id",
            "company_name",
            "description",
            "domain",
        )
    }

    temporal_instruction = f"""
[이번 평가의 기준일]
- 평가 기준일은 {as_of}입니다.
- 채점 기준에서 '현재'는 위 평가 기준일을 의미합니다.
- 과거 기준 평가에서는 기준일 이후에 발표되거나 달성된 정보를
  당시의 채점 근거로 사용하지 않습니다.
- 당시 공개된 전망은 전망으로만 다루며 달성 실적으로 바꾸지 않습니다.
- 현금, 현금 소진액, 경영진, 고객 관계는 해당 기준일에 적용 가능한
  근거인지 확인합니다.
- 입력 자료가 기준일에 적용 가능한지 불명확하면 추측하지 말고
  missing_information에 기록합니다.
- 기업의 실제 사업 영역에 맞춰 같은 채점 원칙을 적용합니다.
  모든 기업에 NPU 성능이나 자체 칩 양산을 일률적으로 요구하지 않습니다.
"""

    user_prompt = INVESTMENT_USER_PROMPT.format(
        query=state["query"],
        candidate_info=_to_json(candidate_info),
        evaluation_context=_to_json(analyses),
    )

    model = get_llm().with_structured_output(
        InvestmentAssessment
    )

    response = model.invoke([
        (
            "system",
            INVESTMENT_SYSTEM_PROMPT + "\n" + temporal_instruction,
        ),
        ("human", user_prompt),
    ])

    assessment = InvestmentAssessment.model_validate(response)

    # 기본점수 적용 전에 LLM의 원래 출력을 검증한다.
    _validate_assessment(assessment, allowed_sources)

    return build_evaluation_result(assessment.model_dump())


def run(state: GraphState) -> dict:
    """여섯 항목 평가와 가중합 결과를 State에 전달한다."""
    result = assess(state)
    cid = state["current_candidate"]

    return evaluation_update(
        cid,
        assessment=result["assessment"],
        scorecard=result["scorecard"],
    )