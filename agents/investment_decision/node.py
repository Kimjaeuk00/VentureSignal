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
- 판정에 쓰는 총점은 핵심 점수다: 필수 5항목(기술력·경쟁 우위·시장성·창업자·실적)의 가중합을 비중 합(95%)으로 나눠
  100점 환산한다. 투자조건은 참고 항목이라 채점되어 있어도 총점에서 항상 제외한다(0점·기본점수로 채우지 않는다).
  필수 항목이 하나라도 미산정이면 총점을 내지 않고 근거 부족으로 HOLD 한다.
- 총점이 하한값 근처이거나 총점이 없으면 채점을 더 해서 항목별 중앙값을 쓴다(채점 흔들림 완화).
- 미산정 항목의 이유와 미확인 사항은 assessment 에 그대로 보존한다.
- 총점을 하한값과 비교해 INVEST/HOLD 를 판정한다(엄격히 초과해야 INVEST). 이 노드가 판정하므로 INVEST 면
  selected_candidate 도 함께 반환해 그래프가 보고서 생성으로 넘어간다.
  - 하한값: 팀이 확정한 기준(fixed_baseline.load_fixed_baseline)이 있으면 그것, 아직 확정 전이면 임시값
    core.config.INVEST_THRESHOLD 를 쓰고 결과의 decision_basis 에 그 사실을 남긴다.
  - 총점이 없으면(투자조건 외 항목 미산정) HOLD.
"""

import json
import logging
import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import date, datetime
from zoneinfo import ZoneInfo

from core.config import INVEST_THRESHOLD, SCORECARD_WEIGHTS
from core.llm import get_llm
from core.sources import SourceBook
from core.state import GraphState, evaluation_update

from .fixed_baseline import load_fixed_baseline
from .prompts import INVESTMENT_SYSTEM_PROMPT, INVESTMENT_USER_PROMPT
from .schemas import InvestmentAssessment
from .scoring import REQUIRED_CRITERIA, build_core_scorecard, decide

logger = logging.getLogger(__name__)


CRITERIA = tuple(SCORECARD_WEIGHTS)

# 채점 결과가 출처·형식 검증에 실패하면 오류를 알려 주고 다시 요청하는 최대 시도 횟수
MAX_ASSESS_ATTEMPTS = 2

# 총점이 하한값에서 이 점수 이내이거나 총점이 없으면(필수 항목이 미산정) 채점을 EXTRA_SAMPLES 번 더 해서
# 항목별 중앙값으로 정한다. 이 모델은 온도를 낮출 수 없어 같은 자료도 채점이 흔들리는데, 판정이 갈리는 구간에서만
# 비용(LLM 호출)을 쓴다. 웹 검색은 다시 하지 않는다.
NEAR_BOUND_MARGIN = 5.0
EXTRA_SAMPLES = 2

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


def _restore_sources(
    assessment: InvestmentAssessment,
    book: SourceBook,
) -> None:
    """LLM 이 쓴 출처 번호(U1…)를 State 의 URL 로 되돌린다. 입력에 없던 번호·URL 은 근거와 본문에서 뺀다.

    LLM 에게는 URL 을 보여 주지 않았으므로 URL 을 베끼다 틀리는 사고가 없다.
    뺀 결과 점수의 근거가 사라지면 이어지는 _validate_assessment 가 실패시켜 다시 요청한다.
    """
    for key in CRITERIA:
        item = getattr(assessment, key)
        item.evidence = book.resolve(item.evidence)
        item.rationale = book.restore(item.rationale)
        item.missing_information = [
            book.restore(text) for text in item.missing_information
        ]


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
    """핵심 점수를 계산한다: 필수 5항목만 100점 환산하고 투자조건은 참고로만 둔다.

    assessment에는 근거에 따른 원래 점수(미산정은 None)를 보존한다.
    scorecard에는 항목별 점수와 핵심 점수(total_score), coverage(총점에 쓴 비중 합), unscored(미산정 항목)를 넣는다.
    필수 항목이 하나라도 미산정이면 scorecard 는 None(근거 부족)이다.
    """
    assessment_data = deepcopy(assessment_data)

    scores = {
        key: assessment_data[key]["score"]
        for key in CRITERIA
    }

    return {
        "assessment": assessment_data,
        "scorecard": build_core_scorecard(scores),
    }


def merge_samples(results: list[dict]) -> dict:
    """여러 번 채점한 결과를 항목별 중앙값으로 합친다.

    항목이 미산정(None)인 채점이 절반 이상이면 그 항목은 미산정이다. 값이 짝수 개면 낮은 쪽 중앙값을 쓴다.
    근거·출처는 선택한 점수와 같은 점수를 낸 첫 채점의 것을 그대로 쓴다(서로 다른 채점의 문장을 섞지 않는다).
    """
    merged = {}

    for key in CRITERIA:
        values = [result["assessment"][key]["score"] for result in results]
        scored = [value for value in values if value is not None]

        chosen = (
            statistics.median_low(scored)
            if len(scored) * 2 > len(values)
            else None
        )

        source = next(
            result["assessment"][key]
            for result in results
            if result["assessment"][key]["score"] == chosen
        )

        merged[key] = {**deepcopy(source), "score": chosen}

    return build_evaluation_result(merged)


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

    # LLM 에게는 URL 대신 출처 번호(U1…)를 보여 준다. RAG 출처 ID(S006)는 그대로 둔다.
    book = SourceBook(rag_ids={s for s in allowed_sources if not s.startswith(("https://", "http://"))})

    user_prompt = INVESTMENT_USER_PROMPT.format(
        query=state["query"],
        candidate_info=_to_json(candidate_info),
        evaluation_context=book.mask(_to_json(analyses)),
    )

    # 프롬프트에 번호로 보여 준 URL 은 모두 입력으로 제공된 출처다 (본문에만 있던 URL 도 포함).
    allowed_sources = allowed_sources | book.urls()

    model = get_llm().with_structured_output(
        InvestmentAssessment
    )

    system_prompt = INVESTMENT_SYSTEM_PROMPT + "\n" + temporal_instruction
    feedback = ""

    for attempt in range(1, MAX_ASSESS_ATTEMPTS + 1):
        human_prompt = user_prompt

        if feedback:
            human_prompt += (
                f"\n\n[이전 시도의 문제] {feedback}\n"
                "입력에 제공된 출처 번호(U1 …)와 S번호만 그대로 사용하고 "
                "출력 규칙을 지켜 다시 작성하세요."
            )

        try:
            response = model.invoke([
                ("system", system_prompt),
                ("human", human_prompt),
            ])

            assessment = InvestmentAssessment.model_validate(response)

            # 출처 번호를 URL 로 되돌린다 (없는 번호는 뺀다).
            _restore_sources(assessment, book)

            # LLM의 원래 출력을 검증한다.
            _validate_assessment(assessment, allowed_sources)
            break
        except ValueError as exc:
            feedback = str(exc)
            logger.warning(
                "채점 결과 검증 실패 (시도 %d/%d): %s",
                attempt, MAX_ASSESS_ATTEMPTS, feedback,
            )

            if attempt == MAX_ASSESS_ATTEMPTS:
                raise

    return build_evaluation_result(assessment.model_dump())


def resolve_lower_bound() -> tuple[float, dict]:
    """(하한값, 판정 근거). 확정된 기준이 있으면 그 하한값, 아직 확정 전이면 임시 하한값."""
    try:
        return load_fixed_baseline()["lower_bound"], {"source": "fixed_baseline"}
    except ValueError as exc:  # 기준 기업 점수·하한값 미확정
        logger.warning("확정된 하한값이 없어 임시값 %s 를 쓴다: %s", INVEST_THRESHOLD, exc)
        return INVEST_THRESHOLD, {"source": "temporary", "reason": str(exc)}


def _needs_more_samples(scorecard, lower_bound: float) -> bool:
    """판정이 갈릴 수 있는 구간인가: 총점이 없거나(필수 항목 미산정 — 한 번의 null 로 HOLD 가 될 수 있다) 하한값 근처."""
    return scorecard is None or abs(scorecard["total_score"] - lower_bound) <= NEAR_BOUND_MARGIN


def _extra_samples(state: GraphState) -> list[dict]:
    """같은 입력으로 채점을 EXTRA_SAMPLES 번 더 한다. 실패한 채점은 버린다."""
    def one(_):
        try:
            return assess(state)
        except Exception:
            logger.exception("추가 채점에 실패해 그 채점은 버린다")
            return None

    with ThreadPoolExecutor(max_workers=EXTRA_SAMPLES) as pool:
        return [result for result in pool.map(one, range(EXTRA_SAMPLES)) if result]


def run(state: GraphState) -> dict:
    """여섯 항목을 채점하고 핵심 점수로 INVEST/HOLD 를 판정한다. INVEST 면 selected_candidate 도 반환한다."""
    cid = state["current_candidate"]
    lower_bound, basis = resolve_lower_bound()

    try:
        first = assess(state)
    except Exception as exc:
        # 한 후보의 채점 실패로 그래프 전체가 멈추지 않게 한다. 이 후보는 HOLD 로 두고 다음 후보로 넘어간다.
        logger.exception("%s 채점에 실패해 HOLD 로 처리한다", cid)
        return evaluation_update(
            cid,
            assessment={},
            scorecard=None,
            decision="HOLD",
            decision_basis={
                "source": "none",
                "hold_reason": "scoring_failed",
                "reason": f"채점 실패: {type(exc).__name__}: {exc}"[:300],
            },
        )

    results = [first]

    if _needs_more_samples(first["scorecard"], lower_bound):
        results += _extra_samples(state)

    result = merge_samples(results) if len(results) > 1 else first
    scorecard = result["scorecard"]

    if scorecard is None:  # 필수 항목이 미산정이라 총점을 내지 않았다
        missing = [
            LABELS[key]
            for key in REQUIRED_CRITERIA
            if result["assessment"][key]["score"] is None
        ]
        decision = "HOLD"
        basis = {
            "source": "none",
            "hold_reason": "insufficient_evidence",
            "reason": f"필수 항목 미산정으로 핵심 점수를 내지 않았다 ({', '.join(missing)})",
            "unscored_required": missing,
            "samples": len(results),
        }
    else:
        decision = decide(scorecard, lower_bound)
        basis = {
            **basis,
            "lower_bound": lower_bound,
            "total_score": scorecard["total_score"],
            "coverage": scorecard["coverage"],
            "unscored": scorecard["unscored"],
            "samples": len(results),
            **({"hold_reason": "score_below_bound"} if decision == "HOLD" else {}),
        }

    update = evaluation_update(
        cid,
        assessment=result["assessment"],
        scorecard=scorecard,
        decision=decision,
        decision_basis=basis,
    )
    if decision == "INVEST":
        update["selected_candidate"] = cid
    return update
