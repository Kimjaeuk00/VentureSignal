"""
결정적 품질 검사 (LLM 호출 없음)

각 함수는 문제 목록(list[str])을 돌려준다. 빈 목록이면 통과다.
"""

import re
from collections import Counter

from agents.quality_check.policy import (
    CITATION_RE,
    MAX_TOP_SOURCE_SHARE,
    MIN_CITATIONS_FOR_SHARE,
    MIN_DISTINCT_SOURCES,
    PROMOTIONAL_PHRASES,
    RANKING_PHRASES,
    REFERENCE_ENTRY_RE,
    REFERENCE_HEADING_RE,
)
from core.state import CandidateEvaluation


def split_report(report: str) -> tuple[str, str]:
    """(본문, REFERENCE 섹션). REFERENCE 헤딩이 없으면 (전체, "")."""
    match = REFERENCE_HEADING_RE.search(report)
    if not match:
        return report, ""
    return report[: match.start()], report[match.end():]


def cited_numbers(body: str) -> list[int]:
    """본문의 [번호] 인용을 등장 순서대로 (중복 포함) 모두 꺼낸다. [1, 2] 는 1과 2."""
    numbers: list[int] = []
    for group in CITATION_RE.findall(body):
        numbers.extend(int(n) for n in re.split(r"\s*,\s*", group))
    return numbers


def is_report(report: str) -> bool:
    """보고서 형식인가. 안내 문자열이나 생성 실패 문구는 평가 대상이 아니다."""
    return bool(report) and bool(REFERENCE_HEADING_RE.search(report))


def check_groundedness(report: str) -> list[str]:
    body, reference = split_report(report)
    referenced = {int(n) for n in REFERENCE_ENTRY_RE.findall(reference)}
    cited = cited_numbers(body)

    problems = []
    if not cited:
        problems.append("본문에 [번호] 출처 인용이 없다")
    dangling = sorted({n for n in cited if n not in referenced})
    if dangling:
        problems.append(f"REFERENCE 에 없는 출처 번호를 인용했다: {', '.join(f'[{n}]' for n in dangling)}")
    return problems


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?。])\s+|\n+", text) if s.strip()]


def check_neutrality(report: str, candidate_names: list[str]) -> list[str]:
    body, _ = split_report(report)
    problems = []

    for phrase in PROMOTIONAL_PHRASES:
        if phrase in body:
            problems.append(f"근거 없는 단정·추천 표현이 있다: '{phrase}'")

    names = [n for n in dict.fromkeys(candidate_names) if n]
    for sentence in _sentences(body):
        mentioned = [n for n in names if n in sentence]
        if len(mentioned) >= 2 and any(p in sentence for p in RANKING_PHRASES):
            problems.append(f"후보 간 우열을 직접 서술한 문장이 있다: '{sentence[:80]}'")
    return problems


def check_bias_control(report: str, evaluations: dict[str, CandidateEvaluation]) -> list[str]:
    body, _ = split_report(report)
    cited = cited_numbers(body)
    problems = []

    distinct = set(cited)
    if cited and len(distinct) < MIN_DISTINCT_SOURCES:
        problems.append(f"인용한 출처가 {len(distinct)}건뿐이다 (최소 {MIN_DISTINCT_SOURCES}건)")

    if len(cited) >= MIN_CITATIONS_FOR_SHARE:
        number, count = Counter(cited).most_common(1)[0]
        share = count / len(cited)
        if share > MAX_TOP_SOURCE_SHARE:
            problems.append(f"출처 [{number}] 하나가 인용의 {share:.0%}를 차지한다 (상한 {MAX_TOP_SOURCE_SHARE:.0%})")

    strengths = risks = 0
    for evaluation in evaluations.values():
        for field in ("technology", "market", "competition", "team", "traction", "deal_terms"):
            analysis = evaluation.get(field) or {}
            strengths += len(analysis.get("strengths") or [])
            risks += len(analysis.get("risks") or [])
    if strengths > 0 and risks == 0:
        problems.append("분석 결과에 강점만 있고 위험 근거가 하나도 없다 (유리한 근거 편중)")

    return problems
