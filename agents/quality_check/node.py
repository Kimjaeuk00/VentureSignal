"""
보고서 품질 평가

입력: state["report"], state["candidates"], state["evaluations"], state["quality_attempts"]
출력: {"quality_verdict": QualityVerdict, "quality_attempts": 평가 횟수}

항목 3개(Groundedness / 중립성 / 편향 통제)를 Hybrid 로 판정한다.
- 결정적 검사(checks.py): 인용 번호가 REFERENCE 에 있는지, 단정·우열 표현, 출처 편중, 강점만 있는 근거
- LLM Judge: 같은 3개 항목에서 문제 문장을 인용과 함께 지적. 인용문이 보고서에 없으면 그 지적은 버린다.
항목은 결정적 검사와 Judge 가 모두 문제를 찾지 못해야 통과한다. LLM 호출이 실패하면 결정적 검사만으로 판정한다.

보고서 본문은 수정하지 않는다. 미달이면 feedback 에 문제를 남기고, 그래프가 Supervisor 를 거쳐 보고서를 다시 쓰게 한다
(재작성 횟수는 policy.MAX_REVISIONS 로 제한한다).
"""

import json
import logging
import re

from agents.quality_check.checks import (
    check_bias_control,
    check_groundedness,
    check_neutrality,
    is_report,
)
from agents.quality_check.policy import JUDGE_MAX_REPORT_CHARS
from agents.quality_check.prompts import JUDGE_SYSTEM
from agents.quality_check.schemas import JudgeOutput
from agents.supervisor.audit import log_decision
from core.llm import get_llm
from core.state import GraphState, QualityVerdict

logger = logging.getLogger(__name__)

ITEM_LABELS = {"groundedness": "Groundedness", "neutrality": "중립성", "bias_control": "편향 통제"}

NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
CITATION_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")
CONTEXT_FIELDS = ("technology", "market", "competition", "team", "traction", "deal_terms")
CONTEXT_CHARS = 8000  # LLM Judge 에 보여 주는 분석 결과 길이 (숫자 대조는 전체로 한다)


def _score_summary(state: GraphState) -> str:
    lines = []
    names = {c["company_id"]: c["company_name"] for c in state["candidates"]}
    for cid, evaluation in state["evaluations"].items():
        scorecard = evaluation.get("scorecard") or {}
        lines.append(f"- {names.get(cid, cid)}: 판정 {evaluation.get('decision', '미확인')}, 총점 {scorecard.get('total_score', '미확인')}")
    return "\n".join(lines) or "- (평가 결과 없음)"


def _numbers(text: str) -> set[str]:
    """문자열 속 숫자 집합. 천 단위 쉼표는 없앤다 ("1,300" → "1300")."""
    return {n.replace(",", "") for n in NUMBER_RE.findall(text)}


def _analysis_text(state: GraphState) -> str:
    """보고서가 근거로 삼은 분석 결과 전체 (숫자 대조용)."""
    return json.dumps(
        {cid: {k: ev.get(k) for k in CONTEXT_FIELDS if ev.get(k)} for cid, ev in state["evaluations"].items()},
        ensure_ascii=False,
    )


def _analysis_context(state: GraphState) -> str:
    """LLM Judge 에 보여 줄 분석 결과 요약 (후보별·항목별 요약과 강점·위험 일부)."""
    names = {c["company_id"]: c["company_name"] for c in state["candidates"]}
    lines = []
    for cid, ev in state["evaluations"].items():
        for key in CONTEXT_FIELDS:
            analysis = ev.get(key) or {}
            if not analysis:
                continue
            extra = "; ".join([*(analysis.get("strengths") or [])[:3], *(analysis.get("risks") or [])[:3]])
            lines.append(f"- {names.get(cid, cid)} / {key}: {analysis.get('summary', '')} | {extra}")
    return "\n".join(lines)[:CONTEXT_CHARS] or "(분석 결과 없음)"


def _ungrounded_numbers(quote: str, analysis_text: str) -> set[str]:
    """인용문의 숫자 중 분석 결과에 없는 것. 인용 번호([3, 4])는 제외한다."""
    return _numbers(CITATION_RE.sub("", quote)) - _numbers(analysis_text)


def judge(state: GraphState, report: str) -> dict[str, list[str]]:
    """LLM Judge 의 지적을 항목별로 모은다. 실패하면 빈 결과.

    LLM 이 출처 본문을 보지 못해 생기는 오탐을 막으려고 지적을 검증한다.
    - 인용문이 보고서에 없으면 버린다
    - groundedness 지적은 인용문에 분석 결과에 없는 숫자가 있을 때만 인정한다 (출처가 뒷받침하는지 같은 추측은 버린다)
    """
    found: dict[str, list[str]] = {item: [] for item in ITEM_LABELS}
    user = (
        f"[후보별 판정]\n{_score_summary(state)}\n\n[분석 결과]\n{_analysis_context(state)}\n\n"
        f"[보고서]\n{report[:JUDGE_MAX_REPORT_CHARS]}"
    )
    try:
        output: JudgeOutput = get_llm().with_structured_output(JudgeOutput).invoke([("system", JUDGE_SYSTEM), ("human", user)])
    except Exception:
        logger.exception("LLM Judge 호출 실패 → 결정적 검사만으로 판정한다")
        return found

    analysis_text = _analysis_text(state)
    for issue in output.issues:
        quote = issue.quote.strip()
        if not quote or quote not in report:
            logger.info("보고서에 없는 인용문이라 Judge 지적을 버린다: %s", quote[:60])
            continue
        reason = issue.reason
        if issue.item == "groundedness":
            missing = _ungrounded_numbers(quote, analysis_text)
            if not missing:
                logger.info("인용문의 숫자가 모두 분석 결과에 있어 groundedness 지적을 버린다: %s", quote[:60])
                continue
            reason += f" (분석 결과에 없는 수치: {', '.join(sorted(missing)[:5])})"
        found[issue.item].append(f"{reason} (\"{quote[:80]}\")")
    return found


def run(state: GraphState) -> dict:
    attempts = state["quality_attempts"] + 1
    report = state["report"] or ""

    if not is_report(report):
        # 안내 문자열이나 생성 실패 문구는 다시 써도 달라지지 않는다. 평가를 건너뛰고 통과로 둔다.
        verdict: QualityVerdict = {
            "groundedness": True,
            "neutrality": True,
            "bias_control": True,
            "passed": True,
            "feedback": ["평가할 보고서 본문이 없어 품질 평가를 건너뛰었다"],
        }
    else:
        names = [c["company_name"] for c in state["candidates"] if c["company_id"] in state["evaluations"]]
        problems = {
            "groundedness": check_groundedness(report),
            "neutrality": check_neutrality(report, names),
            "bias_control": check_bias_control(report, state["evaluations"]),
        }
        for item, issues in judge(state, report).items():
            problems[item] += issues

        verdict = {
            "groundedness": not problems["groundedness"],
            "neutrality": not problems["neutrality"],
            "bias_control": not problems["bias_control"],
            "passed": not any(problems.values()),
            "feedback": [f"[{ITEM_LABELS[item]}] {text}" for item, texts in problems.items() for text in texts],
        }

    log_decision(
        state,
        "quality_check",
        "passed" if verdict["passed"] else "failed",
        " / ".join(verdict["feedback"])[:300],
        attempt=attempts,
    )
    return {"quality_verdict": verdict, "quality_attempts": attempts}
