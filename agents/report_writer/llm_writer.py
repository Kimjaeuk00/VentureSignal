"""
LLM 문장 작성과 근거 검증.

LLM 은 요약·항목별 근거·판단 이유(보류 사유)·재검토 조건만 쓴다. 출력은 검증을 통과해야 쓰고,
검증 실패·호출 실패 시에는 fallback.py 의 결정적 문장으로 대체해 보고서가 항상 완성되게 한다.

검증: 출력에 든 숫자가 모두 입력(자료·점수·비중)에 있어야 한다. 없으면 한 번 다시 시도(어떤 수치가 문제였는지 알려 줌)하고,
그래도 실패하면 폴백을 쓴다. 보류 보고서는 후보별로 검증해 실패한 후보만 폴백으로 바꾼다.
"""

import logging
import re
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Optional

from langsmith import traceable

from core.config import SCORECARD_WEIGHTS
from core.llm import get_llm

from .collect import ITEM_LABEL, ITEM_ORDER, MISSING, CandidateData, ReportContext, get, missing_labels
from .fallback import SCORE_MAX, hold_prose, invest_prose
from .prompts import HOLD_SYSTEM, INVEST_SYSTEM
from .schemas import CompanyHoldProse, HoldProse, InvestProse

logger = logging.getLogger(__name__)

NUMBER_RE = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
INLINE_SOURCE_RE = re.compile(r"\s*\[?\(?(?:S\d{3}|https?://[^\s)\]]+)(?:[,;·\s]+(?:S\d{3}|분석))*\)?\]?")
MAX_ATTEMPTS = 2
LLM_CALL_LIMIT = 3  # 보고서 한 건을 만드는 데 쓰는 LLM 호출 상한 (문장 작성 + 분량 요약). 넘으면 결정적 처리로 대신한다.

_remaining: ContextVar[Optional[list[int]]] = ContextVar("report_llm_remaining", default=None)


class LlmBudgetExceeded(RuntimeError):
    """보고서 한 건의 LLM 호출 상한을 넘었다."""


@contextmanager
def llm_budget(limit: int = LLM_CALL_LIMIT):
    """이 안에서 _invoke 를 부를 수 있는 횟수를 제한한다. 안 쓰면(테스트·다른 용도) 제한이 없다."""
    token = _remaining.set([limit])
    try:
        yield
    finally:
        _remaining.reset(token)


def _clean(text: str) -> str:
    """LLM 이 출처 표기를 따라 쓰지 않도록 입력에서 인라인 출처(S번호·URL)를 뺀다."""
    return " ".join(INLINE_SOURCE_RE.sub("", text or "").split())


def _numbers(text: str) -> set[str]:
    out = set()
    for raw in NUMBER_RE.findall(text or ""):
        out.add(f"{float(raw.replace(',', '')):g}")
    return out


def ungrounded_numbers(output_text: str, source_text: str) -> list[str]:
    """출력에는 있는데 입력에는 없는 숫자."""
    allowed = _numbers(source_text) | {f"{v:g}" for v in (0, 5, 100)} | {f"{w * 100:g}" for w in SCORECARD_WEIGHTS.values()}
    return sorted(_numbers(output_text) - allowed, key=float)


# --------------------------------------------------------------------------- 입력 만들기


def _render_candidate(c: CandidateData) -> str:
    lines = [f"[기업] company_id={c.company_id} · {c.company_name} · 분야 {c.domain or MISSING}", f"소개: {_clean(c.description) or MISSING}"]
    total = c.scorecard.get("total_score")
    lines.append(f"총점: {total:g} / 100" if isinstance(total, (int, float)) else f"총점: {MISSING}")
    for key, label in ITEM_ORDER:
        score = c.scorecard.get(key)
        score_text = f"{score:g}/{SCORE_MAX}" if isinstance(score, (int, float)) else MISSING
        lines.append(f"\n- {label} (비중 {SCORECARD_WEIGHTS[key] * 100:g}%, 점수 {score_text})")
        if c.is_degraded(key):
            lines.append(f"  상태: 근거 부족 또는 자료 없음 — {_clean(get(c.analysis(key), 'summary', default=MISSING))}")
            continue
        a = c.analysis(key)
        lines.append(f"  요약: {_clean(get(a, 'summary', default=MISSING))}")
        for name, field in (("강점", "strengths"), ("위험", "risks")):
            items = [_clean(x) for x in (get(a, field, default=[]) or [])][:4]
            if items:
                lines.append(f"  {name}: " + " / ".join(items))
    missing = missing_labels(c, [k for k, _ in ITEM_ORDER])
    lines.append("\n확인하지 못한 항목: " + (", ".join(missing[:10]) if missing else MISSING))
    return "\n".join(lines)


# --------------------------------------------------------------------------- 호출


def _invoke(schema, system: str, user: str, feedback: str = ""):
    remaining = _remaining.get()
    if remaining is not None:
        if remaining[0] <= 0:
            raise LlmBudgetExceeded(f"LLM 호출 상한({LLM_CALL_LIMIT}회)을 넘었다")
        remaining[0] -= 1
    messages = [("system", system), ("human", user + (f"\n\n[이전 시도의 문제] {feedback}\n입력 자료에 있는 수치만 사용해 다시 작성하라." if feedback else ""))]
    return get_llm().with_structured_output(schema).invoke(messages)


def _dump(model) -> str:
    return " ".join(str(v) for v in model.model_dump().values() if not isinstance(v, (dict, list))) + " " + " ".join(
        str(x) for v in model.model_dump().values() if isinstance(v, dict) for x in v.values())


@traceable(name="report_writer.invest_prose", run_type="chain")
def write_invest_prose(ctx: ReportContext) -> InvestProse:
    """LLM 문장 → 검증 → 실패 시 결정적 폴백."""
    c = ctx.selected
    source = _render_candidate(c)
    user = f"{source}\n\n위 기업의 투자 검토 보고서 문장을 작성하라."
    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            prose: InvestProse = _invoke(InvestProse, INVEST_SYSTEM, user, feedback)
        except Exception:
            logger.exception("보고서 문장 LLM 호출 실패 → 결정적 문장으로 대체합니다")
            break
        bad = ungrounded_numbers(_dump(prose) + " " + " ".join(prose.item_rationales.model_dump().values()), source)
        if not bad:
            return prose
        feedback = f"입력에 없는 숫자가 있다: {', '.join(bad)}"
        logger.warning("LLM 문장이 검증에 실패했다 (시도 %d/%d): %s", attempt, MAX_ATTEMPTS, feedback)
    return invest_prose(ctx)


@traceable(name="report_writer.hold_prose", run_type="chain")
def write_hold_prose(ctx: ReportContext) -> HoldProse:
    """후보별로 검증해 실패한 후보만 결정적 문장으로 바꾼다. 후보 자료는 서로 섞지 않는다."""
    fallback = hold_prose(ctx)
    fb_by_id = {p.company_id: p for p in fallback.companies}
    sources = {c.company_id: _render_candidate(c) for c in ctx.candidates}
    all_source = "\n\n".join(sources.values())
    user = f"{all_source}\n\n위 후보 {len(ctx.candidates)}곳이 모두 보류다. 보류 보고서 문장을 작성하라."

    feedback = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            prose: HoldProse = _invoke(HoldProse, HOLD_SYSTEM, user, feedback)
        except Exception:
            logger.exception("보류 문장 LLM 호출 실패 → 결정적 문장으로 대체합니다")
            return fallback

        companies, problems = [], []
        by_id = {p.company_id: p for p in prose.companies}
        for cand in ctx.candidates:
            p: Optional[CompanyHoldProse] = by_id.get(cand.company_id)
            bad = ungrounded_numbers(f"{p.reasons} {p.revisit_conditions}", sources[cand.company_id]) if p else ["누락"]
            if bad:
                problems.append(f"{cand.company_name}: {', '.join(bad)}")
                companies.append(fb_by_id[cand.company_id])
            else:
                companies.append(p)
        bad_summary = ungrounded_numbers(prose.summary, all_source)
        if bad_summary:
            problems.append(f"요약: {', '.join(bad_summary)}")
        if not problems:
            return HoldProse(summary=prose.summary, companies=companies)
        feedback = "검증에 실패한 부분 — " + "; ".join(problems)
        logger.warning("LLM 보류 문장이 검증에 실패했다 (시도 %d/%d): %s", attempt, MAX_ATTEMPTS, feedback)

    # 재시도 후에도 남은 문제: 통과한 후보의 문장은 살리고 나머지와 요약은 폴백
    return HoldProse(summary=fallback.summary if bad_summary else prose.summary, companies=companies)
