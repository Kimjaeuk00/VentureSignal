"""
결정적 문장 생성 — LLM 없이(또는 LLM 출력이 검증에 실패했을 때) 보고서 문장을 만든다.

State 에 있는 요약·강점·위험·미확인 항목을 이어 붙일 뿐 새 사실을 만들지 않는다.
"""

from core.config import SCORECARD_WEIGHTS

from .collect import ITEM_LABEL, ITEM_ORDER, MISSING, CandidateData, ReportContext, get, missing_labels
from .schemas import CompanyHoldProse, HoldProse, InvestProse, ItemRationales
from .textutil import dedupe, first_sentence, join


def _items(items, sep: str = "; ", limit: int = 2) -> str:
    """항목 나열 — 항목 끝의 마침표·물음표는 떼어 문장 부호가 겹치지 않게 한다."""
    return join((i.rstrip(".。?!… ") for i in items), sep, limit)

SCORE_MAX = 5


def item_rationale(cand: CandidateData, key: str) -> str:
    """해당 분석의 요약 첫 문장. 근거 부족이면 미확인."""
    if cand.is_degraded(key):
        return f"{MISSING} (근거 부족)"
    return first_sentence(get(cand.analysis(key), "summary", default="")) or MISSING


def risks_of(cand: CandidateData, keys=None) -> list[str]:
    keys = keys or [k for k, _ in ITEM_ORDER]
    return dedupe(r for k in keys for r in (get(cand.analysis(k), "risks", default=[]) or []))


def strengths_of(cand: CandidateData) -> list[str]:
    return dedupe(s for k, _ in ITEM_ORDER for s in (get(cand.analysis(k), "strengths", default=[]) or []))


def loss_ranking(cand: CandidateData) -> list[tuple[str, float]]:
    """총점 손실 기여도 (만점 대비 부족한 점수 × 비중) 가 큰 항목 순."""
    rows = []
    for key, _ in ITEM_ORDER:
        score = cand.scorecard.get(key)
        if isinstance(score, (int, float)):
            rows.append((key, (SCORE_MAX - score) * SCORECARD_WEIGHTS[key]))
    return sorted(rows, key=lambda r: (-r[1], -SCORECARD_WEIGHTS[r[0]]))


def _score_text(cand: CandidateData, key: str) -> str:
    score = cand.scorecard.get(key)
    return f"{ITEM_LABEL[key]} {score:g}/{SCORE_MAX}" if isinstance(score, (int, float)) else f"{ITEM_LABEL[key]} {MISSING}"


def _revisit(cand: CandidateData) -> str:
    missing = missing_labels(cand, [k for k, _ in ITEM_ORDER])
    if not missing:
        return f"현재 자료에서 추가로 확인이 필요하다고 남은 항목은 {MISSING}이다. 새 자료가 확보되면 재검토한다."
    return f"다음 항목이 확인되면 판단을 재검토한다: {_items(missing, ', ', 6)}."


def _lowest(cand: CandidateData) -> str:
    """실제 점수가 가장 낮은 항목 (가중 손실 기여와 다르다)."""
    scored = [(cand.scorecard[k], k) for k, _ in ITEM_ORDER if isinstance(cand.scorecard.get(k), (int, float))]
    return _score_text(cand, min(scored)[1]) if scored else MISSING


def invest_prose(ctx: ReportContext) -> InvestProse:
    c = ctx.selected
    tech = first_sentence(get(c.analysis("technology"), "summary", default="")) if not c.is_degraded("technology") else ""
    strengths, risks = _items(strengths_of(c)), _items(risks_of(c))
    total = c.scorecard.get("total_score")
    summary = (
        f"{tech or '핵심 제품·기술: ' + MISSING + '.'} "
        f"주요 강점: {strengths or MISSING}. 핵심 위험: {risks or MISSING}. {_revisit(c)}"
    )
    best = sorted(((c.scorecard.get(k, 0), k) for k, _ in ITEM_ORDER), reverse=True)[:2]
    if isinstance(total, (int, float)):
        reason = f"총점 {total:g}점. 점수가 높은 항목: {', '.join(_score_text(c, k) for _, k in best)}. 점수가 가장 낮은 항목: {_lowest(c)}."
    else:
        reason = f"총점: {MISSING}."
    return InvestProse(
        summary=summary,
        item_rationales=ItemRationales(**{k: item_rationale(c, k) for k, _ in ITEM_ORDER}),
        decision_reason=reason,
        revisit_conditions=_revisit(c),
    )


def hold_prose(ctx: ReportContext) -> HoldProse:
    companies, lines = [], []
    for c in ctx.candidates:
        top = [_score_text(c, k) for k, _ in loss_ranking(c)[:3]]
        risk = _items(risks_of(c))
        total = c.scorecard.get("total_score")
        total_text = f"총점 {total:g}점" if isinstance(total, (int, float)) else f"총점 {MISSING}"
        reasons = (
            f"{total_text}. 총점 손실에 가장 크게 기여한 항목: {', '.join(top) if top else MISSING}. "
            f"점수가 가장 낮은 항목: {_lowest(c)}. 핵심 위험: {risk or MISSING}."
        )
        companies.append(CompanyHoldProse(company_id=c.company_id, reasons=reasons, revisit_conditions=_revisit(c)))
        lines.append(f"{c.company_name}({total_text}): 점수가 가장 낮은 항목은 {_lowest(c)}")
    summary = f"평가한 후보 {len(ctx.candidates)}곳이 모두 보류였다. " + "; ".join(lines) + "."
    return HoldProse(summary=summary, companies=companies)
