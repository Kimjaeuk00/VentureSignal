"""
보고서 조립 — ReportContext 와 문장(prose)으로 Document 를 만든다.

표는 State 값을 옮겨 채우고(결정적) 값이 없으면 MISSING 이다. 문장(요약·항목별 근거·판단 이유·재검토 조건)은
prose 로 받는다. 마크다운과 PDF 는 여기서 만든 Document 를 각자 변환한다.

섹션을 문서 순서대로 만들어야 출처 번호([1], [2] …)가 읽는 순서를 따른다.
"""

import re
from datetime import date

from core.config import SCORECARD_WEIGHTS

from .collect import (
    ITEM_LABEL, ITEM_ORDER, MISSING, REFERENCE_ITEMS, CandidateData, ReportContext, get, missing_labels,
)
from .document import Document, Paragraph, Section, SubHeading, Table
from .fallback import SCORE_MAX, item_rationale, loss_ranking, risks_of
from .references import SourceRegistry
from .schemas import HoldProse, InvestProse
from .textutil import brief, bullets, dedupe, first_sentence, join, sentences

AUTHOR = "VentureSignal"
FOUNDED_RE = re.compile(r"(\d{4})년\s*설립")
FUNDING_RE = re.compile(r"Series\s*[A-Z]|시리즈\s*[A-Z]|투자\s*유치|라운드|펀딩")
S_ID_RE = re.compile(r"S\d{3}")
CITED_RE = re.compile(r"\[(\d+(?:,\s*\d+)*)\]")
MAX_SOURCES = 4  # 한 칸에 적는 출처 번호 수 (본문에 인용이 없을 때)
BULLET = 120  # 표 안 글머리 한 줄의 길이


DEGRADED_TEXT = {
    "insufficient_evidence": "근거 부족", "unverified": "근거 미검증", "no_sources": "확인 가능한 자료 없음", "partial": "일부만 확인",
}


def _meta(ctx: ReportContext, extra: list[tuple[str, str]] = ()) -> list[tuple[str, str]]:
    return [("작성일", date.today().isoformat()), ("작성자", AUTHOR), ("기준일", ctx.as_of), *extra]


def _row(reg: SourceRegistry, label: str, text: str, sources=()) -> list[str]:
    """[항목, 내용, 출처]. 출처 칸에는 그 칸의 본문이 인용한 번호만 적는다(없을 때만 근거 출처를 최대 MAX_SOURCES개)."""
    body = reg.rewrite(text) or MISSING
    cited = sorted({int(n) for m in CITED_RE.finditer(body) for n in re.findall(r"\d+", m.group(1))})
    if cited:
        return [label, body, f"[{', '.join(map(str, cited))}]"]
    return [label, body, reg.pick(body, sources, MAX_SOURCES)]


def _finding(details: dict, key: str) -> tuple[str, list[str]]:
    f = get(details, key, default={})
    if not isinstance(f, dict):
        return "", []
    return f.get("content") or "", f.get("source_ids") or []


def _score(cand: CandidateData, key: str) -> str:
    v = cand.score(key)
    return f"{v:g}" if v is not None else MISSING


def _core_note(cand: CandidateData, named: bool = True) -> str:
    """총점(핵심 점수)이 어떻게 계산됐는지: 참고 항목(투자조건)은 총점에서 제외하고 필수 5항목만 100점으로 환산한다."""
    coverage = cand.scorecard.get("coverage")
    if not isinstance(coverage, (int, float)) or _total(cand) == MISSING:
        return ""
    reference = ", ".join(ITEM_LABEL[k] for k in REFERENCE_ITEMS)
    return (f"{cand.company_name + ': ' if named else ''}{reference}은 판정에 포함하지 않고 참고 정보로만 표시했다"
            f"(웹 조사로는 현금·소진액 등이 확인되지 않는 경우가 많다). "
            f"총점은 나머지 항목의 비중 합({coverage * 100:g}%)을 100점으로 환산한 핵심 점수이다.")


def _hold_type(cand: CandidateData) -> str:
    """보류의 유형: 근거 부족 / 점수 미달 / 채점 실패. 임시 하한값은 확정 전이라 숫자를 적지 않는다."""
    basis = cand.decision_basis
    kind = basis.get("hold_reason")
    if kind == "insufficient_evidence":
        missing = ", ".join(basis.get("unscored_required") or [])
        return f"근거 부족 — 필수 항목({missing or MISSING})이 채점되지 않아 핵심 점수를 산출하지 않았다."
    if kind == "score_below_bound":
        return "점수 미달 — 핵심 점수가 판정 기준 이하이다."
    if kind == "scoring_failed":
        return "채점 실패 — 이 후보는 채점하지 못해 보류로 처리했다."
    return ""


def _total(cand: CandidateData) -> str:
    v = cand.scorecard.get("total_score")
    return f"{v:g}" if isinstance(v, (int, float)) else MISSING


def _own_facts(cand: CandidateData, key: str, cats) -> list[dict]:
    """평가 기업 자신의 사실만 (피어 비교용 사실은 제외)."""
    return [f for f in cand.facts(key, cats) if f.get("subject") in (cand.company_name, None, "")]


def _fact_row(reg: SourceRegistry, cand: CandidateData, key: str, label: str, cats) -> list[str]:
    facts = _own_facts(cand, key, cats)
    return _row(reg, label, bullets((f["text"] for f in facts), limit=3, cap=BULLET), cand.fact_sources(key, facts))


def _limits(cand: CandidateData) -> list[str]:
    """근거 부족·일부 확인 상태인 분석 (섹션 6 '분석의 한계')."""
    out = []
    for key, label in ITEM_ORDER:
        status = cand.status(key)
        if not cand.analysis(key):
            out.append(f"{label} 분석 결과가 없다")
        elif status in DEGRADED_TEXT:
            out.append(f"{label} 분석: {DEGRADED_TEXT[status]}")
    return out


# ============================================================================ 투자 검토 보고서


def _section_summary(c: CandidateData, prose: InvestProse, reg: SourceRegistry) -> Section:
    return Section("1. SUMMARY", [
        Paragraph(f"**최종 판단** 투자 검토 · **총점** {_total(c)} / 100"),
        Paragraph(reg.rewrite(brief(prose.summary, 900))),
    ])


def _section_business(c: CandidateData, reg: SourceRegistry) -> Section:
    tech, market, team = c.details("technology"), c.details("market"), c.details("team")
    customers = get(market, "target_customers", default=[]) or []
    core, core_src = _finding(tech, "core_technology")
    _, stage_src = _finding(tech, "product_stage")
    founded = FOUNDED_RE.search(c.description or "")
    product = get(tech, "product", default="") if not c.is_degraded("technology") else ""

    customer = brief(customers[0], 90) if customers else ""  # 같은 고객 문장이 여러 칸에 반복되지 않게 첫 항목만 짧게 쓴다
    concept = f"해결: {brief(core, 130)}" if core else (f"고객: {customer}" if customer else "")

    members = get(team, "members", default=[]) or []
    people, people_src = [], []
    for m in members:
        names = {m.get("name"), *(m.get("aliases") or [])}
        career = [f for f in c.facts("team", ["career"]) if f.get("subject") in names]
        people_src += c.fact_sources("team", career)
        people.append(f"{m.get('name', MISSING)} / {m.get('role') or MISSING} / {join((f['text'] for f in career), '; ', 2, 80) or MISSING}")
    skills = c.facts("team", ["expertise", "skill", "execution", "publication"])

    rows = [
        _row(reg, "기업명·설립연도", f"{c.company_name} / {founded.group(1) + '년' if founded else MISSING}",
             S_ID_RE.findall(c.description)),
        _row(reg, "핵심 컨셉", concept, core_src),
        _row(reg, "제품·고객", f"제품: {brief(product, 60) or MISSING} / 고객군: {customer or MISSING}", [*core_src, *stage_src]),
        _row(reg, "수익모델", brief(get(market, "business_model", default=""), 160), c.evidence("market")),
        _row(reg, "창업자·핵심 인력", bullets(people, limit=4, cap=BULLET), people_src),
        _row(reg, "팀의 기술 역량", bullets((f["text"] for f in skills), limit=3, cap=BULLET), c.fact_sources("team", skills)),
    ]
    return Section("2. 사업 아이디어와 팀", [Table(["항목", "분석 결과", "출처"], rows, [18, 70, 12])])


def _section_traction(c: CandidateData, reg: SourceRegistry) -> Section:
    runway = get(c.details("deal_terms"), "runway_estimate", default={}) or {}
    money = _own_facts(c, "deal_terms", ["cash", "burn", "runway"])
    parts = [brief(f["text"], BULLET) for f in money][:3]
    if runway.get("months") is not None:
        parts.append(f"기준일 {runway.get('as_of', MISSING)} 단순 추정 런웨이 약 {runway['months']:g}개월 ({runway.get('note', '')})".replace(" ()", ""))
    funding = _fact_row(reg, c, "deal_terms", "투자 이력", ["funding", "terms"])
    if funding[1] == MISSING:  # 웹 조사에 없으면 탐색 단계의 기업 설명(RAG)에 적힌 투자 이력을 쓴다 — 본문(7절)과 어긋나지 않게
        hits = [x for x in sentences(" ".join((c.description or "").split())) if FUNDING_RE.search(x)]
        if hits:
            funding = _row(reg, "투자 이력", bullets(hits, limit=2, cap=BULLET), S_ID_RE.findall(c.description))
    rows = [
        _fact_row(reg, c, "traction", "고객·계약", ["customer", "commercialization", "milestone"]),
        _fact_row(reg, c, "traction", "매출", ["revenue"]),
        funding,
        _row(reg, "자금 상황", bullets(parts, limit=4), c.fact_sources("deal_terms", money)),
    ]
    return Section("3. 사업 실적과 투자조건", [Table(["항목", "확인 결과", "출처"], rows, [18, 70, 12])])


def _performance(tech: dict) -> str:
    lines = []
    for p in get(tech, "performance", default=[]) or []:
        cond = p.get("condition")
        lines.append(f"{p.get('metric', '')}: {p.get('value', '')} {p.get('unit', '')}".strip() + (f" ({cond})" if cond and cond != MISSING else ""))
    return bullets(lines, limit=6, cap=140)


def _section_market(c: CandidateData, reg: SourceRegistry) -> Section:
    m, comp, tech = c.details("market"), c.details("competition"), c.details("technology")
    customers = get(m, "target_customers", default=[]) or []
    market_src = c.evidence("market")
    barriers = bullets(get(m, "adoption_barriers", default=[]) or [], limit=3, cap=BULLET)
    market_rows = [
        _row(reg, "시장 범위", bullets(customers[1:] or customers, limit=3, cap=BULLET), market_src),
        _row(reg, "시장 규모", brief(get(m, "market_size", default=""), 300), market_src),
        _row(reg, "시장 성장률", bullets(get(m, "growth_drivers", default=[]) or [], limit=3, cap=BULLET), market_src),
        _row(reg, "목표 고객·수요", f"도입 장벽\n{barriers}" if barriers else "", market_src),
    ]

    peers = (get(comp, "peer_products", default=[]) or [])[:3]
    product = get(tech, "product", default="") if not c.is_degraded("technology") else ""
    strengths = get(c.analysis("technology"), "strengths", default=[]) or []
    tech_risks = get(c.analysis("technology"), "risks", default=[]) or []
    sw = f"강점: {brief(strengths[0], 100)}" if strengths else ""
    wk = f"약점: {brief(tech_risks[0], 100)}" if tech_risks else ""
    headers = ["비교 항목", c.company_name, *[(p.get("company_name") or MISSING).split(" (")[0] for p in peers]]

    def row(label, mine, key, cap=130):
        return [label, reg.rewrite(mine) or MISSING, *[reg.rewrite(brief(p.get(key) or "", cap)) or MISSING for p in peers]]

    compare = [
        row("기업·제품명", f"{c.company_name} / {product or MISSING}", "product_name"),
        row("용도·고객", brief(customers[0], 90) if customers else "", "customer_and_use"),
        row("성능·전력", _performance(tech), "performance_and_power", 160),
        row("가격·도입 조건", "", "price_and_adoption", 110),
        row("강점·약점", "\n".join(p for p in (sw, wk) if p), "strengths_and_weaknesses"),
        ["출처", reg.cite([*c.evidence("technology")[:3]]), *[reg.cite([p["source_url"]] if p.get("source_url") else []) for p in peers]],
    ]
    # 경쟁사 제품명은 '기업 / 제품' 형태로 맞춘다
    compare[0][2:] = [f"{p.get('company_name') or MISSING} / {p.get('product_name') or MISSING}" for p in peers]
    conditions = join(get(comp, "comparison_conditions", default=[]) or [], "; ", 4, 110)
    blocks = [Table(["시장 지표", "분석 결과", "출처"], market_rows, [18, 70, 12]),
              SubHeading("경쟁 제품 비교"),
              Table(headers, compare, [14, *[86 // (len(headers) - 1)] * (len(headers) - 1)]) if peers else Paragraph(f"경쟁 제품 비교: {MISSING} (경쟁사 제품 정보를 확보하지 못했다)"),
              Paragraph(f"**비교 조건** {reg.rewrite(conditions) or MISSING}")]
    return Section("4. 시장 규모와 경쟁 환경", blocks)


def _section_tech(c: CandidateData, reg: SourceRegistry) -> Section:
    tech = c.details("technology")
    perf_src = [s for p in (get(tech, "performance", default=[]) or []) for s in (p.get("source_ids") or [])]
    limits, limits_src = _finding(tech, "validation_limits")
    unverified = get(tech, "unverified", default=[]) or []
    rows = []
    for label, key in (("핵심 기술", "core_technology"), ("성능·전력", None), ("제품 완성도", "product_stage"),
                       ("사용 환경", "software_environment"), ("검증·한계", "validation_limits")):
        if key is None:
            rows.append(_row(reg, label, _performance(tech), perf_src))
        elif key == "validation_limits":
            text = "\n".join(p for p in (brief(limits, 170), f"미확인: {join((u.rstrip('.。 ') for u in unverified), '; ', 3, 45)}" if unverified else "") if p)
            rows.append(_row(reg, label, text, limits_src))
        else:
            text, src = _finding(tech, key)
            rows.append(_row(reg, label, brief(text, 220), src))
    blocks: list = [Table(["확인 항목", "분석 결과", "출처"], rows, [18, 70, 12])]
    if c.is_degraded("technology"):
        blocks.append(Paragraph(f"※ 기술 분석은 {DEGRADED_TEXT.get(c.status('technology'), '결과 없음')}으로 값을 채우지 못했다."))
    return Section("5. 기술력", blocks)


def _section_risks(c: CandidateData, reg: SourceRegistry) -> Section:
    def lst(*items, limit=4, cap=BULLET):
        return reg.rewrite(bullets(dedupe(i for group in items for i in group), limit=limit, cap=cap)) or MISSING

    def ana(key, name):
        return get(c.analysis(key), name, default=[]) or []

    def det(key, name):
        return get(c.details(key), name, default=[]) or []

    limits = _limits(c)
    rows = [
        ["시장·경쟁", lst(ana("market", "risks"), det("market", "adoption_barriers"), ana("competition", "risks"), det("competition", "critical_risks")),
         lst(det("market", "missing_information"), det("competition", "missing_information"), cap=90)],
        ["기술·규제", lst(ana("technology", "risks"), limit=3), lst(det("technology", "unverified"), cap=70)],
        ["팀·자금", lst(ana("team", "risks"), ana("traction", "risks"), ana("deal_terms", "risks")),
         lst(missing_labels(c, ["team", "traction", "deal_terms"]), limit=6, cap=40)],
        ["분석의 한계", lst(limits) if limits else MISSING, "자료를 보완해 재분석"],
    ]
    return Section("6. 사업 리스크와 한계", [Table(["구분", "주요 위험 또는 한계", "추가 확인 사항"], rows, [16, 54, 30])])


def _section_scorecard(c: CandidateData, prose: InvestProse, reg: SourceRegistry) -> Section:
    rows = []
    for key, label in ITEM_ORDER:
        rationale = getattr(prose.item_rationales, key, "") or MISSING
        weight = "참고" if key in REFERENCE_ITEMS else f"{SCORECARD_WEIGHTS[key] * 100:g}%"
        rows.append([f"{label}(참고)" if key in REFERENCE_ITEMS else label, weight, _score(c, key),
                     f"{reg.rewrite(brief(rationale, 220))} {reg.cite((get(c.assessed(key), 'evidence', default=[]) or c.evidence(key))[:3])}".strip()])
    rows.append(["**핵심 점수**", f"**{sum(w for k, w in SCORECARD_WEIGHTS.items() if k not in REFERENCE_ITEMS) * 100:g}%**", f"**{_total(c)} / 100**", ""])
    note = _core_note(c, named=False)
    return Section("7. 종합 평가와 투자 판단", [
        Table(["평가 항목", "비중", "점수 0~5", "핵심 근거·출처"], rows, [16, 9, 13, 62]),
        *([Paragraph(f"※ {note}")] if note else []),
        Paragraph(f"**최종 판단** 투자 검토 · **판단 이유** {reg.rewrite(brief(prose.decision_reason, 420))}"),
        Paragraph(f"**재검토 조건** {reg.rewrite(brief(prose.revisit_conditions, 420))}"),
    ])


def _section_reference(reg: SourceRegistry, number: str = "8") -> Section:
    lines = reg.references()
    return Section(f"{number}. REFERENCE", [Paragraph(line) for line in lines] or [Paragraph("본문에서 인용한 출처가 없다.")])


def build_invest_document(ctx: ReportContext, prose: InvestProse, reg: SourceRegistry) -> Document:
    c = ctx.selected
    c_unevaluated = ctx.unevaluated
    sections = [_section_summary(c, prose, reg), _section_business(c, reg), _section_traction(c, reg),
                _section_market(c, reg), _section_tech(c, reg)]
    risks = _section_risks(c, reg)
    if c_unevaluated:  # 섹션 6 '분석의 한계'에 평가하지 않은 후보를 덧붙인다
        table = risks.blocks[0]
        note = f"평가하지 않은 후보: {', '.join(c_unevaluated)} (선정 후보가 먼저 확정되어 평가를 종료)"
        table.rows[3][1] = note if table.rows[3][1] == MISSING else f"{table.rows[3][1]}\n• {note}"
    sections += [risks, _section_scorecard(c, prose, reg)]
    sections.append(_section_reference(reg))
    return Document(
        title=f"{c.company_name} 투자 검토 보고서",
        meta=_meta(ctx),
        sections=sections,
        kind="invest",
        file_stem=f"{date.today():%Y%m%d}_{c.company_id}_투자검토보고서",
    )


# ============================================================================ 보류 보고서


def build_hold_document(ctx: ReportContext, prose: HoldProse, reg: SourceRegistry) -> Document:
    by_id = {p.company_id: p for p in prose.companies}
    n = len(ctx.candidates)

    overview = Section("1. SUMMARY", [
        Paragraph(f"**최종 판단** 보류 · 평가한 후보 {n}곳 모두 보류"),
        Paragraph(reg.rewrite(prose.summary)),
    ])

    headers = ["기업", "총점", *[f"{label} ({'참고' if key in REFERENCE_ITEMS else f'{SCORECARD_WEIGHTS[key] * 100:g}%'})" for key, label in ITEM_ORDER]]
    score_rows = [[c.company_name, f"{_total(c)} / 100", *[_score(c, key) for key, _ in ITEM_ORDER]] for c in ctx.candidates]
    notes = dedupe(n for n in (_core_note(c, named=False) for c in ctx.candidates) if n)  # 후보마다 같은 설명이라 한 번만
    scores = Section("2. 후보별 평가 결과", [
        Table(headers, score_rows),
        Paragraph(f"점수는 항목별 0~{SCORE_MAX}점이다."),
        *[Paragraph(f"※ {n}") for n in notes],
    ])

    reasons = Section("3. 후보별 보류 사유")
    for c in ctx.candidates:
        cp = by_id.get(c.company_id)
        rows = []
        for key, loss in loss_ranking(c)[:3]:
            risk = (get(c.analysis(key), "risks", default=[]) or [""])[0]
            text = "\n".join(p for p in (item_rationale(c, key), f"위험: {first_sentence(risk)}" if risk else "") if p)
            rows.append([ITEM_LABEL[key], _score(c, key), reg.rewrite(text), reg.cite(c.evidence(key)[:3])])
        reasons.blocks.append(SubHeading(f"{c.company_name} — 총점 {_total(c)} / 100"))
        if rows:
            reasons.blocks.append(Table(["항목", "점수 0~5", "사유·근거", "출처"], rows, [16, 10, 62, 12]))
        else:
            reasons.blocks.append(Paragraph(f"점수 정보를 확보하지 못했다 ({MISSING})."))
        kind = _hold_type(c)
        if kind:
            reasons.blocks.append(Paragraph(f"**보류 유형** {kind}"))
        reasons.blocks.append(Paragraph(f"**보류 사유** {reg.rewrite(cp.reasons) if cp else MISSING}"))
        reasons.blocks.append(Paragraph(f"**확인하지 못한 항목** {join(missing_labels(c, [k for k, _ in ITEM_ORDER]), ', ', limit=8) or MISSING}"))
        reasons.blocks.append(Paragraph(f"**재검토 조건** {reg.rewrite(cp.revisit_conditions) if cp else MISSING}"))

    limit_rows = []
    for c in ctx.candidates:
        limits = _limits(c)
        limit_rows.append([c.company_name, reg.rewrite(bullets(limits)) if limits else "특이사항 없음"])
    limits_section = Section("4. 분석의 한계", [Table(["후보", "분석의 한계"], limit_rows, [22, 78])])
    if ctx.unevaluated:
        limits_section.blocks.append(Paragraph(f"평가하지 않은 후보: {', '.join(ctx.unevaluated)}"))

    return Document(
        title="투자 검토 결과 보고서 (전 후보 보류)",
        meta=_meta(ctx, [("질의", ctx.query)]),
        sections=[overview, scores, reasons, limits_section, _section_reference(reg, "5")],
        kind="hold",
        file_stem=f"{date.today():%Y%m%d}_보류_투자검토보고서",
    )
