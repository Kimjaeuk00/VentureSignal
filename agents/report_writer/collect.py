"""
State → ReportContext.

팀원 노드가 채운 evaluations/details 의 키를 읽는 곳은 이 파일 하나다. 키가 없거나 근거 부족 상태면
값을 만들지 않고 MISSING("미확인")으로 둔다. 다른 파일은 여기서 만든 값만 쓴다.
"""

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Iterable, Literal, Optional

from core.state import GraphState

MISSING = "미확인"
ANALYSIS_KEYS = ("technology", "market", "competition", "team", "traction", "deal_terms")
DEGRADED = {"insufficient_evidence", "unverified", "no_sources"}  # 분석이 근거를 확보하지 못한 상태
CATEGORY_LABELS = {
    "career": "경력", "education": "학력", "skill": "기술 역량", "expertise": "전문성", "publication": "논문·발표",
    "unverified_publication": "논문·발표", "execution": "실행 이력", "linkedin_activity": "활동 이력",
    "customer": "고객", "revenue": "매출", "commercialization": "사업화", "funding": "투자 이력",
    "terms": "투자 조건", "cash": "보유 현금", "burn": "월 순소진액", "runway": "런웨이", "milestone": "마일스톤",
}


def get(data: Any, *path: str, default: Any = None) -> Any:
    """중첩 dict 를 안전하게 읽는다. 중간에 dict 가 아니거나 값이 비어 있으면 default."""
    for key in path:
        if not isinstance(data, dict):
            return default
        data = data.get(key)
    return default if data in (None, "", [], {}) else data


@dataclass
class CandidateData:
    company_id: str
    company_name: str
    description: str
    domain: str
    analyses: dict[str, dict]  # ANALYSIS_KEYS → AnalysisResult (없으면 {})
    scorecard: dict
    decision: Optional[str]
    assessment: dict = field(default_factory=dict)  # 투자 판단 노드의 항목별 {score, rationale, evidence, missing_information}
    decision_basis: dict = field(default_factory=dict)  # 판정 근거: hold_reason(insufficient_evidence/score_below_bound/scoring_failed) 등

    def assessed(self, key: str) -> dict:
        return self.assessment.get(key) or {}

    def score(self, key: str) -> Optional[float]:
        """항목 점수. scorecard(총점 계산에 쓴 점수)가 없으면(미산정 항목이 있어 총점을 못 낸 경우) 투자 판단의 항목별 채점을 쓴다."""
        for value in (self.scorecard.get(key), get(self.assessed(key), "score")):
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                return value
        return None

    def analysis(self, key: str) -> dict:
        return self.analyses.get(key) or {}

    def details(self, key: str) -> dict:
        return get(self.analysis(key), "details", default={}) or {}

    def status(self, key: str) -> Optional[str]:
        return get(self.details(key), "status")

    def is_degraded(self, key: str) -> bool:
        return not self.analysis(key) or self.status(key) in DEGRADED

    def facts(self, key: str, categories: Iterable[str]) -> list[dict]:
        wanted = set(categories)
        return [f for f in (get(self.details(key), "facts", default=[]) or []) if f.get("category") in wanted]

    def source_urls(self, key: str) -> dict[str, str]:
        """founder_traction 의 내부 출처 ID(S-해시) → URL."""
        return {s["source_id"]: s["url"] for s in (get(self.details(key), "sources", default=[]) or [])
                if s.get("source_id") and s.get("url")}

    def fact_sources(self, key: str, facts: Iterable[dict]) -> list[str]:
        """facts 의 근거를 URL(또는 RAG 출처 ID)로 풀어 돌려준다."""
        mapping = self.source_urls(key)
        out: list[str] = []
        for fact in facts:
            for ev in fact.get("evidence") or []:
                sid = ev.get("source_id")
                if sid:
                    out.append(mapping.get(sid, sid))
        return list(dict.fromkeys(out))

    def evidence(self, key: str) -> list[str]:
        return list(dict.fromkeys(get(self.analysis(key), "evidence", default=[]) or []))


@dataclass
class ReportContext:
    kind: Literal["invest", "hold", "none"]
    query: str
    as_of: str
    candidates: list[CandidateData]  # invest: 선정 후보 1곳, hold: 평가한 후보 전부
    unevaluated: list[str] = field(default_factory=list)  # 평가하지 못한 후보의 이름
    url_titles: dict[str, str] = field(default_factory=dict)

    @property
    def selected(self) -> Optional[CandidateData]:
        return self.candidates[0] if self.kind == "invest" and self.candidates else None


def _candidate_data(cid: str, state: GraphState) -> CandidateData:
    info = next((c for c in state.get("candidates", []) if c.get("company_id") == cid), {})
    evaluation = state.get("evaluations", {}).get(cid, {})
    return CandidateData(
        company_id=cid,
        company_name=info.get("company_name") or cid,
        description=info.get("description") or "",
        domain=info.get("domain") or "",
        analyses={k: evaluation.get(k) or {} for k in ANALYSIS_KEYS},
        scorecard=evaluation.get("scorecard") or {},
        decision=evaluation.get("decision"),
        assessment=evaluation.get("assessment") or {},
        decision_basis=evaluation.get("decision_basis") or {},
    )


def _as_of(candidates: list[CandidateData]) -> str:
    """분석 노드가 남긴 조회일 중 가장 늦은 날짜. 없으면 오늘."""
    dates: list[str] = []
    for c in candidates:
        dates.append(get(c.details("technology"), "web_search", "searched_at", default=""))
        dates.extend(get(c.details(k), "as_of", default="") for k in ANALYSIS_KEYS)
    dates = [d for d in dates if isinstance(d, str) and re.match(r"\d{4}-\d{2}-\d{2}$", d)]
    return max(dates) if dates else date.today().isoformat()


def _url_titles(candidates: list[CandidateData], state_sources: Optional[dict] = None) -> dict[str, str]:
    """URL → 제목. State 의 sources(분석 노드들이 남긴 것)와 창업자·실적 분석의 details.sources 를 합친다."""
    titles: dict[str, str] = {
        url: info["title"] for url, info in (state_sources or {}).items()
        if isinstance(info, dict) and info.get("title") and info["title"] != url}
    for c in candidates:
        for key in ANALYSIS_KEYS:
            for src in get(c.details(key), "sources", default=[]) or []:
                if src.get("url") and src.get("title") and src["title"] != src["url"]:
                    titles[src["url"]] = src["title"]
    return titles


def collect(state: GraphState) -> ReportContext:
    selected = state.get("selected_candidate")
    all_ids = [c["company_id"] for c in state.get("candidates", [])]
    evaluated = [cid for cid in all_ids if cid in state.get("evaluations", {})]
    evaluated += [cid for cid in state.get("evaluations", {}) if cid not in evaluated]

    if selected:
        ids, kind = [selected], "invest"
    elif evaluated:
        ids, kind = evaluated, "hold"
    else:
        return ReportContext(kind="none", query=state.get("query", ""), as_of=date.today().isoformat(), candidates=[])

    candidates = [_candidate_data(cid, state) for cid in ids]
    names = {c["company_id"]: c.get("company_name", c["company_id"]) for c in state.get("candidates", [])}
    return ReportContext(
        kind=kind,
        query=state.get("query", ""),
        as_of=_as_of(candidates),
        candidates=candidates,
        # 평가 결과가 없는 후보만 (INVEST 전에 이미 평가하고 보류한 후보는 평가한 후보다)
        unevaluated=[names[cid] for cid in all_ids if cid not in state.get("evaluations", {})],
        url_titles=_url_titles(candidates, state.get("sources")),
    )


# 총점(핵심 점수)에서 제외하고 참고용으로만 보여 주는 항목 — 투자 판단 노드의 REFERENCE_CRITERIA 와 같다
REFERENCE_ITEMS = ("deal_terms",)

# E절 7절의 평가 항목 순서와 표시 이름 (점수 키는 core.state.ScorecardResult 와 같다)
ITEM_ORDER = [
    ("team", "창업자·팀"), ("market", "시장성"), ("technology", "제품·기술력"),
    ("competition", "경쟁 우위"), ("traction", "실적"), ("deal_terms", "투자조건"),
]
ITEM_LABEL = dict(ITEM_ORDER)


def missing_labels(cand: CandidateData, keys: Iterable[str]) -> list[str]:
    """분석이 확인하지 못했다고 남긴 항목(누락 category, unverified, missing_information)."""
    out: list[str] = []
    for key in keys:
        d = cand.details(key)
        out += [CATEGORY_LABELS.get(c, c) for c in get(d, "missing_categories", default=[]) or []]
        out += get(d, "unverified", default=[]) or []
        out += get(d, "missing_information", default=[]) or []
    return list(dict.fromkeys(out))
