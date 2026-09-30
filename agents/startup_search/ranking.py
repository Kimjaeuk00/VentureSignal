"""
랭킹 — 검색 결과에 조건 필터와 가점을 적용해 최종 순위와 retrieval_score(0~1)를 만든다.

하드 필터 (조건에 걸리면 후보에서 제외)
- exclude_categories        : 질의가 명시적으로 뺀 분야
- exclude_listed_acquired   : 비상장·독립을 요구한 질의에서만 상장·인수 회사 제외 (기본은 유지)
- late_stage                : 후기 성장(Series C 이상·pre-IPO)을 요구한 질의에서 해당 회사만 남김

점수 (하드 필터가 아니라 가점·감점)
- 검색 점수(RRF)를 최대값 기준 0~1 로 정규화한 값이 기본 점수
- include_categories 에 속하면 가점. category 를 하드 필터로 쓰지 않는 이유: 질의가 찾는 분야 밖에도
  정답이 있을 수 있다 (예: RISC-V 질의의 자동차 분야 회사)
- 탐색 태그가 질의 키워드와 겹치면 가점

가중치는 WEIGHTS 한 곳에 모아 평가 결과(T7)로 조정하고 DEVLOG 에 기록한다. 회사명은 쓰지 않는다.
"""

import math
import re
from functools import lru_cache
from typing import Optional, TypedDict

from langsmith import traceable

from .catalog import CompanyProfile, get_catalog
from .retrieval import Hit
from .schemas import SearchConditions

WEIGHTS = {
    "category": 1.5,  # include_categories 일치. 1.2~5.0 에서 결과가 같아(T7 실험) 값에 민감하지 않은 구간
    "tag": 0.1,  # 탐색 태그 키워드 일치 (최대)
    "lexical": 0.6,  # 질의의 드문 기술 용어가 회사 본문에 나오는 정도 (idf 가중)
    "funding": 0.15,  # 사업 근거에 투자 유치 표현이 있음 (투자 후보 탐색이라는 목적에 대한 약한 사전 신호)
}

# 태그 매칭에서 뺄 흔한 단어 (거의 모든 회사에 해당해 변별력이 없다)
STOPWORDS = {
    "ai", "chip", "chips", "company", "companies", "칩", "회사", "기업", "반도체", "스타트업",
    "찾아줘", "관련", "기반", "이나", "또는", "의", "를", "을", "에", "가", "이", "은", "는", "및", "등",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-]*|\d+(?:\.\d+)?\s*W|[가-힣]{2,}", re.IGNORECASE)


class Scored(TypedDict):
    company_id: str
    company_name: str
    retrieval_score: float  # 0~1
    reasons: list[str]  # 가점·감점 근거 (디버깅·평가용)


def _tokens(text: str) -> set[str]:
    out = set()
    for tok in TOKEN_RE.findall(text):
        tok = re.sub(r"\s+", "", tok).lower()
        if tok not in STOPWORDS:
            out.add(tok)
    return out


def _tag_bonus(profile: CompanyProfile, query_tokens: set[str]) -> tuple[float, Optional[str]]:
    """태그 하나라도 질의 키워드와 겹치면 가점. 겹치는 태그 비율에 비례한다."""
    if not profile["tags"] or not query_tokens:
        return 0.0, None
    matched = [t for t in profile["tags"] if _tokens(t) & query_tokens]
    if not matched:
        return 0.0, None
    return WEIGHTS["tag"] * len(matched) / len(profile["tags"]), f"tag:{matched[0]}"


@lru_cache
def _document_frequencies() -> tuple[int, dict[str, str]]:
    """회사 수와 회사별 소문자 본문 (lexical 점수용). 카탈로그가 바뀌지 않으므로 1회만 만든다."""
    catalog = get_catalog()
    return len(catalog), {cid: p["tech_text"].lower() for cid, p in catalog.items()}


def _lexical_bonus(profile: CompanyProfile, query_tokens: set[str]) -> tuple[float, Optional[str]]:
    """
    질의 토큰이 회사 본문에 부분 문자열로 나오면 idf 가중으로 가점한다 (한국어 조사 변형을 흡수하려고 부분 일치).
    모든 회사에 나오는 토큰은 idf 0, 어느 회사에도 없는 토큰은 건너뛴다.
    """
    n, texts = _document_frequencies()
    total = matched = 0.0
    hits = []
    for tok in query_tokens:
        if len(tok) < 2:
            continue
        df = sum(tok in t for t in texts.values())
        if df == 0:
            continue
        idf = math.log(n / df)
        total += idf
        if tok in texts[profile["company_id"]]:
            matched += idf
            hits.append(tok)
    if total <= 0 or matched <= 0:
        return 0.0, None
    return WEIGHTS["lexical"] * matched / total, f"lexical:{','.join(sorted(hits))[:40]}"


def _passes_hard_filters(profile: CompanyProfile, cond: SearchConditions) -> bool:
    if profile["category"] in cond.exclude_categories:
        return False
    if cond.exclude_listed_acquired and profile["status_kind"] in ("listed", "acquired"):
        return False
    if cond.late_stage and not profile["late_stage"]:
        return False
    return True


def _trace_inputs(inputs: dict) -> dict:
    """catalog 는 50개사 본문 전체라 LangSmith 로 보내지 않는다."""
    return {k: v for k, v in inputs.items() if k != "catalog"}


@traceable(name="rerank", run_type="chain", process_inputs=_trace_inputs)
def rerank(
    hits: list[Hit],
    conditions: SearchConditions,
    catalog: Optional[dict[str, CompanyProfile]] = None,
    query: Optional[str] = None,
) -> list[Scored]:
    """하드 필터 → 점수 계산 → 0~1 정규화 → 내림차순 정렬. 필터를 통과한 회사가 없으면 빈 목록."""
    catalog = catalog or get_catalog()
    query_tokens = _tokens(f"{query or ''} {conditions.semantic_query}")
    max_hit_score = max((h["score"] for h in hits), default=0.0) or 1.0

    scored: list[tuple[float, int, Scored]] = []
    for h in hits:
        profile = catalog.get(h["company_id"])
        if profile is None or not _passes_hard_filters(profile, conditions):
            continue

        raw = h["score"] / max_hit_score
        reasons: list[str] = []
        if profile["category"] in conditions.include_categories:
            raw += WEIGHTS["category"]
            reasons.append(f"category:{profile['category']}")
        bonus, why = _tag_bonus(profile, query_tokens)
        if why:
            raw += bonus
            reasons.append(why)
        if WEIGHTS["lexical"]:
            bonus, why = _lexical_bonus(profile, query_tokens)
            if why:
                raw += bonus
                reasons.append(why)
        if WEIGHTS["funding"] and profile["funding_evidence"]:
            raw += WEIGHTS["funding"]
            reasons.append("funding")

        scored.append((raw, h["rank"], {
            "company_id": profile["company_id"],
            "company_name": profile["company_name"],
            "retrieval_score": 0.0,
            "reasons": reasons,
        }))

    if not scored:
        return []
    top = max(raw for raw, _, _ in scored)
    scored.sort(key=lambda x: (-x[0], x[1]))  # 동점이면 검색 순위가 높은 쪽
    out = []
    for raw, _, item in scored:
        item["retrieval_score"] = round(max(0.0, raw) / top, 4) if top > 0 else 0.0
        out.append(item)
    return out
