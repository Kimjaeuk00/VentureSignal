"""
기업 카탈로그 — 조사자료 PDF 의 회사별 페이지를 회사 단위 프로필로 정리한다.

인덱스(payload)에는 없는 정보를 PDF 본문 표에서 뽑는다.
- 기업 상태  : "기업 상태" 행 (상장·인수·사업 전환·인접 분야 등)
- 탐색 태그  : "탐색 태그" 행
- late_stage : 본문에 Series C 이상 또는 pre-IPO 표현이 있는 회사 (후기 성장)

이 모듈은 분류만 한다. 상장·인수 회사를 후보에서 뺄지는 질의 조건(exclude_listed_acquired)이 결정한다.
회사명은 하드코딩하지 않는다. 모든 값은 PDF 본문 규칙으로 뽑는다.

rag.ingest.parse_pages 를 import 해서 재사용한다 (rag/ 는 수정하지 않는다).
"""

import re
from functools import lru_cache
from typing import Optional, TypedDict

from rag.ingest import parse_pages

# 기업 상태 행의 앞부분 → 상태 종류
STATUS_KINDS = [
    ("상장 비교 기업", "listed"),
    ("인수 완료 비교 기업", "acquired"),
    ("사업 전환 사례", "pivot"),
    ("인접 분야", "adjacent"),
]

STATUS_RE = re.compile(r"기업 상태\n(.+)")
TAGS_RE = re.compile(r"탐색 태그\n([\s\S]+?)\n근거")
DESCRIPTION_RE = re.compile(r"기업·사업 근거 \| [^\n]*\n([\s\S]+?)\n핵심 기술")

# Series C 이상 또는 pre-IPO
LATE_STAGE_RE = re.compile(r"Series\s*[C-Z]\b|pre-IPO", re.IGNORECASE)

# 투자 유치 근거 (단계·금액 표현). "기업·사업 근거" 문단에서만 찾는다.
FUNDING_EVIDENCE_RE = re.compile(
    r"Series\s*[A-Z]|pre-IPO|투자\s*유치|유치를|유치했|유치\s*발표|억\s*(?:원|달러)|만\s*달러|million",
    re.IGNORECASE,
)


class CompanyProfile(TypedDict):
    company_id: str
    company_name: str
    region: str  # 국내 / 해외
    category: str
    status: str  # "기업 상태" 행 원문
    status_kind: str  # listed / acquired / pivot / adjacent / other
    tags: list[str]
    late_stage: bool
    funding_mentions: list[str]  # late_stage 판정 근거가 된 문장
    funding_evidence: bool  # 사업 근거 문단에 투자 유치 표현이 있는가 (후기 성장 여부와 별개)
    description: str  # "기업·사업 근거" 문단 (출처 번호 포함)
    tech_text: str  # 기술 페이지 본문
    market_text: str  # 시장성·확인 과제 페이지 본문
    source_ids: list[str]
    pages: list[int]  # 이 회사의 PDF 페이지 (기술, 시장성)


def _status_kind(status: str) -> str:
    for prefix, kind in STATUS_KINDS:
        if status.startswith(prefix):
            return kind
    return "other"


def _first(pattern: re.Pattern, text: str) -> Optional[str]:
    m = pattern.search(text)
    return m.group(1) if m else None


def _funding_mentions(text: str, window: int = 45) -> list[str]:
    """late_stage 근거가 되는 표현 주변 원문을 모은다 (표 형태 본문에서도 짧게 유지)."""
    flat = " ".join(text.split("\n"))
    mentions: list[str] = []
    last_end = -1
    for m in LATE_STAGE_RE.finditer(flat):
        if m.start() < last_end:  # 앞선 근거와 겹치면 건너뜀
            continue
        start, last_end = max(0, m.start() - window), m.end() + window
        mentions.append(flat[start:last_end].strip())
    return mentions


def _build_catalog() -> dict[str, CompanyProfile]:
    pages = parse_pages()
    tech = {p["company_id"]: p for p in pages if p["page_type"] == "company_tech"}
    market = {p["company_id"]: p for p in pages if p["page_type"] == "company_market"}

    catalog: dict[str, CompanyProfile] = {}
    for cid, p in tech.items():
        m = market.get(cid)
        text = p["text"]
        both = text + "\n" + (m["text"] if m else "")

        status = _first(STATUS_RE, text) or ""
        raw_tags = _first(TAGS_RE, text) or ""
        raw_desc = _first(DESCRIPTION_RE, text) or ""
        mentions = _funding_mentions(both)

        catalog[cid] = {
            "company_id": cid,
            "company_name": p["company_name"],
            "region": p["region"],
            "category": p["category"],
            "status": status,
            "status_kind": _status_kind(status),
            "tags": [t.strip() for t in " ".join(raw_tags.split("\n")).split(",") if t.strip()],
            "late_stage": bool(mentions),
            "funding_mentions": mentions,
            "funding_evidence": bool(FUNDING_EVIDENCE_RE.search(raw_desc)),
            "description": " ".join(raw_desc.split("\n")),
            "tech_text": text,
            "market_text": m["text"] if m else "",
            "source_ids": sorted(set(p["source_ids"]) | set(m["source_ids"] if m else [])),
            "pages": [p["page"]] + ([m["page"]] if m else []),
        }
    return catalog


@lru_cache
def get_catalog() -> dict[str, CompanyProfile]:
    """company_id → CompanyProfile (50개사). 프로세스당 1회 생성."""
    return _build_catalog()


def get_categories() -> list[str]:
    """PDF 분류 체계의 category 목록 (질의 파서의 보기로 사용)."""
    return sorted({c["category"] for c in get_catalog().values()})


def find_by_name(name: str) -> Optional[CompanyProfile]:
    for c in get_catalog().values():
        if c["company_name"] == name:
            return c
    return None


def main():
    cat = get_catalog()
    print(f"companies: {len(cat)}")
    kinds: dict[str, list[str]] = {}
    for c in cat.values():
        kinds.setdefault(c["status_kind"], []).append(c["company_name"])
    for kind, names in kinds.items():
        print(f"  {kind:<9} {len(names):>2}: {', '.join(names) if kind != 'other' else '...'}")
    late = [c["company_name"] for c in cat.values() if c["late_stage"]]
    print(f"late_stage {len(late)}: {', '.join(late)}")
    print(f"categories {len(get_categories())}: {get_categories()}")


if __name__ == "__main__":
    main()
