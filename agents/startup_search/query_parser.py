"""
질의 → SearchConditions.

1순위는 LLM 구조화 출력, 실패하면(API 키 없음·호출 오류·형식 오류) 규칙 기반 폴백을 쓴다.
category 값은 카탈로그(PDF 분류 체계)의 목록으로 검증한다. 회사명은 쓰지 않는다.
"""

import re

from langsmith import traceable

from core.llm import get_llm

from .catalog import get_categories
from prompts.startup_search import CONDITION_SYSTEM_PROMPT
from .schemas import SearchConditions

# 연산 칩 category (질의가 "NPU 제외"처럼 연산 칩 전반을 뺄 때)
COMPUTE_CATEGORIES = ["데이터센터 AI 계산", "엣지 AI·영상 계산", "AI·HPC GPU"]

# 규칙 폴백용 분야 키워드 (PDF category 와 탐색 태그의 일반 용어. 회사명 아님)
CATEGORY_KEYWORDS: list[tuple[str, str]] = [
    ("AI 반도체 IP", r"\bIP\b|라이선스|toolchain|통합 스택|통합 설계"),
    ("데이터센터 AI 계산", r"데이터센터|datacenter|HBM|LLM inference|LLM 추론|decode|inference accelerator"),
    ("엣지 AI·영상 계산", r"엣지|edge|CCTV|머신비전|M\.2|온디바이스|로봇|robot|smart camera"),
    ("상시 동작 센서 AI", r"always-on|항상|웨어러블|wearable|earbud|keyword spotting|acoustic|센서"),
    ("뉴로모픽·희소 계산", r"뉴로모픽|neuromorphic|\bSNN\b|spiking|brain-inspired|event-driven|sparse inference"),
    ("메모리·연산 결합", r"Compute-in-Memory|\bCIM\b|in-memory|메모리와 연산|memory wall|3D memory"),
    ("광전자 계산", r"광연산|photonic|optical compute|빛을|광전자"),
    ("광연결 인프라", r"광연결|photonic interconnect|optical interconnect"),
    ("RISC-V AI·HPC", r"RISC-V|open ISA"),
    ("연결·메모리 인프라", r"interconnect|memory fabric|memory pooling|\bCXL\b|SuperNIC|연결.{0,4}병목"),
]

EXCLUDE_LISTED_RE = re.compile(r"비상장|독립|상장사?.{0,6}(빼|제외|말고)|인수되지 않은")
LATE_STAGE_RE = re.compile(r"Series\s*[C-Z]\b|후기 (성장|투자)|pre-IPO", re.IGNORECASE)
EXCLUDE_IP_RE = re.compile(r"\bIP\b.{0,8}(제외|빼|말고)|(제외|빼).{0,4}\bIP\b")
EXCLUDE_NPU_RE = re.compile(r"NPU.{0,6}(제외|빼|말고)")
CARD_EXCLUDE_RE = re.compile(r"완성형.{0,10}(말고|제외)")


def parse_with_rules(query: str) -> SearchConditions:
    """LLM 없이 핵심 키워드만으로 조건을 만든다."""
    exclude: list[str] = []
    if EXCLUDE_IP_RE.search(query):
        exclude.append("AI 반도체 IP")
    if EXCLUDE_NPU_RE.search(query):
        exclude.extend(COMPUTE_CATEGORIES)

    include = [
        cat for cat, pattern in CATEGORY_KEYWORDS
        if cat not in exclude and re.search(pattern, query, re.IGNORECASE)
    ]
    # 연산 칩을 뺀다고 했으면 그 분야는 찾는 분야에서도 뺀다
    include = [c for c in include if c not in exclude]

    return SearchConditions(
        region="국내" if "국내" in query and "국내외" not in query
        else "해외" if "해외" in query else None,
        include_categories=include,
        exclude_categories=exclude,
        late_stage=bool(LATE_STAGE_RE.search(query)),
        exclude_listed_acquired=bool(EXCLUDE_LISTED_RE.search(query)),
        semantic_query=query,
    )


def _validate(conditions: SearchConditions, query: str) -> SearchConditions:
    """category 는 카탈로그 목록 값만 남기고, 비어 있는 semantic_query 는 원 질의로 채운다."""
    valid = set(get_categories())
    conditions.include_categories = [c for c in conditions.include_categories if c in valid]
    conditions.exclude_categories = [c for c in conditions.exclude_categories if c in valid]
    conditions.include_categories = [
        c for c in conditions.include_categories if c not in conditions.exclude_categories
    ]
    if not conditions.semantic_query.strip():
        conditions.semantic_query = query
    return conditions


def parse_with_llm(query: str) -> SearchConditions:
    system = CONDITION_SYSTEM_PROMPT.format(categories="\n".join(f"- {c}" for c in get_categories()))
    result = get_llm().with_structured_output(SearchConditions).invoke(
        [("system", system), ("human", query)]
    )
    return _validate(result, query)


@traceable(name="parse_query", run_type="chain")
def parse_query(query: str) -> SearchConditions:
    """LLM 으로 조건을 뽑고, 어떤 이유로든 실패하면 규칙 기반 폴백을 쓴다."""
    try:
        return parse_with_llm(query)
    except Exception:
        return _validate(parse_with_rules(query), query)
