"""
스타트업 탐색의 검색 조건 스키마.

질의 파서(query_parser)가 만들고 검색(retrieval)·랭킹(ranking)이 사용한다.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SearchConditions(BaseModel):
    """사용자 질의에서 뽑은 검색 조건. 질의에 없는 조건은 기본값(제한 없음)으로 둔다."""

    region: Optional[Literal["국내", "해외"]] = Field(
        default=None, description="질의가 국내 또는 해외 회사만 요구할 때만 지정"
    )
    include_categories: list[str] = Field(
        default_factory=list,
        description="질의가 찾는 회사 분야. 제공된 category 목록 값만 사용",
    )
    exclude_categories: list[str] = Field(
        default_factory=list,
        description="질의가 명시적으로 제외하라고 한 분야. 제공된 category 목록 값만 사용",
    )
    late_stage: bool = Field(
        default=False,
        description="Series C 이상·후기 성장·pre-IPO 등 후기 투자 단계 회사를 요구하면 True",
    )
    exclude_listed_acquired: bool = Field(
        default=False,
        description="비상장·독립 회사만 요구하거나 상장사·인수된 회사를 빼 달라고 명시했을 때만 True",
    )
    semantic_query: str = Field(
        default="",
        description="검색용 질의. 제외·상장·투자 단계 같은 조건 표현은 빼고 기술·제품 키워드는 유지",
    )
