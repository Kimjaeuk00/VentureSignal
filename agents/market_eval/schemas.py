from pydantic import BaseModel, Field
from typing import Literal

class MarketSearchQuery(BaseModel):
    purpose: Literal["market_size_growth", "customer_adoption"] = Field(
        description=(
            "market_size_growth: 세부 시장 규모·성장 근거, "
            "customer_adoption: 목표 고객·구매 수요·도입 장벽"
        )
    )
    query: str = Field(
        min_length=1,
        description="한 가지 정보 요구에 집중한 짧고 구체적인 웹 검색어"
    )

class MarketDetails(BaseModel):
    target_customers: list[str] = Field(
        description="목표 고객과 고객이 해결하려는 문제"
    )
    market_size: str = Field(
        description="접근 가능한 시장 규모. 수치는 범위·연도·단위·출처를 함께 기록하고, 없으면 미확인"
    )
    growth_drivers: list[str] = Field(
        description="시장 수요의 성장 근거. 기업 매출 성장과 구분"
    )
    business_model: str = Field(
        description="IP 라이선스, 로열티, 칩 판매 등 수익 발생 구조"
    )
    adoption_barriers: list[str] = Field(
        description="고객의 도입을 막는 비용, 검증 기간, 통합 부담 등"
    )
    missing_information: list[str] = Field(
        description="현재 확보한 자료로 확인되지 않은 사항. 웹 검색으로 보완한 뒤에는 여전히 미확인인 사항만 기록"
    )
    web_search_queries: list[MarketSearchQuery] = Field(
        max_length=2,
        description=(
            "RAG 초기 분석에서 필요한 웹 검색 계획. "
            "목적별 최대 1개이며 근거가 충분한 목적은 제외. "
            "시장 규모·성장과 고객 수요·도입을 우선한다. "
            "웹 보완 후 최종 분석에서는 빈 목록을 반환한다."
        )
    )


class MarketAnalysis(BaseModel):
    summary: str = Field(
        description="근거에 기반한 시장성 종합 분석. 공개 사실과 분석을 구분"
    )
    strengths: list[str] = Field(
        description="근거가 있는 시장 기회와 강점. 각 내용에 RAG source_id 또는 웹 출처 URL 표시"
    )
    risks: list[str] = Field(
        description="시장·사업화 위험. 분석과 미확인을 구분하고 근거가 있으면 출처 표시"
    )
    evidence: list[str] = Field(
        description="분석에 실제 사용한 RAG source_id와 웹 검색 결과의 원문 URL 목록. 중복과 사용하지 않은 출처 제외"
    )
    details: MarketDetails