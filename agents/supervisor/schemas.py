"""Supervisor LLM 구조화 출력. 다음 행동은 실제 노드 이름 중 하나로만 받는다 (이름이 틀리면 분기가 불가능하므로 형식 자체를 제한)."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

# 분석 단계에서 LLM 이 고를 수 있는 노드. 후보 선택·보고서 단계는 판정 결과에 따라 정해지는 전이라 LLM 선택지가 아니다.
NextAgent = Literal["tech_summary", "market_eval", "competitor_compare", "founder_traction", "investment_decision"]


class AgentReview(BaseModel):
    agent: str = Field(description="방금 보고를 낸 에이전트 이름")
    sufficient: bool = Field(description="그 보고가 투자 판단에 쓸 만큼 근거가 충분한가")
    missing: list[str] = Field(default_factory=list, description="부족하다면 비어 있는 것 (짧게, 최대 3개)")


class SupervisorDecision(BaseModel):
    review: Optional[AgentReview] = Field(default=None, description="방금 돌아온 에이전트가 없으면 null")
    next_agent: NextAgent = Field(description="다음에 시킬 노드")
    reason: str = Field(description="그렇게 정한 이유 한두 문장")
    search_queries: list[str] = Field(
        default_factory=list,
        description=(
            "재작업(같은 에이전트를 다시 고를 때)에서만: 비어 있는 항목을 웹에서 찾기 위한 검색어 최대 3개. "
            "회사·제품 이름을 포함한 짧은 키워드로 쓴다. 재작업이 아니면 빈 목록"
        ),
    )
