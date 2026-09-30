"""보고서 문장(prose) 모델 — LLM 구조화 출력과 결정적 폴백이 같은 타입을 쓴다."""

from pydantic import BaseModel, Field


class ItemRationales(BaseModel):
    team: str = Field(description="창업자·팀 점수의 핵심 근거 한 문장")
    market: str = Field(description="시장성 점수의 핵심 근거 한 문장")
    technology: str = Field(description="제품·기술력 점수의 핵심 근거 한 문장")
    competition: str = Field(description="경쟁 우위 점수의 핵심 근거 한 문장")
    traction: str = Field(description="실적 점수의 핵심 근거 한 문장")
    deal_terms: str = Field(description="투자조건 점수의 핵심 근거 한 문장")


class InvestProse(BaseModel):
    summary: str = Field(description="SUMMARY 문단: 고객 문제와 핵심 제품·기술, 주요 근거, 강점, 핵심 위험, 추가 확인 사항")
    item_rationales: ItemRationales
    decision_reason: str = Field(description="투자 검토로 판단한 핵심 이유")
    revisit_conditions: str = Field(description="어떤 자료·성과가 확인되면 판단을 바꿀지")


class CompanyHoldProse(BaseModel):
    company_id: str
    reasons: str = Field(description="이 후보가 보류된 이유: 점수가 낮은 항목, 핵심 위험, 확인하지 못한 항목")
    revisit_conditions: str = Field(description="이 후보를 다시 검토하려면 무엇이 확인되어야 하는지")


class HoldProse(BaseModel):
    summary: str = Field(description="평가한 후보가 모두 보류인 이유를 후보별로 한두 문장씩 요약")
    companies: list[CompanyHoldProse]


class CondensedCell(BaseModel):
    id: int = Field(description="입력의 id 그대로")
    text: str = Field(description="목표 글자 수 이내로 줄인 글. 원문에 없는 수치·출처를 넣지 않는다")


class CondensedCells(BaseModel):
    cells: list[CondensedCell] = Field(description="입력의 모든 id 에 대해 하나씩")
