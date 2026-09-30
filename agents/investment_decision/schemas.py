from pydantic import BaseModel, Field


class CriterionAssessment(BaseModel):
    """평가 항목 하나의 채점 결과."""

    score: int | None = Field(
        ge=0,
        le=5,
        description=(
            "채점 기준에 따른 0~5점 정수. "
            "근거가 부족하여 채점할 수 없으면 null. "
            "정보 부족을 0점으로 처리하지 않는다."
        ),
    )

    rationale: str = Field(
        description=(
            "제공된 분석과 채점 기준을 연결한 점수 산정 이유. "
            "채점할 수 없다면 그 이유를 설명한다."
        ),
    )

    evidence: list[str] = Field(
        description=(
            "채점에 실제 사용한 입력 분석의 source_id 또는 URL. "
            "제공되지 않은 출처를 만들지 않는다."
        ),
    )

    missing_information: list[str] = Field(
        description=(
            "점수 확정 또는 판단 변경에 필요한 미확인 사항. "
            "없으면 빈 목록."
        ),
    )


class InvestmentAssessment(BaseModel):
    """현재 기업의 여섯 평가 항목에 대한 LLM 출력."""

    technology: CriterionAssessment
    competition: CriterionAssessment
    market: CriterionAssessment
    team: CriterionAssessment
    traction: CriterionAssessment
    deal_terms: CriterionAssessment