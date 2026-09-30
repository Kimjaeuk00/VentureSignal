"""기술 요약 LLM 구조화 출력. details 필드는 보고서 '5. 기술력' 표 항목과 맞춘다."""

from pydantic import BaseModel, Field


class Finding(BaseModel):
    content: str = Field(description="분석 내용. 자료로 확인할 수 없으면 '미확인'")
    source_ids: list[str] = Field(description="이 내용의 출처: source_id (예: S006) 또는 웹 출처 번호 (예: U3). 없으면 빈 목록")


class PerformanceItem(BaseModel):
    metric: str = Field(description="지표 이름 (예: 연산 성능, 전력, 전력 효율)")
    value: str = Field(description="자료에 적힌 수치 그대로. 단위는 unit 칸에만 쓴다 (예: value='25', unit='TOPS')")
    unit: str = Field(description="단위 (예: TFLOPS FP16, W, TOPS/W)")
    condition: str = Field(description="측정 조건·범위 (예: 코어 단위, 칩 단위, 정밀도). 모르면 '미확인'")
    source_ids: list[str] = Field(description="이 수치의 출처: source_id (예: S006) 또는 웹 출처 번호 (예: U3)")


class TechSummaryOutput(BaseModel):
    summary: str = Field(description="핵심 기술·제품·개발 단계를 보고서 문단으로 바로 쓸 수 있게 요약")
    strengths: list[str] = Field(description="기술적 강점")
    risks: list[str] = Field(description="기술적 위험·한계·검증 공백")
    evidence: list[str] = Field(description="사용한 모든 출처 (source_id 또는 웹 출처 번호)")
    product: str = Field(description="제품명만 짧게 (예: DX-M1, LPU IP). 설명·출처 표기·괄호를 붙이지 않는다")
    core_technology: Finding = Field(description="핵심 기술 내용과 차별점")
    product_stage: Finding = Field(description="제품 완성도·개발 단계 (시제품 / 검증 / 양산 등)")
    performance: list[PerformanceItem] = Field(description="출처가 있는 성능·전력 수치만")
    software_environment: Finding = Field(description="사용 환경: 지원 소프트웨어·SDK·고객 도입 준비")
    validation_limits: Finding = Field(description="외부 검증 여부와 미확인 성능 등 한계")
    unverified: list[str] = Field(description="자료에서 확인하지 못한 항목")
