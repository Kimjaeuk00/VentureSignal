from typing import Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    source_id: str
    quote: str


class Founder(BaseModel):
    name: str
    aliases: list[str] = Field(default_factory=list)
    role: str
    linkedin_url: str | None = None
    evidence: list[Evidence] = Field(default_factory=list)


class Discovery(BaseModel):
    founders: list[Founder] = Field(default_factory=list)


class Fact(BaseModel):
    id: str
    subject: str = Field(description="창업자 이름 또는 기업명. 피어는 해당 기업명")
    category: Literal[
        "career", "education", "skill", "expertise", "publication",
        "unverified_publication", "execution", "linkedin_activity", "customer",
        "revenue", "commercialization", "funding", "terms", "cash", "burn",
        "runway", "milestone",
    ]
    text: str
    attributes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "경력: organization/position/start/end; 학력: institution/degree/field/year; "
            "논문: title/doi/year/authors/identity_basis; 수치: value/unit/as_of/period/basis. "
            "cash/burn은 동일 화폐 단위의 숫자 value, YYYY-MM-DD as_of, "
            "월 순소진액 period=month. 미확인은 생략"
        ),
    )
    evidence: list[Evidence] = Field(default_factory=list)


class Statement(BaseModel):
    id: str
    kind: Literal["summary", "strength", "risk", "peer_comparison"]
    text: str
    fact_ids: list[str] = Field(default_factory=list)


class Draft(BaseModel):
    facts: list[Fact] = Field(default_factory=list)
    statements: list[Statement] = Field(default_factory=list)


class Review(BaseModel):
    accepted_fact_ids: list[str] = Field(default_factory=list)
    accepted_statement_ids: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
