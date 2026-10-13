"""LLM Judge 구조화 출력. 지적마다 보고서의 인용문을 함께 내게 해 근거 없는 지적을 걸러낸다."""

from typing import Literal

from pydantic import BaseModel, Field


class JudgeIssue(BaseModel):
    item: Literal["groundedness", "neutrality", "bias_control"]
    quote: str = Field(description="문제가 되는 보고서 문장을 그대로 옮긴 것. 보고서에 없는 문장이면 지적은 무시된다")
    reason: str = Field(description="왜 문제인지 한 문장")


class JudgeOutput(BaseModel):
    issues: list[JudgeIssue] = Field(default_factory=list, description="문제가 없으면 빈 목록")
