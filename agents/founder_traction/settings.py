from dataclasses import dataclass


VERSION = "3.3"

SECTIONS = {
    "team": "CEO·창업자별 경력, 학력, 명시된 기술, 논문, 분야 전문성, 제품 실행 경험",
    "traction": "고객 확보, 유료 PoC/계약/매출 구분, 매출 추이와 질, 상용화 실적",
    "deal_terms": "투자 이력, 제시 조건, 현금, 월 순소진액, 기준일 런웨이, 다음 라운드 전 목표",
}

SUMMARY_CATEGORY_ORDER = {
    "team": ["career", "education", "skill", "expertise", "publication", "execution", "linkedin_activity"],
    "traction": ["customer", "revenue", "commercialization", "execution", "milestone"],
    "deal_terms": ["funding", "terms", "cash", "burn", "runway", "milestone"],
}

MISSING_CATEGORY_TEXT = {
    "career": "검증된 경력 자료",
    "education": "검증된 학력 자료",
    "skill": "명시적으로 확인되는 기술 목록",
    "expertise": "검증된 분야 전문성 자료",
    "publication": "동일인 검증을 통과한 논문 자료",
    "execution": "제품 실행 경험",
    "linkedin_activity": "확인 가능한 LinkedIn 활동",
    "customer": "고객 관련 공개 근거",
    "revenue": "매출",
    "commercialization": "상용화 실적",
    "funding": "투자 유치 이력",
    "terms": "구체적인 투자 조건",
    "cash": "현금 잔액",
    "burn": "월 순소진액",
    "runway": "검증 가능한 런웨이",
    "milestone": "다음 라운드 전 목표",
}


@dataclass(frozen=True)
class AgentConfig:
    max_founders: int = 3
    max_peers: int = 2
    workers: int = 3
    results_per_query: int = 3
    snippet_chars: int = 3000
    page_timeout: int = 45

    def __post_init__(self):
        positive = ("max_founders", "workers", "results_per_query", "snippet_chars", "page_timeout")
        if any(getattr(self, key) < 1 for key in positive):
            raise ValueError("설정값은 양수여야 합니다.")
        if self.max_peers < 0:
            raise ValueError("max_peers는 음수가 될 수 없습니다.")
