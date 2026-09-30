# founder_traction

창업자·팀, 사업 실적, 투자조건을 웹에서 조사하고 검증된 결과를 `GraphState`에 추가하는 노드입니다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `node.py` | LangGraph 진입점 `run(state) -> dict` |
| `agent.py` | 검색·수집·분석·검토 단계 오케스트레이션 |
| `schemas.py` | LLM 구조화 출력용 Pydantic 모델 |
| (루트 `prompts/founder_traction.py`) | 시스템·초안·검토·탐색 프롬프트 |
| `settings.py` | 버전, 영역별 설정, 실행 설정값 |
| `web_research.py` | Tavily 검색 연결, 페이지 추출, LinkedIn URL 정규화 |
| `source_collection.py` | 검색 결과 변환, 페이지 원문 청크화와 수집 상태 기록 |
| `analysis_helpers.py` | 인용 검증, 프로필 섹션 판별, 런웨이 계산, 안전한 요약 fallback |

## 데이터 흐름

```text
GraphState
→ CEO·창업자 식별
→ LinkedIn 개인 프로필 URL 검증
→ 프로필·공식 자료·기업 실적·투자 정보 수집
→ team / traction / deal_terms 구조화
→ 원문 인용 검증
→ 사실과 분석 문장 재검토
→ evaluation_update(company_id, ...)
```

분석 결과만 반환하며 점수나 투자 결정은 생성하지 않습니다. 확인되지 않은 값은 추정하지 않고
`missing_categories`에 남깁니다.

## 테스트

```bash
pytest -q tests/agents/test_founder_traction.py
```
