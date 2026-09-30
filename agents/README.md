# agents/ — 에이전트별 작업 공간

각 팀원은 **담당 에이전트 폴더 안에서만** 작업합니다. 폴더 밖 공용 코드(`core/`, `graph/`, `rag/`, `tools/`, `requirements.txt`)는 팀 공유 후 수정합니다.

## 담당

| 폴더 | 노드 | 방식 | 담당자 |
|---|---|---|---|
| `startup_search/` | 스타트업 탐색 | RAG | |
| `candidate_select/` | 후보 선택 | 함수 (구현 완료) | |
| `tech_summary/` | 기술 요약 | RAG | |
| `market_eval/` | 시장성 평가 | RAG | |
| `competitor_compare/` | 경쟁사 비교 | 웹서치 | |
| `founder_traction/` | 창업자 및 실적 | 웹서치 | |
| `investment_decision/` | 투자 판단 | LLM | |
| `report_writer/` | 보고서 생성 | LLM | |

## 노드 계약

- 진입점은 `node.py`의 `run(state: GraphState) -> dict` 하나. 이름·시그니처를 바꾸지 않습니다 (`graph/builder.py`가 `<폴더>.run`을 import).
- 입력·출력 필드는 각 `node.py` 상단 docstring에 명시되어 있습니다. **자기 필드만 반환**합니다.
- 분석 노드는 `evaluation_update(cid, <필드>=make_analysis(...))`로 반환합니다. `merge_evaluations`가 다른 노드 결과와 병합합니다.
- 분석 노드는 점수를 매기지 않습니다. 채점은 `investment_decision`만 합니다.
- `evidence`에는 출처(`S010` 같은 source_id 또는 URL)를 반드시 포함합니다.

## 폴더 안에서 자유롭게 추가해도 되는 것

- 프롬프트 문자열은 루트 `prompts/<에이전트>.py` 에 둔다 (노드는 `from prompts.<에이전트> import ...`)
- `schemas.py` — LLM 구조화 출력용 Pydantic 모델 (`get_llm().with_structured_output(...)`)
- 기타 헬퍼 모듈

테스트는 `tests/agents/test_<폴더명>.py` 로 따로 파일을 만들어 충돌을 피합니다.

## 공용 API

```python
from core.llm import get_llm                         # LLM (.env 로 OpenAI/Anthropic 전환)
from core.state import evaluation_update, make_analysis
from rag.retriever import search, get_company_pages  # RAG (python -m rag.ingest 선행)
from rag.sources import format_source                # "S010" → 서지정보
from tools.web_search import web_search              # Tavily
```

## 실행해 보기

아직 구현되지 않은 노드는 스텁이 결과를 채우므로, 내 노드만 구현한 상태에서도 `python main.py "질의"`로 전체 그래프가 돌아갑니다.
다른 노드를 임시 함수로 바꿔 끼우려면 `build_graph({노드이름: 함수})`를 사용합니다 (`tests/test_graph.py` 참고).
