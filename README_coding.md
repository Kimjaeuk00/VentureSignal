# VentureSignal

반도체(저전력·고효율 AI 칩) 스타트업을 탐색·분석하고, 반도체 특화 Scorecard로 투자 여부(`INVEST`/`HOLD`)를 판단해 투자 보고서를 만드는 LangGraph 멀티 에이전트 RAG 시스템.

> 울산캠퍼스 2반 2조 · 김상현, 김영준, 김재욱, 임예리, 황민진
> 설계 기준 문서: `docs/RAG-Design_울산캠퍼스-2반_…md` (구현은 이 문서를 따릅니다)

---

## 1. 환경 세팅 (uv)

[uv](https://docs.astral.sh/uv/)로 가상환경과 패키지를 관리합니다. Python 3.11을 사용합니다.

```bash
# uv 설치 (macOS)
brew install uv

# 프로젝트 루트에서 가상환경 생성 + 패키지 설치
uv venv -p 3.11 .venv
uv pip install -p .venv -r requirements.txt

# 가상환경 활성화
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

활성화하지 않고 실행하려면 `uv run` 대신 `.venv/bin/python …`처럼 경로를 직접 써도 됩니다.

| 상황 | 명령 |
|---|---|
| 패키지 추가 | `requirements.txt`에 버전 고정으로 한 줄 추가 → `uv pip install -p .venv -r requirements.txt` |
| 환경 초기화 | `rm -rf .venv` 후 위 설치 과정 다시 실행 |
| 설치된 목록 확인 | `uv pip list -p .venv` |

- 패키지 추가·버전 변경은 **팀원 모두의 환경에 영향**을 주므로, 팀에 공유한 뒤 반영합니다.
- `.venv/`는 git에 올라가지 않습니다. 각자 만듭니다.

### API 키 설정

```bash
cp .env.example .env
```

`.env`를 열어 채웁니다. (`.env`는 git에 올라가지 않습니다. 키를 코드나 채팅에 붙여넣지 마세요.)

| 변수 | 설명 |
|---|---|
| `LLM_PROVIDER` | `anthropic` 또는 `openai` |
| `LLM_MODEL` | 예) `claude-opus-5-5` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | 사용하는 제공자의 키 |
| `TAVILY_API_KEY` | 웹서치용 (경쟁사 비교, 창업자 및 실적 에이전트) |

---

## 2. RAG 인덱스 만들기 (각자 1회)

RAG 인덱스(`data/index/`)는 git에 올라가지 않으므로, **팀원마다 로컬에서 한 번 실행**해야 합니다.

```bash
python -m rag.ingest
```

- 처음에 bge-m3 모델(약 2.3GB)을 내려받아 시간이 걸립니다.
- 성공하면 `sources: 90`, `indexed pages: 140`이 출력됩니다.
- RAG를 쓰는 에이전트(스타트업 탐색 · 기술 요약 · 시장성 평가)는 이게 있어야 동작합니다.
- 인덱스 로직이나 PDF 파싱 규칙이 바뀌면 다시 실행합니다.

RAG 사용법, 에이전트별 검색 범위, 규칙은 **[`rag/README.md`](rag/README.md)** 를 보세요.

---

## 3. 작업 방식 — 내 에이전트 폴더에서만 작업

### 원칙

> **팀원 1명 = 에이전트 폴더 1개.** 내 폴더 안에서만 파일을 만들고 수정합니다.

| 폴더 | 노드 | 방식 | 담당자 |
|---|---|---|---|
| `agents/startup_search/` | 스타트업 탐색 | RAG | |
| `agents/tech_summary/` | 기술 요약 | RAG | |
| `agents/market_eval/` | 시장성 평가 | RAG | |
| `agents/competitor_compare/` | 경쟁사 비교 | 웹서치 | |
| `agents/founder_traction/` | 창업자 및 실적 | 웹서치 | |
| `agents/investment_decision/` | 투자 판단 | LLM | |
| `agents/report_writer/` | 보고서 생성 | LLM | |
| `agents/candidate_select/` | 후보 선택 | 함수 (구현 완료) | |

### 코드 충돌을 피하는 규칙

1. **내 폴더 밖은 수정하지 않습니다.** 다른 사람의 에이전트 폴더, `core/`, `graph/`, `rag/`, `tools/`, `requirements.txt`, `main.py`는 공용입니다. 수정이 필요하면 팀에 먼저 공유하고 한 사람이 반영합니다.
2. **진입점 이름을 바꾸지 않습니다.** 각 폴더의 `node.py` 안 `run(state) -> dict`가 진입점이고, `graph/builder.py`가 이걸 불러옵니다. 함수 이름·시그니처·폴더 이름을 바꾸면 그래프가 깨집니다.
3. **자기 필드만 반환합니다.** 분석 에이전트는 `evaluation_update(cid, technology=make_analysis(...))`처럼 자기 담당 필드만 돌려줍니다. 다른 노드의 결과와는 `merge_evaluations`가 알아서 합칩니다. (담당 필드는 각 `node.py` 상단 docstring에 적혀 있습니다.)
4. **점수는 매기지 않습니다.** 분석 에이전트는 사실·근거·강점·위험요소만 채웁니다. 채점은 `investment_decision` 에이전트만 합니다.
5. **출처를 남깁니다.** `evidence`에 출처(`S010` 같은 source_id 또는 웹 URL)를 반드시 넣습니다. 출처 없는 수치는 쓰지 않습니다.
6. **내 폴더 안에서는 자유롭게 파일을 추가합니다.** 예: `prompts.py`(프롬프트), `schemas.py`(LLM 구조화 출력용 Pydantic 모델), 헬퍼 모듈.
7. **테스트는 별도 파일로 만듭니다.** `tests/agents/test_<폴더명>.py`처럼 파일을 나눠 두면 충돌하지 않습니다. (`tests/test_state.py`, `test_graph.py`, `test_scoring.py`는 공용)

### 공용 도구

```python
from core.llm import get_llm                          # LLM (.env 로 OpenAI/Anthropic 전환)
from core.state import evaluation_update, make_analysis
from rag.retriever import search, get_company_pages   # RAG
from rag.sources import format_source                 # "S010" → 서지정보
from tools.web_search import web_search               # Tavily 웹검색
```

### 내 노드 개발 순서

1. `agents/<내 폴더>/node.py` 상단 docstring에서 입력·출력을 확인합니다.
2. `TODO`로 표시된 스텁을 실제 구현으로 바꿉니다. (프롬프트·스키마는 폴더 안 별도 파일로)
3. `python main.py "국내 저전력 엣지 AI 칩"`으로 전체 그래프를 돌려봅니다. **아직 구현하지 않은 다른 노드는 스텁이 더미 결과를 채우므로**, 내 노드만 구현한 상태에서도 그래프가 끝까지 돕니다.
4. 결과는 `outputs/`에 저장됩니다. (git에는 올라가지 않습니다.)

자세한 규약은 [`agents/README.md`](agents/README.md)를 보세요.

---

## 4. 명령어 모음

```bash
python main.py "질의"                                    # 전체 그래프 실행
pytest                                                   # 전체 테스트
pytest tests/test_graph.py::test_invest_stops_loop       # 단일 테스트
python -m rag.ingest                                     # RAG 인덱스 (재)생성
```

---

## 5. 그래프 흐름

```
START → 스타트업 탐색 → 후보 선택 → [기술 요약 → 시장성 평가 → 경쟁사 비교 → 창업자 및 실적] → 투자 판단
투자 판단 ─ INVEST ─────────────→ 보고서 생성 → END
         ─ HOLD & 후보 남음 ─────→ 후보 선택 (루프)
         ─ HOLD & 후보 소진 ─────→ 보고서 생성 (보류 사유 포함) → END
```

Scorecard 가중치: 기술 35% · 경쟁 우위 25% · 시장성 15% · 창업자 10% · 실적 10% · 투자조건 5%.

### State 데이터 흐름 예시 (HyperAccel, PDF 11~12쪽)

질의 `"국내 LLM 전용 AI 반도체 IP"`에서 첫 후보 **HyperAccel(C03)** 이 평가되는 과정입니다. 분석 내용은 PDF 11~12쪽과 출처 `S006`·`S007`에서 가져왔고, 아래 표시는 구분해서 읽으세요.

- **PDF에 있는 값:** 기업 설명, LPU IP 사양, 550억 원 Series A, 한계·확인 질문, 비교군
- **설명용 더미 값:** 후보 2·3번, `retrieval_score`, 웹서치 결과(5·6번의 경쟁사·인물·고객·`example.com` URL·런웨이), Scorecard 점수 (실제 조사·채점 결과가 아님)

각 노드는 **표에 적힌 필드만** 반환합니다. 필드 정의는 `core/state.py`를 보세요.

> **분량 안내:** 아래 예시 데이터는 흐름을 보여주려고 **간략하게 줄인 것**입니다. 실제 구현에서는 이보다 **훨씬 길고 상세하게** 채워도 됩니다. 보고서(설계 문서 E절의 8개 섹션)를 쓸 때 근거가 많이 필요하므로, 중요하다고 판단되는 내용은 생략하지 말고 충분히 남기세요.
> - `summary`, `strengths`, `risks`는 항목 수·문장 길이 제한이 없습니다. 보고서 문단으로 바로 쓸 수 있을 만큼 구체적으로 씁니다.
> - `evidence`에는 사용한 출처를 빠짐없이 넣습니다. (보고서 REFERENCE의 재료가 됩니다)
> - `details`에는 보고서 표를 채울 수 있는 값(수치·단위·측정 조건·날짜·출처 번호, 경쟁사 비교표, 인물 이력 등)을 구조화해서 넣습니다.
> - 단, 길게 쓰더라도 **근거 없는 내용을 채우지 않습니다.** 확인하지 못한 값은 "미확인"으로 표시합니다.

| # | 노드 | State에 채워지는 것 |
|---|---|---|
| 0 | START | `query="국내 LLM 전용 AI 반도체 IP"`, `candidates=[]`, `candidate_index=0`, `current_candidate=None`, `evaluations={}`, `selected_candidate=None`, `report=None` |
| 1 | 스타트업 탐색 | `candidates=[C03 HyperAccel, C06 AiM Future, C04 Mobilint]` (11쪽 탐색 태그 "LLM 전용 반도체 IP, 국내 LPU 설계"에 매칭) |
| 2 | 후보 선택 | `current_candidate="C03"`, `candidate_index=1` |
| 3 | 기술 요약 | `evaluations["C03"]["technology"]` |
| 4 | 시장성 평가 | `evaluations["C03"]["market"]` |
| 5 | 경쟁사 비교 | `evaluations["C03"]["competition"]` |
| 6 | 창업자 및 실적 | `evaluations["C03"]["team"]`, `["traction"]`, `["deal_terms"]` |
| 7 | 투자 판단 | `evaluations["C03"]["scorecard"]`, `["decision"]="HOLD"` |
| 8 | (분기) | `HOLD`이고 `candidate_index(1) < len(candidates)(3)` → 2번으로 돌아가 `C06`을 평가. INVEST가 나오면 `selected_candidate`를 채우고 보고서로 이동 |

**1번 `candidates` 한 칸**

```python
{
    "company_id": "C03",
    "company_name": "HyperAccel",
    "description": "LLM 전용 프로세서 LPU 기술을 개발하는 국내 AI 반도체 IP 기업",
    "domain": "AI 반도체 IP",              # 11쪽 헤더: C03 | 국내 | AI 반도체 IP
    "retrieval_score": 0.83,              # 설명용
}
```

**3번 기술 요약 (RAG, 11쪽)**: 수치는 `source_id`가 있는 것만, 조건과 함께 적습니다.

```python
{
    "summary": "LPU IP는 메모리·연산 구성을 조절하는 모듈형 설계와 PCIe·UCIe 통합을 제시한다 (회사 공개).",
    "strengths": ["모듈형 설계로 고객 SoC에 맞춰 구성 조절 가능", "LLM 전용 구조(LPU)"],
    "risks": [
        "공개 수치는 코어 단위 예시라 완성 카드·서버 수치와 직접 비교 불가",
        "독립 실측 없음 (제품 페이지의 공개 주장)",
        "LPU 명칭은 업체별로 의미가 달라 company_id와 함께 색인 필요",
    ],
    "evidence": ["S006"],
    "details": {
        "product": "LPU IP",
        "core_examples": [
            {"type": "latency 지향", "tflops_fp16": 8, "power_w": 3.55},
            {"type": "high-performance", "tflops": 12, "power_w": 4.72},
        ],
        "scope": "코어 단위 예시 (완성 카드·서버 수치 아님)",
        "unverified": ["목표 공정·면적", "검증 조건", "완성 실리콘 측정값"],
    },
}
```

**4번 시장성 평가 (RAG, 12쪽)**

```python
{
    "summary": "고객 SoC에 통합하려는 IP 수요와 추론 서버 수요를 나누어 평가할 수 있다 (분석).",
    "strengths": ["LLM에 맞춘 IP를 SoC에 통합하려는 수요 존재 (분석)"],
    "risks": [
        "IP 사업은 설계 채택과 라이선스·양산 로열티 시점이 달라 매출 발생 구조 확인 필요",
        "양산 일정·수율, 지원 모델, IP 라이선스 형태, 고객 설계 채택 여부 미확인",
    ],
    "evidence": ["S006", "S007"],
    "details": {
        "buyer_split": ["IP 구매자 (공정·검증·통합 일정 중심)", "서버 구매자 (처리량·서비스 비용·운영 지원 중심)"],
        "open_questions": ["실제 유상 고객·반복 구매", "IP/서버 구매자 구분", "다음 세대로 모델·SDK 이전 가능 여부"],
    },
}
```

**5번 경쟁사 비교 (웹서치)**: 비교군은 PDF 12쪽 기준이고, 나머지는 웹서치를 했다고 가정한 **더미 값**입니다. 실제 구현에서는 `web_search()` 결과의 URL을 `evidence`에 남깁니다.

```python
{
    "summary": "AiM Future, Synthara와 같은 AI 반도체 IP 범주에서 비교된다. LLM 전용 구조(LPU)가 차별점이나, 동등한 제품·성능 순위는 아니다.",
    "strengths": ["LLM 추론에 특화된 모듈형 IP", "PCIe·UCIe 통합 지원"],
    "risks": ["범용 NPU IP 업체가 LLM 지원을 확대할 가능성", "IP 코어 수치 기준이 업체마다 달라 직접 비교가 어렵다"],
    "evidence": ["S006", "https://example.com/dummy/edge-npu-ip-comparison"],   # 더미 URL
    "details": {
        "peer_group": ["AiM Future", "Synthara"],
        "compare_axes": ["PPA", "통합 기간", "지원 연산"],
        "comparison": [                                                        # 보고서 4절 비교표용 (더미)
            {"company": "HyperAccel", "product": "LPU IP", "target": "LLM 추론 SoC", "strength": "LLM 전용 구조", "weakness": "완성 실리콘 검증 미공개"},
            {"company": "AiM Future", "product": "NeuroMosAIc", "target": "엣지 AI SoC", "strength": "다양한 모델 지원", "weakness": "LLM 특화 근거 부족"},
        ],
    },
}
```

**6번 창업자 및 실적 (웹서치)**: 아래 인물·고객·런웨이는 **모두 더미 값**입니다. 실제 구현에서는 웹서치 결과로 채우고, 못 찾은 항목만 "미확인"으로 남깁니다.

```python
"team": make_analysis(
    summary="창업팀은 반도체 설계와 AI 모델 서비스 경험을 함께 가진 구성 (더미)",
    strengths=["칩 설계 경력 보유 (더미)", "LLM 서비스 경험 보유 (더미)"],
    risks=["칩 양산 경험 인력 규모 불명 (더미)"],
    evidence=["https://example.com/dummy/hyperaccel-team"],
    details={"members": [{"name": "홍길동", "role": "CEO", "career": "반도체 설계 10년+"}]},   # 더미
),
"traction": make_analysis(
    summary="시험 도입 단계 고객이 있으나 반복 주문·매출은 공개 자료에서 확인되지 않음 (더미)",
    strengths=["SoC 설계사 대상 PoC 진행 (더미)"],
    risks=["투자 발표의 미래 생산 계획을 달성한 실적으로 저장하지 않는다"],
    evidence=["https://example.com/dummy/hyperaccel-customers"],
    details={"customers": [{"name": "더미 고객사", "stage": "PoC"}], "revenue": "비공개"},
),
"deal_terms": make_analysis(
    summary="2025년 550억 원 Series A 유치 발표 (회사 공개, 연도 수준으로 기록). 다음 라운드까지의 런웨이는 약 2년으로 추정 (더미)",
    strengths=["대규모 시리즈 A로 개발 자금 확보"],
    risks=["페이지 날짜와 본문 발표일 표현이 일치하는지 불명확", "현재 법인·자본 상태는 별도 확인 필요"],
    evidence=["S007", "https://example.com/dummy/hyperaccel-funding"],
    details={"round": "Series A", "amount_krw": 55_000_000_000, "year": 2025, "runway_months": 24},   # runway 는 더미
),
```

**7번 투자 판단 후 `evaluations["C03"]`** (병합 결과)

```python
{
    "technology": {...}, "market": {...}, "competition": {...},   # 3~5번
    "team": {...}, "traction": {...}, "deal_terms": {...},        # 6번
    "scorecard": {                       # 설명용 점수: 항목별 0~5 + 가중합(0~100)
        "technology": 3, "competition": 3, "market": 3,
        "team": 2, "traction": 1, "deal_terms": 3,
        "total_score": 54.0,             # 임계값(70) 미만
    },
    "decision": "HOLD",
}
```

핵심 규칙:
- 3~6번 노드는 서로의 결과를 덮어쓰지 않습니다. `merge_evaluations`가 `company_id`별로 필드를 합칩니다.
- 채점(`scorecard`, `decision`)은 7번 투자 판단 노드만 합니다.
- 분석 노드는 점수 없이 사실·근거·위험요소만 담습니다. 이 예시는 흐름을 보여주려고 5·6번을 더미로 채웠지만, **실제 구현에서는 확인하지 못한 값을 만들어 채우지 않고 "미확인"으로 남깁니다.** (PDF 12쪽: 실적 결과가 없으면 값을 만들어 평가 점수를 채우지 않음)
- 전 후보가 HOLD여도 `selected_candidate`는 `None`인 채로 보고서(보류 사유 포함)로 갑니다.

## 6. 팀에서 정해야 할 것

- **`INVEST_THRESHOLD`** (`core/config.py`, 현재 70): 설계 문서에 없는 임시값입니다. 합의가 필요합니다.
- **RAG 검색 품질:** 기본 수준입니다. 개선 방향은 `rag/README.md`의 "알려진 한계"를 보세요.
