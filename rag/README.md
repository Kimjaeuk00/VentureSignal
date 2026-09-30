# rag/ — RAG 공용 모듈

조사자료(`data/AI_Chip_Startup_RAG_Research_150p.pdf`)를 검색하는 공용 코드입니다. RAG를 쓰는 에이전트(스타트업 탐색 · 기술 요약 · 시장성 평가)는 이 모듈을 가져다 쓰고, 직접 임베딩·벡터 DB를 다루지 않습니다.

## 구성

| 파일 | 역할 |
|---|---|
| `ingest.py` | PDF 파싱 → 페이지 단위 청크 + 메타데이터 → Qdrant 저장. 출처 90개는 `data/index/sources.json`으로 추출 |
| `embeddings.py` | bge-m3로 dense + sparse 벡터 생성 |
| `store.py` | Qdrant 클라이언트 (로컬 파일 모드) |
| `retriever.py` | 에이전트가 쓰는 검색 함수 |
| `sources.py` | 출처 번호(`S010`) → 서지정보 |

## 처음 한 번: 인덱스 만들기

인덱스(`data/index/`)는 git에 올리지 않으므로 **팀원마다 로컬에서 한 번 실행**해야 합니다.

```bash
python -m rag.ingest
```

- 처음 실행 시 bge-m3 모델(약 2.3GB)을 내려받습니다.
- 성공하면 `indexed pages: 140`, `sources: 90`이 출력됩니다.
- 이 명령을 실행하지 않으면 `search()` 등이 실패합니다.

## 사용법

```python
from rag.retriever import search, get_company_pages
from rag.sources import format_source

# 하이브리드 검색 (dense + sparse, RRF 결합)
search("저전력 엣지 NPU", top_k=5, page_types=["company_tech"])

# company_id 로 기업 페이지 2장(기술 / 시장성)을 검색 없이 그대로 가져오기
get_company_pages("C05")

# 보고서 REFERENCE 용 서지정보
format_source("S010")
```

## 에이전트별 검색 범위 (`page_type`)

| page_type | 내용 | 쪽 |
|---|---|---|
| `company_index` | 기업 색인 | 4~6 |
| `company_tech` | 기업별 사업·제품·기술 근거 | 기업당 1p |
| `company_market` | 기업별 시장성·한계·확인 과제 | 기업당 1p |
| `tech_topic` | 기술 구조·성능 비교 15개 주제 (T01~T15) | 107~121 |
| `market_topic` | 수요·시장·상용화 15개 주제 (T16~T30) | 122~136 |
| `meta` | 표지·방법론·RAG 설계 | — |

| 에이전트 | 사용할 page_type |
|---|---|
| 스타트업 탐색 | `company_index`, `company_tech` |
| 기술 요약 | `company_tech` + `tech_topic` |
| 시장성 평가 | `company_market` + `market_topic` |

각 청크에는 `company_id`(C01~C50), `topic_id`, `region`, `category`, `source_ids`(S001~S090) 메타데이터가 붙어 있습니다.

## 팀 규칙

1. **`rag/`는 공용 코드입니다.** 에이전트 폴더 안에서만 작업하는 원칙대로, 검색 품질이 부족해도 각자 `rag/`를 고치지 말고 팀에 공유한 뒤 함께 수정합니다. (에이전트 전용 로직은 자기 폴더에 두세요.)
2. **`data/index/`는 git에 올리지 않습니다.** 코드나 PDF 파싱 규칙이 바뀌면 각자 `python -m rag.ingest`를 다시 실행합니다.
3. **Qdrant 클라이언트는 `rag/store.get_client()`로만 얻습니다.** 로컬 파일 모드는 한 경로에 프로세스 하나만 붙을 수 있어서, 직접 `QdrantClient(path=...)`를 만들면 잠금 오류가 납니다. 노트북·스크립트를 여러 개 동시에 열어 둘 때도 주의하세요.
4. **`evidence`에는 출처를 남깁니다.** 검색 결과의 `source_ids`를 그대로 `AnalysisResult.evidence`에 담아야 보고서 REFERENCE로 이어집니다. **source_id가 없는 성능 수치는 쓰지 않습니다.** (PDF 138쪽)
5. **sparse 검색을 끄지 않습니다.** `HBM3E`, `TOPS/W`, `CoWoS` 같은 전문용어는 정확 일치가 중요합니다.
6. **조사자료의 표시를 구분합니다.** 공개 사실 / 분석 / 미확인은 다릅니다. 분석·가설을 확정 사실처럼 쓰지 않고, 미확인 항목은 추정 수치로 채우지 않습니다.
7. **상장·인수 기업은 투자 후보가 아닙니다.** 비교용 사례이므로 후보로 반환하지 않습니다. (PDF 132쪽, 139쪽 평가 질의 참고)

## 알려진 한계 (개선 필요)

- **검색 품질은 기본 수준입니다.** "국내 5W 이하 NPU" 질의(PDF 139쪽 평가 질의)를 넣으면 기대 정답인 DEEPX 대신 Mobilint가 1위로 나옵니다. 스타트업 탐색 담당자가 지역 필터, 질의 조건 추출, 리랭킹 등으로 개선해야 합니다. 필터에 쓸 `region`, `category`는 메타데이터에 이미 있습니다.
- **청크는 페이지 단위입니다.** PDF 137쪽은 의미 단위 분할(400~800 토큰)을 권장합니다. 검색 평가 결과를 보고 조정합니다.
- **`retriever.search()`에는 `region` 필터 인자가 아직 없습니다.** 필요하면 팀에 공유해 추가하세요.
- 검색 평가용 질의와 정답 조건은 PDF 139쪽에 있습니다. 개선 작업 시 회귀 테스트로 쓸 수 있습니다.
