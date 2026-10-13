"""
보고서 품질 평가 기준 (결정적 검사에 쓰는 상수와 패턴)

평가 항목은 docs/1013과제.md D절의 3개이다.
- Groundedness: 주장이 검색된 출처로 추적되는가 (본문의 [번호] 인용이 REFERENCE 에 있는가)
- 중립성: 근거 없는 단정·추천·후보 간 우열 서술이 없는가 (INVEST/HOLD 판정 자체는 허용한다)
- 편향 통제: 단일 출처·유리한 근거 편중이 없는가

보고서의 문단은 "출처 번호를 쓰지 않는다"는 규칙으로 작성되고 출처는 표의 출처 칸에 붙는다(prompts/report_writer.py).
그래서 "수치 문장마다 [번호]가 있는가"는 결정적으로 검사하지 않고 LLM Judge 가 인용문과 함께 지적하게 한다.
"""

import re

# 설정값: 품질 평가의 가드레일 기준. 실행 결과를 보고 조정한다.
MAX_REVISIONS = 1  # 품질 미달 시 보고서를 다시 쓰는 최대 횟수 (평가는 최대 MAX_REVISIONS + 1 번)
MAX_TOP_SOURCE_SHARE = 0.5  # 가장 많이 인용된 출처가 전체 인용에서 차지하는 비율 상한
MIN_DISTINCT_SOURCES = 2  # 본문이 인용하는 서로 다른 출처의 최소 개수
MIN_CITATIONS_FOR_SHARE = 4  # 인용이 이보다 적으면 비율 검사를 하지 않는다
JUDGE_MAX_REPORT_CHARS = 30000  # LLM Judge 에 보내는 보고서 최대 길이

# 보고서 형식 ("## 8. REFERENCE" 또는 "## 5. REFERENCE")
REFERENCE_HEADING_RE = re.compile(r"^##\s*(?:\d+\.\s*)?REFERENCE\s*$", re.MULTILINE)
REFERENCE_ENTRY_RE = re.compile(r"^\[(\d+)\]", re.MULTILINE)
CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# 근거 없는 단정·추천 표현 (설정 목록)
PROMOTIONAL_PHRASES = ["강력 추천", "강력히 추천", "반드시 투자", "무조건", "압도적", "확실한 성공", "틀림없", "최고의 기업"]

# 후보 두 곳 이상이 한 문장에 함께 나올 때 우열을 뜻하는 표현 (설정 목록)
RANKING_PHRASES = ["보다 우수", "보다 낫", "보다 뛰어", "보다 유망", "앞선다", "앞서 있", "뒤처", "열위"]
