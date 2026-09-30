"""상장 성공 기업 3곳의 기준 점수 산정 — 우리 그래프와 같은 흐름(분석 4개 노드 + 웹 검색 → 투자 판단 채점).

실행: python -m agents.investment_decision.reference_scoring [--out 경로] [--resume 경로] [--only 회사ID ...]

탐색·보고서 노드를 뺀 그래프 흐름 그대로다: 기술 요약 → 시장성 평가 → 경쟁사 비교 → 창업자 및 실적(모두 웹 검색 포함) → 투자 판단 채점.
`data/상장 성공 기업 5종 자료.pdf`(상장일 기준)는 RAG 조사자료가 하는 역할을 한다. 세 기업은 RAG 색인에 없어 기술 요약이
"기술 페이지를 찾지 못함"으로 끝나므로, PDF 에서 옮겨 적은 사실을 그 기업의 조사자료 페이지(company_tech / company_market)로 노드에 끼워 준다.
노드 코드는 고치지 않는다.

분석 노드의 웹 검색은 상장일이 아니라 실행 시점의 정보다. 채점은 후보와 같은 `assess` 를 쓰되, 흔들림을 줄이려고 기준 기업은 항상 3번 채점해
항목별 중앙값을 쓴다(`merge_samples`). 3곳의 핵심 점수 평균을 하한값 후보로 낸다.
전체 과정은 JSON 으로 남기고(단계마다 저장, --resume 으로 이어 실행) 웹 검색 호출 수를 센다.
"""

import argparse
import copy
import json
import logging
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from core.config import ROOT_DIR
from core.state import initial_state

from . import node as decision_node
from .benchmark import REFERENCE_COMPANIES

logger = logging.getLogger(__name__)

PDF_PATH = ROOT_DIR / "data" / "상장 성공 기업 5종 자료.pdf"
SAMPLES = 3  # 기업마다 채점을 이만큼 하고 항목별 중앙값을 쓴다
STALL_SECONDS = 300  # 한 단계가 이 시간을 넘으면 중단한다 (API 예산을 지키기 위해)

# PDF 쪽 번호 → 출처 번호. 실제 RAG 출처(S001~S090)와 겹치지 않게 S9xx 를 쓴다.
PDF_SOURCES = {
    "S901": "상장 성공 기업 5종 자료.pdf 1쪽 (상장일 기준 개요, 싸이닉솔루션 시장성)",
    "S902": "상장 성공 기업 5종 자료.pdf 2쪽 (싸이닉솔루션, 그린리소스 시작)",
    "S903": "상장 성공 기업 5종 자료.pdf 3쪽 (그린리소스)",
    "S904": "상장 성공 기업 5종 자료.pdf 4쪽 (Ambiq Micro 시장성)",
    "S905": "상장 성공 기업 5종 자료.pdf 5쪽 (Ambiq Micro 제품·경쟁 우위)",
}

LISTING = {
    "BENCH_GREEN": "그린리소스: 상장일 2023-11-24, 코스닥, 평가 기준일 2023-11-24, 신주 100% 공모.",
    "BENCH_SCINIC": "싸이닉솔루션: 상장일 2025-07-07, 코스닥, 평가 기준일 2025-07-07, 신주 100% 공모.",
    "BENCH_AMBIQ": "Ambiq Micro: 상장일 2025-07-30, 뉴욕증권거래소(NYSE), 평가 기준일 2025-07-30, 공모가 $24 / 첫날 종가 $38.53.",
}

# PDF 를 이미지로 읽고 옮겨 적은 사실. 텍스트 추출은 쪽 경계에서 표가 뒤섞이므로 쓰지 않았다.
# {분석: (본문, [출처 번호])}. 본문에 없는 내용을 더하지 않았고, 원문의 표기를 그대로 옮겼다.
FACTS = {
    "BENCH_SCINIC": {
        "market": ("설립연도: 2005년. 협업 팹리스 고객: 220개 이상 (잠재 시장 접근성을 수치화하는 지표). 누적 개발 프로젝트: 273건. 양산 제품: 946종. "
                   "프로젝트 양산 전환율: 70% 이상 (단순 설계 용역을 넘어선 상용화 전환 능력 지표). 2024년 매출: 1,674억 원. "
                   "최근 5년 CAGR: 매출 24.1% | 영업이익 32.9% | 순이익 57.2%. "
                   "분석 및 해석: 220개 이상의 팹리스 고객 수치는 높은 시장 접근성을 보여주나, 현재 매출 고객과 과거 누적 고객의 구분이 필요합니다. "
                   "양산 전환율 70% 이상은 뛰어난 상용화 능력을 나타내지만, 경쟁 디자인하우스의 동일 지표 공시가 부재하여 "
                   "'경쟁사 대비 % 우위'로 직접 비교하는 것은 지양해야 합니다.", ["S901", "S902"]),
        "technology": ("주요 제품군: PMIC, CIS, DDI. PMIC·CIS 매출 비중: 약 88% (특정 제품군 전문성 확보 및 집중도 표시). "
                       "누적 개발 프로젝트 / 양산 제품: 273건 / 946종 (설계 검증 및 생산 관리 경험 반영). 양산 전환율: 70% 이상. "
                       "주요 공정: 45nm 이상, 8인치 레거시 공정.", ["S902"]),
        "competition": ("핵심 경쟁력: 국내외 팹리스 고객 220개 이상, SK하이닉스시스템아이씨의 '유일한 디자인하우스'로 명시. "
                        "추가 검증 필요 사항: 독점 계약 및 우선협상권 유무, 파운드리 관련 실제 매출 비중, 계약 해지 조건, 타 디자인하우스의 동일 파운드리 접근 가능 여부. "
                        "현재 미확인 항목 (공개 자료 미제공): 경쟁사 대비 설계기간 단축률, 고객 원가 절감률, 설계 오류·재설계율, 칩 수율 개선폭 등.", ["S902"]),
    },
    "BENCH_GREEN": {
        "market": ("사업 특징: 반도체 장비 부품의 수명 연장, 내식성, 내플라즈마성 코팅 기술 보유. 설립연도 / 공모주식 수: 2011년 / 164만 주. "
                   "적용 공정: 3nm·5nm 등 초미세 식각공정. 영업이익 전망 (2024년 → 2025년): 119억 원 → 202억 원 (전망치 기준 약 69.7% 증가). "
                   "분석: 초미세 공정 적용은 높은 기술 난도를 의미하나 시장 규모 자체와 직결되지는 않으며, 2025년 실적은 상장 당시 전망치임을 감안해야 합니다.",
                   ["S902", "S903"]),
        "technology": ("자체 개발 PVD 코팅 성능: 기존 방식 대비 내식성·내마모성 5배 이상 주장. 기술 수직계열화: 코팅 소재, 코팅 장비, 코팅 공정 전체를 자체 확보. "
                       "주요 기능: 식각 플라즈마 내성, 부품 마모/부식 감소, 파티클/오염물질 저감, 장비 부품 교체주기 연장. "
                       "주의사항: '5배 이상' 수치는 회사 발표 자료이므로 비교 기준(기존 방식의 정의, 측정 단위, 플라즈마 조건, 코팅 두께, 독립기관 검증 여부)이 "
                       "확인되기 전까지는 '회사 주장 수치이며 독립 검증 필요'로 제한하여 기재합니다.", ["S903"]),
        "competition": ("수직계열화 & 국산화: 소재·장비·코팅 공정 자체 보유, 일본 의존도 높은 SPS 소재·장비 국산화. "
                        "기술평가 & 적용공정: 기술특례상장 당시 A·BBB, 3nm·5nm 식각공정 대응. 특허: 코팅 소재 및 장비 관련 특허/출원 보유. "
                        "독점성 판단: 법적 독점권, 특정 장비업체와 독점 공급계약 여부, 실제 고객사 부품 교체주기 증가율 수치는 미확인 상태이므로 "
                        "'독점 기술'로 단정하지 않고 차별화 요소로 명시합니다.", ["S903"]),
    },
    "BENCH_AMBIQ": {
        "market": ("목표 시장 규모 (TAM): 2023년 128억 달러 → 2028년 225억 달러 (증가액 97억 달러). 시장 연평균 성장률 (CAGR): 약 12.0%. "
                   "특징: MCU, ASIC, 무선칩, 엣지 AI 프로세서를 포함하는 광범위한 목표 시장 수치입니다.", ["S904"]),
        "technology": ("Apollo4 처리성능 / 에너지 효율: 최대 192MHz / 5µA/MHz. Apollo4 초저전력 대기전류: 모델별 14~55nA. "
                       "누적 출하량: 2억 9,000만 개 이상 (기술 양산성 검증 지표). MLPerf Tiny 에너지 효율 (자체 테스트): 경쟁 MCU 대비 300~930% 우위. "
                       "SPOT 기술 전력 우위: 경쟁 설계 대비 2~5배. 신뢰도 구분: 300~930% 효율 개선 수치는 공식 제출 결과가 아닌 회사 내부 MLPerf Tiny 테스트 결과이므로 "
                       "'자체 테스트 결과'로 표기합니다. (300% 개선 = 약 4배 효율, 930% 개선 = 약 10.3배 효율 수준)", ["S904", "S905"]),
        "competition": ("주요 고객 및 집중도: Google, Garmin, Huawei 등 핵심 고객 보유. 상위 3개 고객의 매출 비중이 85% 이상으로 높은 고객 집중 위험(Concentration Risk)을 수반합니다. "
                        "경쟁력의 본질: 단순 연산 성능 수치보다 동일 AI 작업을 훨씬 적은 전력으로 처리하는 에너지 효율성에 강점이 있습니다.", ["S905"]),
    },
}

# 기술 요약의 기본 웹 검색어는 "mass production customers" 처럼 시점이 없어 오래된 발표가 잡힐 수 있다.
# 상장 시점 전후의 양산·출하·고객 자료를 찾도록 연도를 붙인 검색어. --dated-tech 로 켠다. {target} = "기업명 제품명".
DATED_TECH_QUERIES = [
    "{target} production shipments customers 2024",
    "{target} 2024 annual revenue units shipped deployments",
    "{target} listing 2025 customers production ramp",
]

_web_calls = {"search": 0, "extract": 0}
_last_progress = {"step": "", "at": time.time()}


def _progress(step: str) -> None:
    _last_progress.update(step=step, at=time.time())
    print(f"[{datetime.now():%H:%M:%S}] {step}", flush=True)


def _watchdog() -> None:
    """한 단계가 STALL_SECONDS 를 넘기면 중단한다 (웹 검색·LLM 예산을 지키기 위해)."""
    import os

    while True:
        time.sleep(5)
        if time.time() - _last_progress["at"] > STALL_SECONDS:
            print(f"[중단] '{_last_progress['step']}' 단계가 {STALL_SECONDS}초를 넘겨 실행을 멈춘다", flush=True)
            os._exit(3)


def _count_web_calls() -> None:
    """Tavily 호출 수를 센다 (검색·추출)."""
    import tavily

    for name in ("search", "extract"):
        original = getattr(tavily.TavilyClient, name, None)
        if not original:
            continue

        def wrap(orig, key):
            def counted(self, *args, **kwargs):
                _web_calls[key] += 1
                return orig(self, *args, **kwargs)

            return counted

        setattr(tavily.TavilyClient, name, wrap(original, name))


# --------------------------------------------------------------------------- 1. 분석 노드 (PDF 조사자료 + 웹 검색)


def _pseudo_pages(company: dict, page_type: str, analyses: tuple[str, ...]) -> list[dict]:
    """PDF 에서 옮겨 적은 사실을 RAG 페이지 모양으로 만든다 (노드는 page_type 별로 골라 쓴다)."""
    cid = company["company_id"]
    ids, texts = ["S901"], [LISTING[cid]]
    for name in analyses:
        text, source_ids = FACTS[cid][name]
        texts.append(text)
        ids += [i for i in source_ids if i not in ids]
    return [{"page": "PDF", "page_type": page_type, "topic_id": None, "company_id": cid, "company_name": company["company_name"],
             "source_ids": ids, "text": " ".join(texts)}]


def _with_pdf_pages(module, page_type: str, analyses: tuple[str, ...], company: dict):
    """module.get_company_pages 를 잠시 바꿔 실제 RAG 페이지 뒤에 PDF 페이지를 덧붙인다."""
    original = module.get_company_pages

    def patched(cid):
        real = list(original(cid))
        return real + (_pseudo_pages(company, page_type, analyses) if cid == company["company_id"] else [])

    module.get_company_pages = patched
    return lambda: setattr(module, "get_company_pages", original)


def run_analysis(name: str, company: dict, state: dict, dated_tech: bool = False) -> dict:
    """분석 노드 하나를 실제로 실행하고 그 기업의 분석 필드들을 돌려준다. dated_tech 면 기술 요약이 연도를 붙인 검색어를 쓴다."""
    from agents.competitor_compare import node as competitor
    from agents.founder_traction import node as founder
    from agents.market_eval import node as market
    from agents.tech_summary import node as tech

    cid = company["company_id"]
    restore = None
    if name == "tech_summary":
        restore_pages = _with_pdf_pages(tech, "company_tech", ("technology", "competition"), company)
        original_queries = tech.WEB_QUERIES
        if dated_tech:
            tech.WEB_QUERIES = DATED_TECH_QUERIES

        def restore():
            restore_pages()
            tech.WEB_QUERIES = original_queries

        run, fields = tech.run, ("technology",)
    elif name == "market_eval":
        restore, run, fields = _with_pdf_pages(market, "company_market", ("market",), company), market.run, ("market",)
    elif name == "competitor_compare":
        run, fields = competitor.run, ("competition",)
    else:
        run, fields = founder.run, ("team", "traction", "deal_terms")
    try:
        update = run(state)
    finally:
        if restore:
            restore()
    evaluation = update["evaluations"][cid]
    return {field: evaluation[field] for field in fields if field in evaluation}


ANALYSIS_ORDER = ("tech_summary", "market_eval", "competitor_compare", "founder_traction")


# --------------------------------------------------------------------------- 2. 채점


def score_company(state: dict) -> dict:
    """후보와 같은 assess 로 SAMPLES 번 채점하고 항목별 중앙값으로 합친다. 검증에 실패한 채점은 버리되 하나도 없으면 실패다."""
    def one(_):
        try:
            return decision_node.assess(copy.deepcopy(state))
        except Exception as exc:
            print(f"  채점 1회 실패(버림): {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            return None

    with ThreadPoolExecutor(max_workers=SAMPLES) as pool:
        samples = [s for s in pool.map(one, range(SAMPLES)) if s]
    if not samples:
        raise RuntimeError("채점이 모두 실패했다")
    merged = decision_node.merge_samples(samples)
    return {
        "samples": [{"scores": {k: v["score"] for k, v in s["assessment"].items()}, "total_score": (s["scorecard"] or {}).get("total_score")} for s in samples],
        "assessment": merged["assessment"],
        "scorecard": merged["scorecard"],
    }


# --------------------------------------------------------------------------- 실행


def run_company(company: dict, result: dict, save, redo: tuple[str, ...] = (), dated_tech: bool = False) -> dict:
    """한 기업을 끝까지 처리한다. result 에 이미 있는 노드는 다시 실행하지 않고(웹 검색·LLM 호출 절약), 노드마다 save() 로 저장한다."""
    result.update(company_id=company["company_id"], company_name=company["company_name"], as_of=company["as_of"])
    cid = company["company_id"]
    query = f"{company['company_name']}을(를) 분석한다. (상장일 {company['as_of']} 기준 자료가 조사자료로 주어진다)"
    started = time.time()

    candidate = {**{k: company[k] for k in ("company_id", "company_name", "description", "domain")}, "retrieval_score": 0.0}
    state = initial_state(query)
    state.update(candidates=[candidate], current_candidate=cid, candidate_index=1)
    state["evaluations"] = {}

    analyses = result.setdefault("analyses", {})
    web = result.setdefault("web_calls", {})
    if redo:  # 다시 하는 단계의 이전 결과는 지우지 않고 history 에 남긴다 (좋은 결과만 고르지 않았다는 기록)
        result.setdefault("history", []).append({
            "redone": list(redo), "at": datetime.now().isoformat(timespec="seconds"), "dated_tech_queries": dated_tech,
            "analyses": {k: analyses[k] for k in redo if k in analyses}, "scoring": result.get("scoring"),
        })
        for name in redo:
            analyses.pop(name, None)
        result.pop("scoring", None)
        save()
    for name in ANALYSIS_ORDER:
        if name not in analyses:
            _progress(f"{cid} {name} (PDF 조사자료 + 웹 검색)")
            before = dict(_web_calls)
            state["evaluations"] = {cid: {k: v for fields in analyses.values() for k, v in fields.items()}}
            analyses[name] = run_analysis(name, company, state, dated_tech=dated_tech and name == "tech_summary")
            web[name] = {k: _web_calls[k] - before[k] for k in _web_calls}
            save()
    state["evaluations"] = {cid: {k: v for fields in analyses.values() for k, v in fields.items()}}

    if "scoring" not in result:
        _progress(f"{cid} 채점 ({SAMPLES}회, 항목별 중앙값)")
        result["scoring"] = score_company(state)
        result["dated_tech_queries"] = dated_tech or result.get("dated_tech_queries", False)

    result["seconds"] = round(result.get("seconds", 0) + time.time() - started)
    save()
    return result


def summarize(results: list[dict]) -> dict:
    totals = {r["company_id"]: (r["scoring"]["scorecard"] or {}).get("total_score") for r in results}
    scored = [t for t in totals.values() if t is not None]
    complete = len(scored) == len(REFERENCE_COMPANIES)
    return {
        "total_scores": totals,
        "mean": round(statistics.mean(scored), 1) if complete else None,
        "min": min(scored) if complete else None,
        "complete": complete,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="이 company_id 만 처리")
    parser.add_argument("--redo", nargs="*", default=(), choices=list(ANALYSIS_ORDER), help="--only 로 고른 기업의 이 노드부터 다시 실행 (채점은 항상 다시)")
    parser.add_argument("--dated-tech", action="store_true", help="기술 요약이 연도를 붙인 검색어를 쓴다 (--redo tech_summary 와 함께)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    threading.Thread(target=_watchdog, daemon=True).start()
    _count_web_calls()

    out = args.resume or args.out or (ROOT_DIR / "outputs" / "benchmark" / f"pdf_reference_{datetime.now(ZoneInfo('Asia/Seoul')):%Y%m%d_%H%M%S}.json")
    data = json.loads(out.read_text(encoding="utf-8")) if args.resume else {"companies": {}}
    out.parent.mkdir(parents=True, exist_ok=True)

    for company in REFERENCE_COMPANIES:
        if args.only and company["company_id"] not in args.only:
            continue
        cid = company["company_id"]

        def save():
            out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")  # 단계마다 저장 (중간 실패에도 결과 보존)

        run_company(company, data["companies"].setdefault(cid, {}), save, redo=tuple(args.redo), dated_tech=args.dated_tech)
        sc = data["companies"][cid]["scoring"]["scorecard"]
        print(f"  → {company['company_name']}: 핵심 점수 {sc['total_score'] if sc else '없음(필수 항목 미산정)'} | 웹 호출 누계 {_web_calls}", flush=True)

    done = [data["companies"][c["company_id"]] for c in REFERENCE_COMPANIES if c["company_id"] in data["companies"]]
    data["web_calls_this_run"] = dict(_web_calls)
    data["summary"] = summarize(done) if len(done) == len(REFERENCE_COMPANIES) else {"complete": False}
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n결과 저장: {out}\n요약: {data['summary']}", flush=True)


if __name__ == "__main__":
    main()
