"""Founder traction 웹 조사 흐름을 조율하는 에이전트.

Search -> identify founders -> read profile URLs -> collect supporting sources
-> analyze three sections -> review individual facts -> GraphState update.
No scores, no changes to shared utilities, no automatic full-output retries.
"""
from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from core.state import GraphState, evaluation_update, make_analysis

from .analysis_helpers import (
    build_verified_fact_summary,
    estimate_runway,
    valid_evidence,
)
from .prompts import SYSTEM
from .schemas import Discovery, Draft, Fact, Review
from .settings import AgentConfig, SECTIONS, VERSION
from .source_collection import build_sources, collect_pages
from .web_research import profile_url, read_pages, web_search

LOG = logging.getLogger(__name__)


class FounderPerformanceAgent:
    def __init__(self, llm=None, search_fn: Callable = web_search,
                 page_reader: Callable = read_pages, config: AgentConfig | None = None):
        if llm is None:
            from core.llm import get_llm
            llm = get_llm()
        self.llm, self.search_fn, self.page_reader = llm, search_fn, page_reader
        self.config = config or AgentConfig()
        self.page_metadata: dict[str, dict] = {}

    def _ask(self, schema, instruction, payload):
        def call():
            # OpenAI strict JSON schema rejects free-form dict attributes.
            # Function calling + Pydantic parsing preserves the existing output contract.
            from langchain_openai.chat_models.base import BaseChatOpenAI
            options = {"method": "function_calling", "strict": False} if isinstance(self.llm, BaseChatOpenAI) else {}
            value = self.llm.with_structured_output(schema, **options).invoke([
                ("system", SYSTEM + "\n" + instruction),
                ("human", json.dumps(payload, ensure_ascii=False)),
            ])
            return value if isinstance(value, schema) else schema.model_validate(value)
        return call()

    def _search(self, queries):
        def fetch(q):
            return self.search_fn(q, max_results=self.config.results_per_query)
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            return [build_sources(rows, "search_snippet", self.config.snippet_chars)
                    for rows in pool.map(fetch, queries)]

    def _pages(self, urls):
        sources, statuses, metadata = collect_pages(
            urls,
            page_reader=self.page_reader,
            timeout=self.config.page_timeout,
            snippet_chars=self.config.snippet_chars,
        )
        self.page_metadata.update(metadata)
        return sources, statuses

    def _analyze(self, section, docs, context):
        sources = {d["source_id"]: d for d in docs}
        if not sources:
            return make_analysis(summary="확인 가능한 자료를 확보하지 못했습니다.",
                                 risks=["추가 자료 확인 필요"], details={"facts": [], "status": "no_sources"})
        draft = self._ask(Draft,
            f"{section}: {SECTIONS[section]}을 분석. fact id와 statement id는 각각 고유하게. "
            "각 사실은 원문 evidence 필수. statements는 사실 id를 참조하며 요약/강점/위험/피어비교 작성. "
            "분석 범위 밖 사실은 제외. 기업 본인과 피어 사실은 subject로 구분. "
            "team은 page_extract 중 확인된 LinkedIn 프로필 본문을 우선 읽어 "
            "Experience/Education/Skills/Publications를 정리하고 전문성/실행 경험을 분석. "
            "검색 발췌만 확보했다면 프로필 본문을 읽었다고 말하지 말라. "
            "프로필에 없는 논문·경력은 공식 약력/학술 자료로 보완하고 각각 출처를 구분. "
            "논문 전체 목록이나 전체 팀이라고 단정 금지. 피어 우열은 비교 가능한 양측 fact가 있을 때만. "
            "런웨이는 출처가 명시한 기준일 추정만 수집하고 현재 런웨이로 단정 금지. "
            "unverified_publication은 역량 평가의 근거로 사용 금지. "
            "같은 사실은 한 번만 기록하고 불확실한 속성만 생략하라.",
            {"context": context, "sources": list(sources.values()),
             "instruction_detail": "LinkedIn 페이지의 전체 수집 본문을 여러 page_extract chunk로 제공한다. "
             "모든 chunk를 검토하고 각 Experience/Education/Skills/Publication 항목과 "
             "창업자가 직접 작성/공유한 중요한 게시물·발표·기술 의견·제품 시연·협업/고객 반응을 "
             "각각 linkedin_activity fact로 추출한다. 게시물 날짜·유형·주제를 attributes에 둔다. "
             "게시물 본문을 과도하게 합쳐 누락하지 말고 의미 단위별 사실을 기록한다. "
             "타인이 작성한 댓글/게시물은 CEO 성과로 귀속 금지. 회사 계정 게시물은 CEO 개인 활동과 구분. "
             "원문에서 확인 가능한 항목은 누락 없이 추출하되 반복/광고성 문구는 요약하고 원문 evidence 유지."})
        # Validate locally before one semantic review. Never discard a whole section.
        facts, seen = [], set()
        for f in draft.facts:
            if f.id not in seen and valid_evidence(f.evidence, sources):
                facts.append(f)
                seen.add(f.id)
        valid_ids = {f.id for f in facts if f.category != "unverified_publication"}
        statements = [s for s in draft.statements if s.fact_ids and set(s.fact_ids) <= valid_ids]
        review = self._ask(Review,
            "항목별 근거 검증. 인용이 사실과 모든 속성을 지지하고 인물/기업 귀속이 맞는 fact id만 승인. "
            "논문 confirmed 역할인 publication은 이름 외 동일인 연결 근거를 요구. "
            "중복 사실은 하나만 승인. 오류가 있는 항목만 거절하고 정상 항목은 유지. "
            "statements는 승인 사실로 뒷받침되는 추론/서술만 승인. 피어 우열에는 양측 비교 근거 필요. "
            "출처가 외부라는 것과 여러 독립 출처로 교차 검증됐다는 것을 혼동하지 말라. "
            "거절 이유는 reasons에 간결하게 작성. 원문 인용이 있다는 이유만으로 승인하지 말라.",
            {"context": context, "facts": [f.model_dump() for f in facts],
             "statements": [s.model_dump() for s in statements], "sources": list(sources.values())})
        facts = [f for f in facts if f.id in review.accepted_fact_ids]
        accepted = {f.id for f in facts if f.category != "unverified_publication"}
        statements = [s for s in statements if s.id in review.accepted_statement_ids and set(s.fact_ids) <= accepted]
        used = {e.source_id for f in facts for e in f.evidence}
        by_kind = lambda k: list(dict.fromkeys(s.text for s in statements if s.kind == k))
        if review.reasons:
            LOG.debug("%s 검토 제외 사유: %s", section, review.reasons)
        target_subjects = set(context["founder_names"] if section == "team" else [context["company"]["company_name"]])
        missing = [cat for cat in {"team": ["career", "education", "skill", "publication", "execution", "linkedin_activity"],
                   "traction": ["customer", "revenue", "commercialization"],
                   "deal_terms": ["terms", "cash", "burn", "runway", "milestone"]}[section]
                   if not any(f.category == cat and f.subject in target_subjects for f in facts)]
        runway = estimate_runway(facts, context["company"]["company_name"], context["as_of"]) if section == "deal_terms" else None
        if runway and "runway" in missing:
            missing.remove("runway")
        reviewed_summaries = by_kind("summary")
        if reviewed_summaries:
            summary = "\n".join(reviewed_summaries)
            summary_basis = {
                "mode": "reviewed_statement",
                "statement_ids": [s.id for s in statements if s.kind == "summary"],
            }
        else:
            summary, summary_fact_ids = build_verified_fact_summary(
                section, facts, target_subjects, missing
            )
            summary_basis = {
                "mode": "verified_fact_fallback",
                "fact_ids": summary_fact_ids,
            }
        return make_analysis(
            summary=summary,
            strengths=by_kind("strength"), risks=by_kind("risk") + (["일부 항목 근거 검증 실패"] if len(facts) < len(draft.facts) else []),
            evidence=[sources[s]["url"] for s in sorted(used)],
            details={"facts": [f.model_dump() for f in facts], "missing_categories": missing,
                     "peer_comparison": by_kind("peer_comparison") or ["비교 가능한 근거 부족"],
                     "sources": [{k: v for k, v in sources[s].items() if k != "content"} for s in sorted(used)],
                     "status": "partial" if facts else "unverified", "summary_basis": summary_basis,
                     **({"runway_estimate": runway} if section == "deal_terms" else {})})

    def __call__(self, state: GraphState):
        self.page_metadata = {}
        cid = state.get("current_candidate")
        company = next((c for c in state.get("candidates", []) if c["company_id"] == cid), None)
        if company is None:
            raise ValueError("current_candidate에 해당하는 candidates 항목이 필요합니다.")
        name = company["company_name"]
        prior = state.get("evaluations", {}).get(cid, {})
        peers = prior.get("competition", {}).get("details", {}).get("peer_group", [])
        peers = list(dict.fromkeys(p for p in peers if isinstance(p, str) and p != name))[:self.config.max_peers]
        # 1. Identify names/roles before searching any person's LinkedIn page.
        initial = sum(self._search([f'"{name}" CEO leadership official biography',
                                    f'"{name}" 대표이사 창업자 이름']), [])
        found = self._ask(Discovery,
            "현재 CEO/대표이사 이름을 우선 식별하고 확인된 창업자도 포함. "
            "CEO라는 이유로 창업자라고 쓰지 말고 role에 실제 역할을 구분. CEO를 먼저 정렬. "
            "한글/영문 별칭은 출처에 있는 것만. 이 단계 linkedin_url은 null. "
            "이름·역할의 원문 evidence 필수. 과거 CEO와 현재 CEO를 구분.",
            {"company": company, "sources": initial}) if initial else Discovery()
        available = {d["source_id"]: d for d in initial}
        founders, names = [], set()
        for f in found.founders:
            if f.name.casefold() in names or not valid_evidence(f.evidence, available):
                continue
            f.linkedin_url = None
            founders.append(f)
            names.add(f.name.casefold())
        founders = founders[:self.config.max_founders]
        # 2. Explicitly search LinkedIn using each identified person's name + company.
        profile_queries = [f'"{f.name}" "{name}" site:linkedin.com/in/' for f in founders]
        profile_hits = sum(self._search(profile_queries), []) if profile_queries else []
        if profile_hits:
            observed = {profile_url(u) for d in profile_hits for u in [d["url"]] +
                        re.findall(r'https?://[^\s<>"\)]+', d["content"])} - {None}
            resolved = self._ask(Discovery,
                "주어진 CEO/창업자 각각의 LinkedIn 본인 프로필 URL만 선택. "
                "observed_urls 중 이름과 회사/경력이 연결되는 /in/ 주소만 허용. "
                "다른 직원/동명이인 제외. 입력 name을 그대로 유지. 연결 근거 evidence 필수. "
                "확인 불가하면 linkedin_url=null.",
                {"company": company, "people": [f.model_dump() for f in founders],
                 "observed_urls": sorted(observed), "sources": profile_hits})
            source_map = {d["source_id"]: d for d in profile_hits}
            for f in founders:
                match = next((r for r in resolved.founders if r.name == f.name
                              and valid_evidence(r.evidence, source_map)), None)
                if match and profile_url(match.linkedin_url) in observed:
                    f.linkedin_url = profile_url(match.linkedin_url)
        # 3. Fetch the selected profile URL, not a search summary of that URL.
        profile_docs, statuses = self._pages([f.linkedin_url for f in founders if f.linkedin_url])
        queries = [f'"{name}" 고객 매출 계약 양산 revenue customers',
                   f'"{name}" investment funding terms cash burn runway 투자 조건 목표']
        for f in founders:
            alias = next((a for a in f.aliases if re.search("[A-Za-z]", a)), f.name)
            queries += [f'"{alias}" "{name}" official biography career education LinkedIn',
                        f'"{alias}" "{name}" publications paper DOI research affiliation']
        queries += [f'"{p}" founders customers revenue funding runway' for p in peers]
        batches = self._search(queries)
        urls = []
        for i in range(len(founders)):
            urls += [d["url"] for d in batches[2 + 2*i] if not profile_url(d["url"])][:1]
        official, _ = self._pages(urls)
        own_profiles = {f.linkedin_url for f in founders if f.linkedin_url}
        team_docs = profile_docs + initial + profile_hits + official + sum(batches[2:2 + 2*len(founders)], [])
        team_docs = [d for d in team_docs if not profile_url(d["url"]) or profile_url(d["url"]) in own_profiles]
        peer_docs = sum(batches[2 + 2*len(founders):], [])
        context = {"company": company, "founder_names": [f.name for f in founders],
                   "founders": [f.model_dump() for f in founders], "peers": peers,
                   "profile_collection": statuses,
                   "prior_hints": {k: prior.get(k, {}).get("summary", "") for k in ("technology", "market", "competition")},
                   "as_of": datetime.now(ZoneInfo("Asia/Seoul")).date().isoformat()}
        docs = {"team": team_docs, "traction": batches[0], "deal_terms": batches[1]}
        with ThreadPoolExecutor(max_workers=self.config.workers) as pool:
            results = list(pool.map(lambda s: self._analyze(s, docs[s] + peer_docs, context), SECTIONS))
        output = dict(zip(SECTIONS, results))
        output["team"]["details"]["linkedin_pages"] = [
            {"url": url, "status": status, **self.page_metadata.get(url, {})}
            for url, status in statuses.items()]
        output["team"]["details"]["members"] = [
            {"name": f.name, "aliases": f.aliases, "role": f.role, "linkedin_url": f.linkedin_url,
             "linkedin_content": statuses.get(f.linkedin_url, "not_found")} for f in founders]
        for result in output.values():
            result["details"].update(agent_version=VERSION, as_of=context["as_of"])
        LOG.info("founder_traction v%s complete", VERSION)
        return evaluation_update(cid, **output)


def run(state: GraphState) -> dict:
    return FounderPerformanceAgent()(state)
