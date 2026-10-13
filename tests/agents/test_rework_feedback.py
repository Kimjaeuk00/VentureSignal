"""재작업 피드백: Supervisor 가 지적한 부족한 항목이 재작업되는 에이전트의 검색어·프롬프트에 반영된다 (LLM·웹검색 호출 없음)."""

import json

from core.rework import MAX_NOTE_CHARS, MAX_NOTES, MAX_QUERIES, focus_text, rework_notes_for, rework_queries_for
from core.state import initial_state


def _state(notes=None):
    state = initial_state("q")
    state["current_candidate"] = "C1"
    if notes is not None:
        state["rework_notes"] = {"C1:tech_summary": notes}
    return state


def test_no_notes_means_no_change_for_first_run():
    assert rework_notes_for(_state(), "C1", "tech_summary") == []
    assert focus_text([]) == ""


def test_notes_are_trimmed_and_capped():
    notes = rework_notes_for(_state(["양산 고객\n\n  수율 " + "x" * 500, "b", "c", "d"]), "C1", "tech_summary")
    assert len(notes) == MAX_NOTES and all(len(n) <= MAX_NOTE_CHARS for n in notes) and "\n" not in notes[0]


def test_notes_are_per_company_and_agent():
    state = _state(["양산 고객"])
    assert rework_notes_for(state, "C2", "tech_summary") == [] and rework_notes_for(state, "C1", "market_eval") == []


def test_focus_text_lists_the_missing_items():
    text = focus_text(["양산 고객", "독립 실측"])
    assert "재작업 지시" in text and "- 양산 고객" in text and "- 독립 실측" in text


def test_tech_summary_adds_queries_from_notes(monkeypatch):
    from agents.tech_summary import node as tech
    from core.sources import SourceBook

    asked = []
    monkeypatch.setattr(tech, "web_search", lambda q, max_results=3: asked.append(q) or [])

    tech._web_search("DEEPX", "DX-M1", SourceBook(canonicalize=True))
    first_run = list(asked)
    asked.clear()
    tech._web_search("DEEPX", "DX-M1", SourceBook(canonicalize=True), ["양산 고객", "수율"])

    assert len(asked) == len(first_run) + 2  # 고정 검색어는 그대로, 부족한 항목마다 검색어가 더해진다
    assert "DEEPX DX-M1 양산 고객" in asked and "DEEPX DX-M1 수율" in asked


def test_tech_summary_web_prompt_carries_the_focus():
    from prompts.tech_summary import WEB_PROMPT

    text = WEB_PROMPT.format(company_name="DEEPX", company_id="C1", draft="{}", context="", web_context="", focus=focus_text(["수율"]))
    assert "재작업 지시" in text and "- 수율" in text
    assert "재작업 지시" not in WEB_PROMPT.format(company_name="DEEPX", company_id="C1", draft="{}", context="", web_context="", focus="")


def test_competitor_compare_user_message_carries_the_focus(monkeypatch):
    from agents.competitor_compare import node as comp

    captured = {}

    class FakeAgent:
        def invoke(self, payload, config=None):
            captured["content"] = payload["messages"][0]["content"]

    monkeypatch.setattr(comp, "init_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(comp, "create_agent", lambda **kwargs: FakeAgent())

    state = initial_state("q")
    state["candidates"] = [{"company_id": "C1", "company_name": "Co", "description": "d", "domain": "NPU", "retrieval_score": 0.0}]
    state["current_candidate"] = "C1"

    comp.run(state)  # 첫 실행: 지시 없음
    assert "재작업 지시" not in captured["content"]

    state["rework_notes"] = {"C1:competitor_compare": ["경쟁 제품의 전력 비교 조건"]}
    comp.run(state)  # 재작업: 부족한 항목이 사용자 메시지에 실린다
    assert "재작업 지시" in captured["content"] and "경쟁 제품의 전력 비교 조건" in captured["content"]
    assert json.dumps("Co", ensure_ascii=False) in captured["content"]


# --------------------------------------------------------------------------- market_eval / founder_traction


def test_market_eval_stage_text_carries_the_focus_only_on_rework():
    from agents.market_eval.node import _stage

    assert _stage("RAG 초기 분석", [], plan=True) == "RAG 초기 분석"  # 첫 실행은 기존 문구 그대로
    text = _stage("RAG 초기 분석", ["국내 시장 규모"], plan=True)
    assert text.startswith("RAG 초기 분석") and "- 국내 시장 규모" in text and "web_search_queries" in text
    assert "web_search_queries" not in _stage("웹 보완 후 최종 분석", ["국내 시장 규모"])  # 최종 분석은 검색 계획을 다시 만들지 않는다


class _FakeFounderLLM:
    """_ask 가 부르는 with_structured_output(...).invoke(...) 를 흉내 낸다. 받은 지시문을 기록한다."""

    def __init__(self):
        self.instructions = []

    def with_structured_output(self, schema, **options):
        self.schema = schema
        return self

    def invoke(self, messages):
        self.instructions.append(messages[0][1])
        return self.schema()


def _founder_agent(search_log=None):
    from agents.founder_traction.agent import FounderPerformanceAgent

    llm = _FakeFounderLLM()

    def search(query, max_results=5):
        if search_log is not None:
            search_log.append(query)
        return [{"url": f"https://example.com/{abs(hash(query)) % 10000}", "title": query, "content": f"본문 {query}"}]

    return FounderPerformanceAgent(llm=llm, search_fn=search, page_reader=lambda *a, **k: ({}, {}, {})), llm


def test_founder_traction_draft_instruction_carries_the_focus_on_rework():
    agent, llm = _founder_agent()
    docs = [{"source_id": "S-1", "url": "https://example.com/a", "title": "t", "content": "본문", "scope": "x"}]
    context = {"company": {"company_name": "Co"}, "founder_names": [], "rework_focus": ["유상 고객·매출 규모"]}

    agent._analyze("traction", docs, context)
    assert any("재작업 지시" in text and "유상 고객·매출 규모" in text for text in llm.instructions)

    llm.instructions.clear()
    agent._analyze("traction", docs, {**context, "rework_focus": []})
    assert not any("재작업 지시" in text for text in llm.instructions)  # 첫 실행은 기존과 같다


def test_founder_traction_runs_extra_searches_in_a_separate_batch_on_rework():
    queries = []
    agent, _ = _founder_agent(queries)
    state = initial_state("q")
    state["candidates"] = [{"company_id": "C1", "company_name": "Co", "description": "d", "domain": "NPU", "retrieval_score": 0.0}]
    state["current_candidate"] = "C1"

    agent(state)
    first_run = list(queries)
    assert not any("유상 PoC" in q for q in first_run)

    queries.clear()
    state["rework_notes"] = {"C1:founder_traction": ["유상 PoC 매출"]}
    agent(state)
    assert any('"Co" 유상 PoC 매출' == q for q in queries)  # 부족한 항목 검색어가 더해진다
    assert all(q in queries for q in first_run if "유상" not in q)  # 기존 검색어는 그대로 유지된다


# --------------------------------------------------------------------------- Supervisor 가 정한 검색어


def _state_with(note):
    state = initial_state("q")
    state["current_candidate"] = "C1"
    state["rework_notes"] = {"C1:tech_summary": note}
    return state


def test_queries_get_the_company_name_and_are_capped():
    state = _state_with({"missing": ["x"], "queries": ["DX-M1 customers", "DEEPX 수율", "a", "b", "c"]})
    queries = rework_queries_for(state, "C1", "tech_summary", "DEEPX")
    assert queries[:2] == ["DEEPX DX-M1 customers", "DEEPX 수율"]  # 이름이 없으면 붙이고 있으면 그대로 둔다
    assert len(queries) == MAX_QUERIES


def test_queries_are_empty_without_supervisor_queries():
    assert rework_queries_for(_state_with({"missing": ["x"], "queries": []}), "C1", "tech_summary", "DEEPX") == []
    assert rework_queries_for(initial_state("q"), "C1", "tech_summary", "DEEPX") == []


def test_old_list_format_is_still_readable():
    state = _state_with(["양산 고객"])
    assert rework_notes_for(state, "C1", "tech_summary") == ["양산 고객"] and rework_queries_for(state, "C1", "tech_summary") == []


def test_focus_text_lists_supervisor_queries():
    text = focus_text(["수율"], ["DEEPX DX-M1 yield"])
    assert "- 수율" in text and "Supervisor 제안" in text and "- DEEPX DX-M1 yield" in text
    assert "Supervisor 제안" not in focus_text(["수율"])


def test_tech_summary_uses_supervisor_queries_instead_of_the_note_text(monkeypatch):
    from agents.tech_summary import node as tech
    from core.sources import SourceBook

    asked = []
    monkeypatch.setattr(tech, "web_search", lambda q, max_results=3: asked.append(q) or [])
    tech._web_search("DEEPX", "DX-M1", SourceBook(canonicalize=True), ["양산 고객 단계와 시점"], ["DEEPX DX-M1 customers"])

    assert "DEEPX DX-M1 customers" in asked
    assert not any("양산 고객 단계와 시점" in q for q in asked)  # 검색어가 있으면 설명문을 검색어로 쓰지 않는다


def test_founder_traction_searches_with_the_supervisor_queries(monkeypatch):
    queries = []
    agent, _ = _founder_agent(queries)
    state = initial_state("q")
    state["candidates"] = [{"company_id": "C1", "company_name": "Co", "description": "d", "domain": "NPU", "retrieval_score": 0.0}]
    state["current_candidate"] = "C1"
    state["rework_notes"] = {"C1:founder_traction": {"missing": ["유상 PoC 매출 근거"], "queries": ["Co revenue customers 2025"]}}
    agent(state)
    assert "Co revenue customers 2025" in queries
    assert not any("유상 PoC 매출 근거" in q for q in queries)


def test_market_stage_text_includes_supervisor_queries():
    from agents.market_eval.node import _stage

    assert "Co market size" in _stage("RAG 초기 분석", ["시장 규모"], plan=True, queries=["Co market size"])


def test_competitor_compare_user_message_carries_the_supervisor_queries(monkeypatch):
    from agents.competitor_compare import node as comp

    captured = {}

    class FakeAgent:
        def invoke(self, payload, config=None):
            captured["content"] = payload["messages"][0]["content"]

    monkeypatch.setattr(comp, "init_chat_model", lambda **kwargs: object())
    monkeypatch.setattr(comp, "create_agent", lambda **kwargs: FakeAgent())
    state = initial_state("q")
    state["candidates"] = [{"company_id": "C1", "company_name": "Co", "description": "d", "domain": "NPU", "retrieval_score": 0.0}]
    state["current_candidate"] = "C1"
    state["rework_notes"] = {"C1:competitor_compare": {"missing": ["전력 비교 조건"], "queries": ["Hailo-8 vs Co power efficiency"]}}
    comp.run(state)
    assert "- Hailo-8 vs Co power efficiency" in captured["content"] and "Supervisor 제안" in captured["content"]
