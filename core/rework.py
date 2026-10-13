"""재작업 피드백: Supervisor 가 남긴 "부족한 항목"과 검색어를 재작업되는 에이전트가 읽는 곳.

Supervisor 는 재작업을 요청할 때 state["rework_notes"]["{company_id}:{agent}"] 에
{"missing": [부족한 항목...], "queries": [웹 검색어...]} 를 남긴다.
첫 실행에는 없으므로 항상 빈 목록을 돌려줄 수 있어야 한다 (노트가 없으면 에이전트는 기존과 똑같이 동작한다).
"""

from core.state import GraphState, agent_key

MAX_NOTES = 3
MAX_NOTE_CHARS = 120
MAX_QUERIES = 3
MAX_QUERY_CHARS = 100


def _note(state: GraphState, cid: str, agent: str) -> dict:
    note = (state.get("rework_notes") or {}).get(agent_key(cid, agent)) or {}
    if isinstance(note, list):  # 이전 형식(부족한 항목만)도 읽는다
        return {"missing": note, "queries": []}
    return note if isinstance(note, dict) else {}


def _clean(items, limit: int, count: int) -> list[str]:
    return [" ".join(str(i).split())[:limit] for i in (items or []) if str(i).strip()][:count]


def rework_notes_for(state: GraphState, cid: str, agent: str) -> list[str]:
    """근거가 부족하다고 평가한 항목."""
    return _clean(_note(state, cid, agent).get("missing"), MAX_NOTE_CHARS, MAX_NOTES)


def rework_queries_for(state: GraphState, cid: str, agent: str, company_name: str = "") -> list[str]:
    """Supervisor 가 정한 웹 검색어. 회사 이름이 빠져 있으면 앞에 붙인다 (동명 검색을 피한다)."""
    queries = _clean(_note(state, cid, agent).get("queries"), MAX_QUERY_CHARS, MAX_QUERIES)
    if company_name:
        queries = [q if company_name.casefold() in q.casefold() else f"{company_name} {q}" for q in queries]
    return list(dict.fromkeys(queries))


def focus_text(notes: list[str], queries: list[str] = ()) -> str:
    """프롬프트에 붙일 재작업 지시. 부족한 항목이 없으면 빈 문자열."""
    if not notes:
        return ""
    text = (
        "\n[재작업 지시] 이전 결과에서 근거가 부족하다고 지적된 항목이다. 이 항목을 우선 보강하라. "
        "찾을 수 없으면 지어내지 말고 미확인으로 남긴다.\n" + "\n".join(f"- {n}" for n in notes) + "\n"
    )
    if queries:
        text += "참고할 웹 검색어(Supervisor 제안):\n" + "\n".join(f"- {q}" for q in queries) + "\n"
    return text
