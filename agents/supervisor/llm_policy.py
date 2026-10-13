"""
Supervisor 의 LLM 판단 (분석 단계)

하위 에이전트가 돌아올 때마다 LLM 이 (1) 방금 온 보고가 충분한지 평가하고 (2) 다음에 누구에게 일을 시킬지 고른다.
실행 순서·의존 관계·재작업 기준은 프롬프트(prompts/supervisor.py)에 있고 코드에는 없다.

코드가 검사하는 것은 순서가 아니라 종료·안전에 필요한 불변 조건뿐이다 (validate).
- 필요한 분석이 끝나기 전에 투자 판단으로 가지 않는다 (분석 없이 채점하지 않는다)
- 재작업은 에이전트별 상한(MAX_REWORK) 안에서만, 이미 제외한 에이전트는 다시 시키지 않는다
위반하면 문제를 알려 한 번 다시 묻고, 그래도 안 되거나 LLM 호출이 실패하면 호출한 쪽(node.py)이 규칙 기반 로직으로 대신한다.

환경 변수 SUPERVISOR_LLM=off 면 LLM 을 쓰지 않고 규칙 기반으로만 동작한다 (비용을 아끼거나 테스트할 때).
"""

import logging
import os
from dataclasses import dataclass
from typing import Optional

from agents.supervisor.policy import ANALYSIS_AGENTS, INVESTMENT_DECISION, MAX_REWORK
from agents.supervisor.schemas import SupervisorDecision
from core.llm import get_llm
from core.state import GraphState, agent_key
from prompts.supervisor import RETRY_NOTE, SUPERVISOR_HUMAN, SUPERVISOR_SYSTEM

logger = logging.getLogger(__name__)

ENABLED = os.getenv("SUPERVISOR_LLM", "on").strip().lower() != "off"

# 설정값: 호출 비용과 응답 안정성을 보고 조정한다.
MAX_ASK = 2  # 불변 조건을 어긴 응답에 한 번 더 묻는다 (진입 한 번에 LLM 호출은 최대 이 횟수)
TEXT_CHARS = 300  # 보고 요약을 LLM 에 보여 줄 때 자르는 길이
LIST_ITEMS = 3  # 위험·미확인 항목을 보여 줄 개수


@dataclass
class LlmResult:
    decision: Optional[SupervisorDecision]
    note: str = ""  # decision 이 없을 때 규칙 기반으로 대신하는 이유


def analysis_statuses(state: GraphState, cid: str, evaluation: dict) -> dict[str, str]:
    """분석 에이전트별 상태: excluded / done(결과 있음) / failed / pending(아직 실행 안 함)."""
    node_status = state["node_status"]
    out = {}
    for agent, fields in ANALYSIS_AGENTS.items():
        status = node_status.get(agent_key(cid, agent))
        if status == "excluded":
            out[agent] = "excluded"
        elif fields[0] in evaluation:
            out[agent] = "done"
        elif status == "failed":
            out[agent] = "failed"
        else:
            out[agent] = "pending"
    return out


def rework_failures(state: GraphState, cid: str, evaluation: dict) -> dict[str, str]:
    """결과가 이미 있는데 실행이 실패로 기록된 에이전트 = 재작업 실행이 실패해 이전 결과가 그대로인 경우. {에이전트: 오류}"""
    out = {}
    for agent, fields in ANALYSIS_AGENTS.items():
        if fields[0] in evaluation and state["node_status"].get(agent_key(cid, agent)) == "failed":
            error = state["last_error"] or ""
            out[agent] = error if error.startswith(agent) else "오류 내용 없음"
    return out


def _cut(text, limit: int = TEXT_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _strings(value) -> list[str]:
    return [str(v) for v in value if isinstance(v, (str, int, float))] if isinstance(value, list) else []


def _describe_result(analysis: dict) -> str:
    details = analysis.get("details") or {}
    parts = [
        f"출처 {len([e for e in analysis.get('evidence') or [] if e])}건",
        f"강점 {len(analysis.get('strengths') or [])}개",
        f"위험 {len(analysis.get('risks') or [])}개",
    ]
    if details.get("status"):
        parts.append(f"상태표시 {details['status']}" + (f"({_cut(details['reason'], 80)})" if details.get("reason") else ""))
    text = " | ".join(parts) + f" | 요약: {_cut(analysis.get('summary', '')) or '없음'}"
    risks = _strings(analysis.get("risks"))[:LIST_ITEMS]
    if risks:
        text += " | 위험: " + "; ".join(_cut(r, 100) for r in risks)
    missing = [m for key in ("unverified", "missing_information") for m in _strings(details.get(key))][:LIST_ITEMS]
    if missing:
        text += " | 미확인: " + "; ".join(_cut(m, 100) for m in missing)
    return text


def summarize_agents(state: GraphState, cid: str, evaluation: dict, statuses: dict[str, str]) -> str:
    failures = rework_failures(state, cid, evaluation)
    lines = []
    for agent, fields in ANALYSIS_AGENTS.items():
        used = state["rework_count"].get(agent_key(cid, agent), 0)
        status = statuses[agent]
        if status == "pending":
            lines.append(f"- {agent}: 아직 실행하지 않음")
        elif status == "excluded":
            lines.append(f"- {agent}: 제외됨 (근거가 부족한 채로 확정, 미확인으로 처리)")
        else:
            body = " // ".join(f"[{field}] {_describe_result(evaluation.get(field) or {})}" for field in fields)
            warning = f" | ⚠ 직전 재작업 실행이 실패해 아래는 이전 결과 그대로다 (오류: {_cut(failures[agent], 150)})" if agent in failures else ""
            lines.append(f"- {agent}: 완료, 재작업 {used}/{MAX_REWORK}{warning} | {body}")
    return "\n".join(lines)


def validate(decision: SupervisorDecision, state: GraphState, cid: str, statuses: dict[str, str]) -> str:
    """종료·안전 불변 조건 검사. 문제가 있으면 사유를, 없으면 빈 문자열."""
    target = decision.next_agent
    if target == INVESTMENT_DECISION:
        open_agents = [a for a, s in statuses.items() if s not in ("done", "excluded")]
        return f"아직 실행하지 않은 분석이 있어 투자 판단으로 갈 수 없다: {', '.join(open_agents)}" if open_agents else ""

    status = statuses[target]
    if status == "excluded":
        return f"{target} 는 이미 제외했다"
    if status == "done" and state["rework_count"].get(agent_key(cid, target), 0) >= MAX_REWORK:
        return f"{target} 의 재작업 상한({MAX_REWORK}회)에 이미 닿았다"
    return ""


def _ask(state: GraphState, cid: str, evaluation: dict, statuses: dict[str, str], problem: str) -> SupervisorDecision:
    info = next((c for c in state["candidates"] if c["company_id"] == cid), {})
    candidate = f"{info.get('company_name', cid)} ({cid}) — {_cut(info.get('description', ''), 200)} / 분야: {info.get('domain', '미확인')}"
    last = state["next_agent"] if state["next_agent"] in ANALYSIS_AGENTS else "없음 (이 후보의 첫 판단)"
    human = SUPERVISOR_HUMAN.format(
        candidate=candidate,
        query=_cut(state["query"], 200),
        step=state["step_count"] + 1,
        max_steps=state["max_steps"],
        remaining=len(state["candidates"]) - state["candidate_index"],
        agents=summarize_agents(state, cid, evaluation, statuses),
        last=last,
        retry=RETRY_NOTE.format(problem=problem) if problem else "",
    )
    messages = [("system", SUPERVISOR_SYSTEM.format(max_rework=MAX_REWORK)), ("human", human)]
    return get_llm().with_structured_output(SupervisorDecision).invoke(messages)


def decide(state: GraphState, cid: str, evaluation: dict, statuses: dict[str, str]) -> LlmResult:
    problem = ""
    for attempt in range(1, MAX_ASK + 1):
        try:
            decision = _ask(state, cid, evaluation, statuses, problem)
        except Exception as exc:
            logger.exception("Supervisor LLM 호출 실패 → 규칙 기반으로 대신한다")
            return LlmResult(None, f"LLM 호출 실패 ({type(exc).__name__})")
        problem = validate(decision, state, cid, statuses)
        if not problem:
            return LlmResult(decision)
        logger.warning("Supervisor LLM 이 허용되지 않는 선택을 했다 (%d/%d): %s", attempt, MAX_ASK, problem)
    return LlmResult(None, f"허용되지 않는 선택이 반복됨: {problem}")
