"""
Supervisor (조정 계층)

입력: State 의 제어 메타데이터(current_candidate, candidate_index, node_status, rework_count, step_count, max_steps)와
      evaluations 의 분석 결과·판정
출력: 제어 필드만 — next_agent, step_count, last_decision, node_status, rework_count
      (페이로드 필드는 건드리지 않는다)

하위 에이전트는 항상 Supervisor 로 돌아오고, 다음에 어디로 갈지는 이 노드가 State 를 보고 정한다.
graph/builder.py 의 add_conditional_edges 가 next_agent 로 분기하므로 실행 순서는 코드에 고정되어 있지 않다.

분석 단계(후보를 고른 뒤 투자 판단이 나오기 전)에서는 LLM 이 돌아온 보고를 평가하고 다음 에이전트를 고른다
(llm_policy.py, 프롬프트는 prompts/supervisor.py). 코드는 종료·안전 불변 조건만 검사한다.
LLM 호출이 실패하거나 불변 조건을 어기는 응답이 반복되면 아래의 규칙 기반 로직(_decide)이 대신한다.

규칙 기반 결정 순서 (LLM 을 쓰지 않거나 대체할 때, 위에서부터 처음 맞는 것):
0. 보고서 단계 (report_writer·quality_check 도 하위 노드라 Supervisor 로만 돌아온다)
   - 품질 평가가 돌아왔다: 통과이거나 평가 결과가 없으면 종료 / 재작성 상한이면 종료 / 미달이면 보고서를 다시 쓴다
   - 보고서 작성이 돌아왔다: 품질 평가로 보낸다
1. step_count 가 max_steps 를 넘으면 보고서로 간다 (종료 보장). 보고서 단계가 이미 시작됐다면 적용하지 않는다
2. 후보가 없으면 보고서
3. 후보를 아직 꺼내지 않았으면 후보 선택
4. 현재 후보의 판정이 있으면: INVEST → 보고서 / HOLD 이고 후보가 남음 → 후보 선택 / 소진 → 보고서
5. 분석 결과의 근거가 부족하거나 실행이 실패했고 재작업 상한 미만이면 그 에이전트 재작업 (상한이면 제외하고 진행)
6. 아직 실행하지 않은 분석 에이전트가 있으면 그 에이전트
7. 분석이 끝났으면 투자 판단

결정 로그는 State 가 아니라 logger 로 내보낸다(trace_id 로 State 와 연결). State 에는 마지막 결정 한 건만 남긴다.
"""

from typing import Optional

from agents.supervisor import llm_policy
from agents.supervisor.audit import log_decision
from agents.quality_check.policy import MAX_REVISIONS
from agents.supervisor.policy import (
    ANALYSIS_AGENTS,
    CANDIDATE_SELECT,
    END_TARGET,
    INVESTMENT_DECISION,
    MAX_REWORK,
    QUALITY_CHECK,
    REPORT_STAGE_KEYS,
    REPORT_WRITER,
    assess_agent,
)
from core.state import GraphState, agent_key


def _report_stage(state: GraphState) -> Optional[tuple[str, str, dict]]:
    """보고서 단계에서 돌아온 노드를 보고 다음을 정한다. 보고서 단계가 아니면 None. (다음 노드, 사유, node_status 변경분)"""
    writer_key, check_key = REPORT_STAGE_KEYS[REPORT_WRITER], REPORT_STAGE_KEYS[QUALITY_CHECK]
    status = state["node_status"]

    if status.get(check_key) == "pending":  # 품질 평가가 돌아왔다
        verdict, attempts = state["quality_verdict"], state["quality_attempts"]
        done = {check_key: "done"}
        if verdict is None:
            return END_TARGET, "품질 평가 결과가 없어 보고서를 그대로 확정한다", done
        if verdict["passed"]:
            return END_TARGET, "품질 평가를 통과해 종료한다", done
        if attempts > MAX_REVISIONS:
            feedback = " / ".join(verdict["feedback"])[:200]
            return END_TARGET, f"품질 미달이지만 재작성 상한({MAX_REVISIONS}회)이라 종료한다: {feedback}", done
        feedback = " / ".join(verdict["feedback"])[:200]
        return REPORT_WRITER, f"보고서 품질 평가 미달로 다시 쓴다 (평가 {attempts}회): {feedback}", done

    if status.get(writer_key) == "pending":  # 보고서 작성이 돌아왔다
        return QUALITY_CHECK, "보고서가 작성돼 품질 평가로 보낸다", {writer_key: "done"}

    return None


def _decide(state: GraphState) -> tuple[str, str, dict, dict]:
    """(다음 노드, 사유, node_status 변경분, rework_count 변경분)"""
    candidates = state["candidates"]
    if not candidates:
        return REPORT_WRITER, "후보가 없어 바로 보고서로 간다", {}, {}

    cid = state["current_candidate"]
    if cid is None:
        return CANDIDATE_SELECT, "아직 후보를 꺼내지 않았다", {}, {}

    evaluation = state["evaluations"].get(cid, {})

    decision = evaluation.get("decision")
    if decision == "INVEST":
        return REPORT_WRITER, f"{cid} 가 INVEST 라 평가를 끝내고 보고서로 간다", {}, {}
    if decision == "HOLD":
        if state["candidate_index"] < len(candidates):
            return CANDIDATE_SELECT, f"{cid} 가 HOLD 이고 후보가 남아 다음 후보로 간다", {}, {}
        return REPORT_WRITER, f"{cid} 가 HOLD 이고 후보를 모두 소진해 보류 보고서로 간다", {}, {}

    status: dict[str, str] = {}
    rework: dict[str, int] = {}
    pending: Optional[str] = None

    for agent, fields in ANALYSIS_AGENTS.items():
        key = agent_key(cid, agent)
        if fields[0] not in evaluation:
            if state["node_status"].get(key) == "excluded":
                continue  # 이미 제외했다. 다시 시키지 않고 미확인으로 진행한다
            if state["node_status"].get(key) == "failed":
                # 실행이 예외로 끝났다 (agents/supervisor/guard.py). 재시도 상한 안에서만 다시 시킨다.
                used = state["rework_count"].get(key, 0)
                if used < MAX_REWORK:
                    rework[key] = used + 1
                    status[key] = "needs_rework"
                    return (
                        agent,
                        f"{agent} 실행 실패로 재시도한다 ({used + 1}/{MAX_REWORK}): {state['last_error']}",
                        status,
                        rework,
                    )
                status[key] = "excluded"  # 이 항목은 미확인으로 두고 진행한다
                continue
            if pending is None:
                pending = agent
            status[key] = "pending"
            continue

        verdict, reason = assess_agent(evaluation, agent)
        used = state["rework_count"].get(key, 0)
        if verdict == "ok":
            status[key] = "done"
        elif used < MAX_REWORK:
            rework[key] = used + 1
            status[key] = "needs_rework"
            return (
                agent,
                f"{agent} 근거 부족으로 재작업을 요청한다 ({used + 1}/{MAX_REWORK}): {reason}",
                status,
                rework,
            )
        else:
            # 상한까지 다시 시켰지만 부족하다 — 이 항목은 미확인으로 두고 진행한다
            status[key] = "excluded"

    if pending is not None:
        return pending, f"{pending} 를 아직 실행하지 않았다", status, rework
    return INVESTMENT_DECISION, "분석이 끝나 투자 판단으로 간다", status, rework


def _decide_llm(state: GraphState) -> tuple[Optional[tuple[str, str, dict, dict, dict]], str]:
    """분석 단계에서 LLM 이 정한 다음 행동. 적용 대상이 아니거나 대체해야 하면 (None, 사유)."""
    if not llm_policy.ENABLED:
        return None, ""
    cid = state["current_candidate"]
    if not state["candidates"] or cid is None:
        return None, ""
    evaluation = state["evaluations"].get(cid, {})
    if evaluation.get("decision"):
        return None, ""  # 판정이 난 뒤의 전이(다음 후보·보고서)는 판정 결과로 정해진다

    statuses = llm_policy.analysis_statuses(state, cid, evaluation)
    if "failed" in statuses.values():
        return None, ""  # 실행 실패의 재시도·제외는 규칙으로 처리한다 (판단할 보고가 없다)

    result = llm_policy.decide(state, cid, evaluation, statuses)
    if result.decision is None:
        return None, result.note

    decision = result.decision
    target = decision.next_agent
    last = state["next_agent"]
    review = decision.review
    status = {agent_key(cid, a): ("done" if s == "done" else s) for a, s in statuses.items()}
    rework: dict[str, int] = {}
    notes: dict[str, dict] = {}

    verdict_text = ""
    if review is not None and review.agent == last and last in statuses:
        verdict_text = f"[{last} {'충분' if review.sufficient else '부족: ' + ', '.join(review.missing[:3])}] "
        if not review.sufficient and target != last:
            status[agent_key(cid, last)] = "excluded"  # 부족하지만 다시 시키지 않는다: 미확인으로 두고 진행

    if target != llm_policy.INVESTMENT_DECISION and statuses[target] == "done":  # 결과가 있는 에이전트를 다시 고르면 재작업
        key = agent_key(cid, target)
        rework[key] = state["rework_count"].get(key, 0) + 1
        status[key] = "needs_rework"
        # 부족하다고 평가한 항목을 재작업되는 에이전트에게 넘긴다 (평가가 target 의 것이 아니면 사유를 대신 쓴다)
        missing = list(review.missing) if review is not None and review.agent == target and review.missing else [decision.reason]
        notes[key] = {
            "missing": [m[:200] for m in missing[:llm_policy.LIST_ITEMS]],
            "queries": [q for q in decision.search_queries if q.strip()][:3],
        }

    return (target, f"{verdict_text}{decision.reason}", status, rework, notes), ""


def run(state: GraphState) -> dict:
    step = state["step_count"] + 1
    report_stage = _report_stage(state)
    started = REPORT_STAGE_KEYS[REPORT_WRITER] in state["node_status"]  # 보고서 단계가 이미 시작됐는가

    rework_notes: dict[str, dict] = {}
    if report_stage is not None:
        target, reason, node_status = report_stage
        rework_count = {}
    elif step > state["max_steps"] and not started:
        target, reason = REPORT_WRITER, f"step_count 가 max_steps({state['max_steps']}) 를 넘어 보고서로 강제 이동한다"
        node_status, rework_count = {}, {}
    else:
        llm_result, fallback_note = _decide_llm(state)
        if llm_result is not None:
            target, reason, node_status, rework_count, rework_notes = llm_result
        else:
            target, reason, node_status, rework_count = _decide(state)
            if fallback_note:
                reason = f"[규칙 기반으로 대신함: {fallback_note}] {reason}"

    # 보고서 단계로 보낸 노드는 돌아올 때까지 pending 으로 표시한다 (Supervisor 가 다시 진입하면 돌아온 것이다)
    if target in (REPORT_WRITER, QUALITY_CHECK):
        node_status = {**node_status, REPORT_STAGE_KEYS[target]: "pending"}

    log_decision(state, "supervisor", target, reason, step=step)

    update: dict = {
        "step_count": step,
        "next_agent": target,
        "last_decision": {"next": target, "reason": reason},
    }
    if node_status:
        update["node_status"] = node_status
    if rework_count:
        update["rework_count"] = rework_count
    if rework_notes:
        update["rework_notes"] = rework_notes
    return update


def route(state: GraphState) -> str:
    """add_conditional_edges 용 분기 함수. Supervisor 가 정한 next_agent 를 그대로 따른다."""
    return state["next_agent"]
