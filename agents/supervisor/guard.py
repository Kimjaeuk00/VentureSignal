"""
하위 에이전트 실행 보호 래퍼

분석 노드가 예외를 던져도 그래프가 멈추지 않게 한다. 예외는 State 의 제어 필드에 기록하고
Supervisor 가 재시도(상한 내) 또는 제외(미확인으로 진행) 중 하나를 정한다.
노드 자체는 수정하지 않는다 (graph/builder.py 가 노드를 등록할 때 감싼다).
"""

import logging
from typing import Callable

from agents.supervisor.audit import log_decision
from core.state import GraphState, agent_key

logger = logging.getLogger(__name__)

ERROR_MAX_CHARS = 300


def guarded(agent: str, fn: Callable[[GraphState], dict]) -> Callable[[GraphState], dict]:
    def run(state: GraphState) -> dict:
        try:
            return fn(state)
        except Exception as exc:
            cid = state["current_candidate"] or ""
            message = f"{agent}: {type(exc).__name__}: {exc}"[:ERROR_MAX_CHARS]
            logger.exception("%s 실행 실패 (%s)", agent, cid)
            log_decision(state, agent, "failed", message)
            return {"last_error": message, "node_status": {agent_key(cid, agent): "failed"}}

    run.__name__ = getattr(fn, "__name__", "run")
    return run
