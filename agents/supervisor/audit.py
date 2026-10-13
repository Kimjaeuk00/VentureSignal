"""
결정 로그와 실행 상관 (관측성 계층)

결정 로그(누가 무엇을 왜 결정했는가)는 State 에 쌓지 않고 이 로거로 내보낸다. 한 줄이 JSON 한 건이고
trace_id 로 State(`state["trace_id"]`)와 LangSmith run metadata 에 연결된다.
State 에는 마지막 결정 한 건(last_decision)만 남아 재개 시 직전 사유를 확인하는 용도로 쓴다.

LangSmith 는 이 모듈이 호출하지 않는다. run_config 가 만드는 metadata/tags 는 사용자가 트레이싱을 켠 경우
(LANGSMITH_TRACING 등 환경 변수)에만 그 run 에 붙는 값이다. 트레이싱을 켜지 않으면 아무 호출도 일어나지 않는다.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from core.state import GraphState

DECISION_LOGGER = "venturesignal.decision"
logger = logging.getLogger(DECISION_LOGGER)

RUN_TAGS = ["venturesignal", "supervisor"]


def log_decision(state: GraphState, node: str, decision: str, reason: str, **extra) -> dict:
    """결정 한 건을 JSON 한 줄로 기록하고 그 기록을 돌려준다."""
    record = {
        "trace_id": state["trace_id"],
        "node": node,
        "decision": decision,
        "reason": reason,
        **extra,
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    logger.info(json.dumps(record, ensure_ascii=False))
    return record


def attach_file_logger(directory: Path, trace_id: str) -> Path:
    """결정 로그를 {directory}/{trace_id}.jsonl 에 저장한다. 파일 경로를 돌려준다."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{trace_id}.jsonl"
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return path


def recursion_limit(state: GraphState) -> int:
    """max_steps 와 품질 재작성 상한으로 계산한 LangGraph recursion_limit.

    Supervisor 한 번 진입마다 super-step 이 둘(supervisor + 하위 노드) 쓰이고, 보고서 단계는 한 번에 넷(보고서, supervisor, 평가, supervisor)이다.
    max_steps 를 넘기면 보고서로 강제 이동하므로 그 뒤의 보고서·평가와 여유분을 더한다.
    """
    # quality_check 패키지가 이 모듈을 import 하므로 순환을 피하려고 함수 안에서 가져온다
    from agents.quality_check.policy import MAX_REVISIONS

    return 2 * (state["max_steps"] + 1) + 4 * (MAX_REVISIONS + 2) + 5


def run_config(state: GraphState) -> dict:
    """graph.invoke 의 config. trace_id 를 run metadata 와 tags 에 실어 외부 트레이스와 잇는다."""
    return {
        "recursion_limit": recursion_limit(state),
        "metadata": {"trace_id": state["trace_id"]},
        "tags": [*RUN_TAGS, f"trace:{state['trace_id']}"],
    }
