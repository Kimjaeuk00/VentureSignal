"""관측성: 결정 로그는 trace_id 와 함께 외부(로거·파일)로 나가고 State 에는 쌓이지 않는다 (LangSmith 호출 없음)."""

import json
import logging

from agents.quality_check.policy import MAX_REVISIONS
from agents.supervisor import audit
from agents.supervisor.node import run
from core.state import initial_state


def _state():
    state = initial_state("q")
    state["candidates"] = [{"company_id": "C0", "company_name": "Co0", "description": "", "domain": "", "retrieval_score": 0.0}]
    return state


def test_log_decision_writes_one_json_line_with_trace_id(caplog):
    state = _state()
    with caplog.at_level("INFO", logger=audit.DECISION_LOGGER):
        record = audit.log_decision(state, "supervisor", "tech_summary", "사유", step=1)

    line = json.loads(caplog.records[-1].getMessage())
    assert line == record
    assert line["trace_id"] == state["trace_id"]
    assert {"node", "decision", "reason", "step", "ts"} <= set(line)


def test_file_logger_collects_decisions_by_trace_id(tmp_path):
    state = _state()
    path = audit.attach_file_logger(tmp_path, state["trace_id"])
    try:
        run(state)
        run({**state, "step_count": 1})
    finally:
        for handler in list(audit.logger.handlers):
            audit.logger.removeHandler(handler)
            handler.close()

    assert path == tmp_path / f"{state['trace_id']}.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(lines) == 2 and all(line["trace_id"] == state["trace_id"] for line in lines)


def test_state_keeps_only_last_decision_not_a_log():
    state = _state()
    for _ in range(3):
        update = run(state)
        state = {**state, **update}
    assert isinstance(state["last_decision"], dict) and set(state["last_decision"]) == {"next", "reason"}
    assert state["step_count"] == 3  # 로그 본문이 State 에 누적되지 않고 카운터만 늘어난다


def test_run_config_carries_trace_id_as_metadata_and_tag():
    state = _state()
    config = audit.run_config(state)
    assert config["metadata"] == {"trace_id": state["trace_id"]}
    assert f"trace:{state['trace_id']}" in config["tags"]


def test_recursion_limit_covers_worst_case_steps():
    state = _state()
    # Supervisor 진입마다 2 super-step, max_steps 초과 후 강제 보고서, 품질 재작성 루프까지 들어가야 한다
    assert audit.recursion_limit(state) >= 2 * (state["max_steps"] + 1) + 3 * (MAX_REVISIONS + 1) + 1
    assert audit.recursion_limit(state) > 50  # main.py 의 기존 고정값보다 크다
