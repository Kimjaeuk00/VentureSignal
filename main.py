import argparse
from datetime import datetime

from agents.supervisor.audit import attach_file_logger, run_config
from core.config import OUTPUT_DIR
from core.state import initial_state
from graph.builder import build_graph


def main():
    parser = argparse.ArgumentParser(description="VentureSignal — 반도체 AI 스타트업 투자 검토")
    parser.add_argument("query", help='예) "국내 저전력 엣지 AI 추론 칩 스타트업"')
    args = parser.parse_args()

    graph = build_graph()
    state = initial_state(args.query)
    # 결정 로그는 State 가 아니라 파일에 남기고 trace_id 로 잇는다. LangSmith 는 트레이싱을 켠 경우에만 같은 trace_id 를 metadata 로 받는다.
    log_path = attach_file_logger(OUTPUT_DIR / "logs", state["trace_id"])
    result = graph.invoke(state, config=run_config(state))

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = result.get("selected_candidate") or "HOLD"
    path = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.md"
    path.write_text(result["report"], encoding="utf-8")
    print(result["report"])
    print(f"\n→ 저장: {path}")
    print(f"→ 결정 로그: {log_path} (trace_id {state['trace_id']})")


if __name__ == "__main__":
    main()
