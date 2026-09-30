import argparse
from datetime import datetime

from core.config import OUTPUT_DIR
from core.state import initial_state
from graph.builder import build_graph


def main():
    parser = argparse.ArgumentParser(description="VentureSignal — 반도체 AI 스타트업 투자 검토")
    parser.add_argument("query", help='예) "국내 저전력 엣지 AI 추론 칩 스타트업"')
    args = parser.parse_args()

    graph = build_graph()
    result = graph.invoke(initial_state(args.query), config={"recursion_limit": 50})

    OUTPUT_DIR.mkdir(exist_ok=True)
    name = result.get("selected_candidate") or "HOLD"
    path = OUTPUT_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.md"
    path.write_text(result["report"], encoding="utf-8")
    print(result["report"])
    print(f"\n→ 저장: {path}")


if __name__ == "__main__":
    main()
