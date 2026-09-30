"""
보고서 생성 (LLM)

입력: state 전체 (selected_candidate 가 None 이면 전 후보 보류)
출력: {"report": str}  — template.md(설계 문서 E절) 양식의 마크다운

- 출처 번호는 evidence 의 source_id / URL 에서 만들고 REFERENCE 에 서지정보를 적는다.
  (rag.sources.format_source 로 S번호 → 서지정보 변환)
- 전 후보 HOLD 인 경우 보류 사유와 재검토 조건을 포함한다.
"""

from pathlib import Path

from core.state import GraphState

TEMPLATE = (Path(__file__).parent / "template.md").read_text(encoding="utf-8")


def run(state: GraphState) -> dict:
    # TODO: 구현 — 스텁은 판정 요약만 출력
    lines = [f"# 투자 검토 보고서 (STUB)", "", f"질의: {state['query']}", ""]
    for cid, ev in state["evaluations"].items():
        sc = ev.get("scorecard", {})
        lines.append(f"- {cid}: {ev.get('decision')} (총점 {sc.get('total_score')})")
    lines.append("")
    lines.append(f"선정 기업: {state.get('selected_candidate') or '없음 (전 후보 보류)'}")
    return {"report": "\n".join(lines)}
