"""
투자 판단 (LLM)

입력: state["evaluations"][current_candidate] 의 6개 AnalysisResult
출력: evaluation_update(cid, scorecard=..., decision=...)
      INVEST 인 경우 {"selected_candidate": cid} 도 함께 반환

- LLM 은 6개 항목 점수(0~5)와 근거만 산출하고,
  가중합·판정은 scoring.build_scorecard / scoring.decide 로 코드에서 계산한다.
"""

from core.state import GraphState, evaluation_update

from .scoring import build_scorecard, decide


def run(state: GraphState) -> dict:
    cid = state["current_candidate"]
    # TODO: 구현 — LLM 채점. 스텁은 전 항목 0점 → HOLD
    scores = {k: 0.0 for k in ("technology", "competition", "market", "team", "traction", "deal_terms")}
    scorecard = build_scorecard(scores)
    decision = decide(scorecard)

    update = evaluation_update(cid, scorecard=scorecard, decision=decision)
    if decision == "INVEST":
        update["selected_candidate"] = cid
    return update
