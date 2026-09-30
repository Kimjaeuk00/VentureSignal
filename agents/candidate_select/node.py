"""
후보 선택 (일반 함수)

입력: state["candidates"], state["candidate_index"]
출력: {"current_candidate": company_id, "candidate_index": 다음 인덱스}

candidate_index 규칙: "지금까지 꺼낸 후보 수".
이 노드가 candidates[candidate_index] 를 꺼내고 1 증가시킨다.
따라서 투자 판단 이후 candidate_index < len(candidates) 이면 남은 후보가 있다.
"""

from core.state import GraphState


def run(state: GraphState) -> dict:
    idx = state["candidate_index"]
    candidate = state["candidates"][idx]
    return {
        "current_candidate": candidate["company_id"],
        "candidate_index": idx + 1,
    }
