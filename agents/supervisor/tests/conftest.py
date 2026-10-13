import pytest

from agents.supervisor import llm_policy


@pytest.fixture(autouse=True)
def rule_based_by_default(monkeypatch):
    """기본은 규칙 기반이다 (테스트에서 실제 LLM 을 부르지 않는다). LLM 동작을 보는 테스트는 llm_on 을 쓴다."""
    monkeypatch.setattr(llm_policy, "ENABLED", False)


class FakeLLM:
    """with_structured_output(...).invoke(messages) 를 흉내 낸다. responses 는 순서대로 쓰이는 응답(예외도 가능)이거나 messages → 응답 함수."""

    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.calls.append(messages)
        if callable(self.responses):
            return self.responses(messages)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def llm_on(monkeypatch):
    """LLM 판단을 켜고 가짜 LLM 을 꽂는다. 사용: fake = llm_on([...응답...])"""

    def install(responses):
        fake = FakeLLM(responses)
        monkeypatch.setattr(llm_policy, "ENABLED", True)
        monkeypatch.setattr(llm_policy, "get_llm", lambda: fake)
        return fake

    return install
