from functools import lru_cache

from langchain.chat_models import init_chat_model

from core.config import LLM_MODEL, LLM_PROVIDER


@lru_cache
def get_llm():
    """모든 에이전트가 공유하는 Chat 모델. 제공자/모델은 .env 의 LLM_PROVIDER, LLM_MODEL 로 전환."""
    return init_chat_model(LLM_MODEL, model_provider=LLM_PROVIDER)
