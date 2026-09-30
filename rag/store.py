from functools import lru_cache

from qdrant_client import QdrantClient

from core.config import QDRANT_PATH


@lru_cache
def get_client() -> QdrantClient:
    # 로컬 파일 모드는 프로세스당 하나의 클라이언트만 허용하므로 공유한다.
    QDRANT_PATH.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))
