from functools import lru_cache

from qdrant_client import models

from core.config import EMBEDDING_MODEL


@lru_cache
def _get_model():
    # 무거운 import 는 실제 사용 시점에만 (테스트·스텁 실행 시 torch 로딩 방지)
    from FlagEmbedding import BGEM3FlagModel

    return BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=False)


def embed(texts: list[str], batch_size: int = 8) -> list[tuple[list[float], models.SparseVector]]:
    """bge-m3 로 (dense 벡터, sparse 벡터) 쌍을 만든다."""
    out = _get_model().encode(
        texts,
        batch_size=batch_size,
        max_length=8192,
        return_dense=True,
        return_sparse=True,
        return_colbert_vecs=False,
    )
    results = []
    for dense, lexical in zip(out["dense_vecs"], out["lexical_weights"]):
        sparse = models.SparseVector(
            indices=[int(k) for k in lexical],
            values=[float(v) for v in lexical.values()],
        )
        results.append((dense.tolist(), sparse))
    return results
