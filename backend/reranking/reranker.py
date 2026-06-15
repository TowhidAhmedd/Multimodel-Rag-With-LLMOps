"""
backend/reranking/reranker.py
Cross-Encoder re-ranking using sentence-transformers MiniLM model.
Uses lazy import so the module can be imported without sentence_transformers
being loaded (allows mocking in tests and faster cold starts).
"""

import asyncio
from functools import lru_cache
from typing import List

from backend.logging_config import get_logger

logger = get_logger(__name__)

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _load_reranker():
    """Load and cache the CrossEncoder model (lazy import)."""
    from sentence_transformers import CrossEncoder  # noqa: PLC0415

    logger.info("Loading re-ranker model", extra={"model": RERANKER_MODEL})
    model = CrossEncoder(RERANKER_MODEL)
    logger.info("Re-ranker loaded")
    return model


def rerank(query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
    """
    Re-rank retrieved chunks using a cross-encoder.
    Returns up to top_k chunks sorted by cross-encoder score (descending).
    Mutates each chunk dict to add a 'rerank_score' key.
    """
    if not chunks:
        return []

    model = _load_reranker()
    pairs = [(query, chunk["content"]) for chunk in chunks]
    scores = model.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = round(float(score), 4)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return ranked[:top_k]


async def async_rerank(query: str, chunks: List[dict], top_k: int = 5) -> List[dict]:
    """Non-blocking wrapper — runs rerank() in the thread-pool executor."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, rerank, query, chunks, top_k)
