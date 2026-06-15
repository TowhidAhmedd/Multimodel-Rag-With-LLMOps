"""
backend/embeddings/embedder.py
Singleton sentence-transformer embedder using BAAI/bge-small-en-v1.5.
Uses lazy import pattern so the module can be imported without torch installed
(enables faster test collection and clean mocking).
"""

import asyncio
from functools import lru_cache
from typing import List

from backend.config import get_settings
from backend.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


@lru_cache(maxsize=1)
def _load_model():
    """Load the embedding model once and cache it (lazy import)."""
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    logger.info("Loading embedding model", extra={"model": settings.embedding_model})
    model = SentenceTransformer(settings.embedding_model)
    logger.info("Embedding model loaded", extra={"model": settings.embedding_model})
    return model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Synchronously embed a list of strings.
    Embeddings are L2-normalised (unit vectors) for cosine similarity.
    """
    model = _load_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    return [emb.tolist() for emb in embeddings]


async def async_embed_texts(texts: List[str]) -> List[List[float]]:
    """Run embed_texts() in the thread-pool so the event loop is not blocked."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, embed_texts, texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string (synchronous)."""
    return embed_texts([query])[0]


async def async_embed_query(query: str) -> List[float]:
    """Embed a single query string (async)."""
    results = await async_embed_texts([query])
    return results[0]
