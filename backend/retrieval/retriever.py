"""
backend/retrieval/retriever.py
Retrieval pipeline: embed query → Pinecone search → re-rank.
"""

from typing import List, Optional

from backend.embeddings import async_embed_query
from backend.vectorstore import similarity_search, VectorStoreSession
from backend.reranking import async_rerank
from backend.config import get_settings
from backend.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


async def retrieve(
    query: str,
    session: VectorStoreSession,
    top_k: Optional[int] = None,
    source_file: Optional[str] = None,
) -> List[dict]:
    """
    Full retrieval pipeline:
      1. Embed the query
      2. Cosine similarity search in Pinecone
      3. Re-rank with cross-encoder
      4. Return top_k chunks
    """
    top_k = top_k or settings.top_k

    # Step 1: Embed
    query_embedding = await async_embed_query(query)

    # Step 2: Vector search (fetch 3× top_k candidates for re-ranking)
    candidates = await similarity_search(
        session=session,
        query_embedding=query_embedding,
        top_k=top_k * 3,
        source_file=source_file,
    )

    if not candidates:
        logger.info("No candidates found", extra={"query": query[:100]})
        return []

    # Step 3: Re-rank
    reranked = await async_rerank(query, candidates, top_k=top_k)

    logger.info(
        "Retrieval complete",
        extra={
            "query": query[:100],
            "candidates": len(candidates),
            "returned": len(reranked),
        },
    )
    return reranked
