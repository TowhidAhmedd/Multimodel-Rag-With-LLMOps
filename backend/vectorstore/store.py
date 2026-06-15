"""
backend/vectorstore/store.py
Pinecone vector store operations: upsert, similarity search, delete, list.

Public function signatures are identical to the old PostgreSQL/pgvector
implementation so all callers remain unchanged.
"""

import uuid
from typing import Any, List, Optional

from backend.vectorstore.database import VectorStoreSession
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Maximum vectors Pinecone allows per upsert call
_UPSERT_BATCH = 100


async def insert_chunks(
    session: VectorStoreSession,
    chunks: List[dict],
) -> int:
    """
    Upsert a list of chunk dicts into Pinecone.

    Each dict must contain:
      content, source_file, modality, embedding, chunk_id
      page_number (optional), start_timestamp (optional),
      end_timestamp (optional), metadata (optional)
    """
    if not chunks:
        return 0

    vectors = []
    for chunk in chunks:
        vector_id = str(uuid.uuid4())
        metadata = {
            "content":         chunk["content"],
            "source_file":     chunk["source_file"],
            "modality":        chunk["modality"],
            "chunk_id":        chunk.get("chunk_id", 0),
            "page_number":     chunk.get("page_number"),
            "start_timestamp": chunk.get("start_timestamp"),
            "end_timestamp":   chunk.get("end_timestamp"),
        }
        # Merge any extra metadata, filtering None values (Pinecone rejects them)
        extra = chunk.get("metadata") or {}
        metadata.update({k: v for k, v in extra.items() if v is not None})
        # Also strip None from core fields
        metadata = {k: v for k, v in metadata.items() if v is not None}

        vectors.append({
            "id":       vector_id,
            "values":   chunk["embedding"],
            "metadata": metadata,
        })

    # Pinecone recommends batches of ≤ 100 vectors
    inserted = 0
    for i in range(0, len(vectors), _UPSERT_BATCH):
        batch = vectors[i : i + _UPSERT_BATCH]
        session.index.upsert(vectors=batch)
        inserted += len(batch)

    logger.info(
        "Chunks upserted to Pinecone",
        extra={"count": inserted, "source_file": chunks[0]["source_file"]},
    )
    return inserted


async def similarity_search(
    session: VectorStoreSession,
    query_embedding: List[float],
    top_k: int = 5,
    source_file: Optional[str] = None,
) -> List[dict]:
    """
    Cosine similarity search in Pinecone.
    Returns up to top_k chunks ordered by similarity (highest first).
    """
    query_kwargs: dict = {
        "vector":          query_embedding,
        "top_k":           top_k,
        "include_metadata": True,
        "include_values":  False,
    }

    if source_file:
        query_kwargs["filter"] = {"source_file": {"$eq": source_file}}

    response = session.index.query(**query_kwargs)
    matches = response.get("matches", [])

    chunks = []
    for match in matches:
        meta = match.get("metadata", {})
        chunks.append({
            "id":              match["id"],
            "content":         meta.get("content", ""),
            "source_file":     meta.get("source_file", ""),
            "modality":        meta.get("modality", "text"),
            "page_number":     meta.get("page_number"),
            "start_timestamp": meta.get("start_timestamp"),
            "end_timestamp":   meta.get("end_timestamp"),
            "chunk_id":        meta.get("chunk_id", 0),
            "metadata":        {},
            "score":           round(float(match.get("score", 0.0)), 4),
        })

    logger.info(
        "Similarity search complete",
        extra={"results": len(chunks), "top_k": top_k},
    )
    return chunks


async def delete_by_source(
    session: VectorStoreSession,
    source_file: str,
) -> int:
    """
    Delete all vectors whose metadata.source_file matches source_file.
    Returns the number of vectors deleted (Pinecone does not return a count,
    so we fetch IDs first then delete).
    """
    # Fetch matching IDs with a dummy query (fetch all via pagination)
    index_stats = session.index.describe_index_stats()
    total = index_stats.get("total_vector_count", 0)
    if total == 0:
        return 0

    # Use list_paginated if supported, else query with a zero vector
    try:
        # Pinecone v3+ supports delete by metadata filter directly
        session.index.delete(filter={"source_file": {"$eq": source_file}})
        logger.info("Chunks deleted by source (filter)", extra={"source_file": source_file})
        # Count is unknown with filter-delete; return sentinel
        return -1
    except Exception:
        pass

    # Fallback: query → collect IDs → delete IDs
    dummy = [0.0] * 384
    resp = session.index.query(
        vector=dummy,
        top_k=10000,
        filter={"source_file": {"$eq": source_file}},
        include_metadata=False,
        include_values=False,
    )
    ids = [m["id"] for m in resp.get("matches", [])]
    if ids:
        session.index.delete(ids=ids)
    logger.info(
        "Chunks deleted by source (id list)",
        extra={"source_file": source_file, "count": len(ids)},
    )
    return len(ids)


async def list_sources(session: VectorStoreSession) -> List[str]:
    """
    Return a deduplicated list of source_file values in the index.
    Pinecone has no GROUP BY, so we query with a dummy vector and
    extract unique source_file values from metadata.
    """
    dummy = [0.0] * 384
    resp = session.index.query(
        vector=dummy,
        top_k=10000,
        include_metadata=True,
        include_values=False,
    )
    sources = sorted({
        m["metadata"].get("source_file", "")
        for m in resp.get("matches", [])
        if m.get("metadata", {}).get("source_file")
    })
    return sources
