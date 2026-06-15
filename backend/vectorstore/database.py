"""
backend/vectorstore/database.py
Pinecone vector store — connection, index management, and session.

Replaces the PostgreSQL + pgvector implementation.
The public API is identical so all callers (routes, retriever,
pipeline, rag_engine) remain unchanged.

All chunk metadata (content, source_file, modality, page_number,
start/end timestamp, chunk_id) is stored inside Pinecone's metadata
fields alongside the vector — no separate SQL database is required.
"""

import uuid
from typing import AsyncGenerator, Any

from backend.config import get_settings
from backend.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 output dimension


# ── Pinecone client (lazy, cached) ────────────────────────────────────────────

_pinecone_client = None
_pinecone_index  = None


def _get_client():
    global _pinecone_client
    if _pinecone_client is None:
        from pinecone import Pinecone  # noqa: PLC0415
        if not settings.pinecone_api_key or settings.pinecone_api_key in ("", "changeme"):
            raise RuntimeError(
                "PINECONE_API_KEY is missing. "
                "Add it to your .env file — get a free key at https://app.pinecone.io"
            )
        logger.info("Initialising Pinecone client")
        _pinecone_client = Pinecone(api_key=settings.pinecone_api_key)
    return _pinecone_client


def get_index():
    """Return (and cache) a handle to the Pinecone index."""
    global _pinecone_index
    if _pinecone_index is None:
        pc = _get_client()
        index_name = settings.pinecone_index_name
        if not pc.has_index(index_name):
            from pinecone import ServerlessSpec  # noqa: PLC0415
            logger.info("Creating Pinecone index", extra={"index": index_name})
            pc.create_index(
                name=index_name,
                dimension=EMBEDDING_DIM,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region,
                ),
            )
            logger.info("Pinecone index created", extra={"index": index_name})
        _pinecone_index = pc.Index(index_name)
    return _pinecone_index


# ── Session abstraction ───────────────────────────────────────────────────────

class VectorStoreSession:
    """
    Drop-in replacement for SQLAlchemy's AsyncSession.

    Pinecone has no transactional session — every call is an immediate
    API call.  This class wraps the Pinecone index so that routes,
    retriever, pipeline, and rag_engine can keep their existing
    `session` parameter without any signature changes.
    """

    def __init__(self):
        self.index = get_index()

    async def health_ping(self) -> str:
        """Used by GET /health to confirm Pinecone reachability."""
        try:
            self.index.describe_index_stats()
            return "connected"
        except Exception as exc:  # noqa: BLE001
            return f"error: {exc}"


async def get_db() -> AsyncGenerator[VectorStoreSession, None]:
    """FastAPI dependency — yields one VectorStoreSession per request."""
    session = VectorStoreSession()
    try:
        yield session
    finally:
        pass  # No connection to close; Pinecone client is module-level


# ── Init ──────────────────────────────────────────────────────────────────────

async def init_db() -> None:
    """
    Called once at startup (main.py lifespan).
    Ensures the Pinecone index exists, creating it on first run.
    """
    get_index()          # creates the index if absent
    logger.info(
        "Pinecone index ready",
        extra={"index": settings.pinecone_index_name, "action": "init_db"},
    )
