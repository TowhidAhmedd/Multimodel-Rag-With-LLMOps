"""
backend/api/routes.py
FastAPI route handlers – upload, query, health, metrics, cache management.
"""

import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.api.schemas import (
    HealthResponse,
    MetricsResponse,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from backend.cache import cache_stats, clear_cache
from backend.config import get_settings
from backend.core import answer_query
from backend.ingestion import AUDIO_EXTS, DOCUMENT_EXTS, VIDEO_EXTS, run_ingestion_pipeline
from backend.llm import SUPPORTED_MODELS
from backend.logging_config import get_logger
from backend.vectorstore import get_db, VectorStoreSession

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()

ALLOWED_EXTENSIONS: set[str] = DOCUMENT_EXTS | AUDIO_EXTS | VIDEO_EXTS
MAX_SIZE_BYTES: int = settings.max_upload_size_mb * 1024 * 1024


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _sanitize_filename(filename: str) -> str:
    name = Path(filename).name
    safe = "".join(c for c in name if c.isalnum() or c in (".", "-", "_"))
    return safe or "uploaded_file"


def _validate_extension(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '{ext}' is not supported. "
                f"Allowed types: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )
    return ext


# ─── POST /upload ─────────────────────────────────────────────────────────────

@router.post("/upload", response_model=UploadResponse, summary="Upload and ingest a file")
async def upload_file(
    file: UploadFile = File(...),
    session: VectorStoreSession = Depends(get_db),
) -> UploadResponse:
    """
    Upload a document (PDF/DOCX/TXT), audio (MP3/WAV/M4A), or
    video (MP4/AVI/MOV) file. The file is chunked, embedded, and
    stored in Pinecone immediately.
    """
    original_name = file.filename or "unknown"
    _validate_extension(original_name)
    safe_name = _sanitize_filename(original_name)

    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    unique_name = f"{uuid.uuid4().hex}_{safe_name}"
    save_path = upload_dir / unique_name

    total_bytes = 0
    try:
        with open(save_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_bytes += len(chunk)
                if total_bytes > MAX_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=(
                            f"File exceeds the maximum allowed size "
                            f"of {settings.max_upload_size_mb} MB."
                        ),
                    )
                f.write(chunk)
    except HTTPException:
        if save_path.exists():
            save_path.unlink()
        raise
    except Exception as exc:
        if save_path.exists():
            save_path.unlink()
        logger.error("File write failed", extra={"error": str(exc), "file": safe_name})
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}")

    logger.info(
        "File saved",
        extra={"filename": safe_name, "size_bytes": total_bytes, "path": str(save_path)},
    )

    try:
        result = await run_ingestion_pipeline(str(save_path), session)
    except Exception as exc:
        logger.error("Ingestion failed", extra={"error": str(exc), "file": safe_name})
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return UploadResponse(
        status=result["status"],
        source_file=safe_name,
        chunk_count=result.get("chunk_count", 0),
        modality=result.get("modality", "unknown"),
        message=result.get("message"),
    )


# ─── POST /query ─────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, summary="Ask a question")
async def query_documents(
    request: QueryRequest,
    session: VectorStoreSession = Depends(get_db),
) -> QueryResponse:
    """
    Ask a question grounded in uploaded content.
    """
    if request.model not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Model '{request.model}' is not supported. "
                f"Choose from: {sorted(SUPPORTED_MODELS)}"
            ),
        )

    try:
        result = await answer_query(
            query=request.query,
            session=session,
            model=request.model,
            source_file=request.source_file,
            top_k=request.top_k,
        )
    except Exception as exc:
        logger.error("Query failed", extra={"error": str(exc), "query": request.query[:100]})
        raise HTTPException(status_code=500, detail=f"Query processing failed: {exc}")

    return QueryResponse(**result)


# ─── GET /health ──────────────────────────────────────────────────────────────

@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health_check(session: VectorStoreSession = Depends(get_db)) -> HealthResponse:
    """Return current system health including Pinecone connectivity."""
    db_status = await session.health_ping()
    return HealthResponse(
        status="ok" if db_status == "connected" else "degraded",
        database=db_status,
        embedding_model=settings.embedding_model,
        default_model=settings.default_llm_model,
    )


# ─── GET /metrics ─────────────────────────────────────────────────────────────

@router.get("/metrics", response_model=MetricsResponse, summary="System metrics")
async def get_metrics() -> MetricsResponse:
    stats = cache_stats()
    return MetricsResponse(
        cache_size=stats["size"],
        cache_maxsize=stats["maxsize"],
        cache_ttl_seconds=stats["ttl_seconds"],
        supported_models=sorted(SUPPORTED_MODELS),
        embedding_model=settings.embedding_model,
    )


# ─── DELETE /cache ────────────────────────────────────────────────────────────

@router.delete("/cache", summary="Clear query cache")
async def clear_query_cache() -> dict:
    cleared = clear_cache()
    return {"cleared": cleared, "status": "ok"}
