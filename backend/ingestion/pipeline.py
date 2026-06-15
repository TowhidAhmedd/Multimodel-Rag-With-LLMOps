"""
backend/ingestion/pipeline.py
Orchestrates ingestion: file → chunks → embeddings → Pinecone.
"""

import asyncio
from pathlib import Path
from typing import List

from backend.ingestion.document_ingestor import ingest_document
from backend.ingestion.audio_ingestor import ingest_audio
from backend.ingestion.video_ingestor import ingest_video
from backend.chunking import Chunk
from backend.embeddings import async_embed_texts
from backend.vectorstore import insert_chunks, VectorStoreSession
from backend.logging_config import get_logger

logger = get_logger(__name__)

DOCUMENT_EXTS = {".pdf", ".docx", ".txt"}
AUDIO_EXTS    = {".mp3", ".wav", ".m4a"}
VIDEO_EXTS    = {".mp4", ".avi", ".mov"}


def get_modality(file_path: str) -> str:
    ext = Path(file_path).suffix.lower()
    if ext in DOCUMENT_EXTS:
        return "text"
    elif ext in AUDIO_EXTS:
        return "audio"
    elif ext in VIDEO_EXTS:
        return "video"
    raise ValueError(f"Unsupported file extension: {ext}")


def extract_chunks(file_path: str) -> List[Chunk]:
    ext = Path(file_path).suffix.lower()
    if ext in DOCUMENT_EXTS:
        return ingest_document(file_path)
    elif ext in AUDIO_EXTS:
        return ingest_audio(file_path)
    elif ext in VIDEO_EXTS:
        return ingest_video(file_path)
    raise ValueError(f"Unsupported file type: {ext}")


async def run_ingestion_pipeline(
    file_path: str,
    session: VectorStoreSession,
) -> dict:
    """
    Full ingestion pipeline:
      1. Extract chunks from file (CPU-bound – runs in executor)
      2. Generate embeddings
      3. Upsert into Pinecone
    """
    logger.info("Ingestion started", extra={"file": file_path})
    modality = get_modality(file_path)
    source_name = Path(file_path).name

    # Step 1: Extract
    loop = asyncio.get_running_loop()
    chunks: List[Chunk] = await loop.run_in_executor(None, extract_chunks, file_path)

    if not chunks:
        logger.warning("No content extracted", extra={"file": file_path})
        return {
            "status": "warning",
            "source_file": source_name,
            "modality": modality,
            "message": "No extractable content found in this file.",
            "chunk_count": 0,
        }

    # Step 2: Embed
    texts = [c.content for c in chunks]
    embeddings = await async_embed_texts(texts)

    # Step 3: Upsert to Pinecone
    chunk_dicts = [
        {
            "content":         chunk.content,
            "source_file":     chunk.source_file,
            "modality":        chunk.modality,
            "page_number":     chunk.page_number,
            "start_timestamp": chunk.start_timestamp,
            "end_timestamp":   chunk.end_timestamp,
            "chunk_id":        chunk.chunk_id,
            "embedding":       embedding,
            "metadata":        chunk.metadata,
        }
        for chunk, embedding in zip(chunks, embeddings)
    ]

    inserted = await insert_chunks(session, chunk_dicts)
    logger.info(
        "Ingestion complete",
        extra={"file": file_path, "chunks_inserted": inserted},
    )

    return {
        "status": "success",
        "source_file": source_name,
        "modality": modality,
        "chunk_count": inserted,
    }
