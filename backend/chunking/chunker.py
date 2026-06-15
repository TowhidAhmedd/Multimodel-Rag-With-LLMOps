"""
backend/chunking/chunker.py
Reusable text chunker that preserves source metadata.
"""

from typing import List, Optional
from dataclasses import dataclass, field

from backend.config import get_settings

settings = get_settings()


@dataclass
class Chunk:
    content: str
    chunk_id: int
    source_file: str
    modality: str                        # text | audio | video
    page_number: Optional[int] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    metadata: dict = field(default_factory=dict)


def chunk_text(
    text: str,
    source_file: str,
    modality: str = "text",
    page_number: Optional[int] = None,
    start_timestamp: Optional[float] = None,
    end_timestamp: Optional[float] = None,
    metadata: Optional[dict] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Split `text` into overlapping chunks and return Chunk objects.

    Uses a simple character-level sliding window with word-boundary awareness.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    metadata = metadata or {}

    text = text.strip()
    if not text:
        return []

    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at a word boundary
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        chunk_content = text[start:end].strip()
        if chunk_content:
            chunks.append(
                Chunk(
                    content=chunk_content,
                    chunk_id=chunk_id,
                    source_file=source_file,
                    modality=modality,
                    page_number=page_number,
                    start_timestamp=start_timestamp,
                    end_timestamp=end_timestamp,
                    metadata=metadata,
                )
            )
            chunk_id += 1

        # Move forward by (chunk_size - overlap)
        start = end - chunk_overlap
        if start <= 0:
            start = end  # safety guard

    return chunks


def chunks_from_transcript(
    segments: List[dict],
    source_file: str,
    modality: str = "audio",
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> List[Chunk]:
    """
    Build chunks from Whisper transcript segments, preserving timestamps.

    Each segment is: {"text": str, "start": float, "end": float}
    We accumulate text until chunk_size is reached, then emit a chunk
    whose timestamp spans the accumulated segments.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks: List[Chunk] = []
    buffer_text = ""
    buffer_start: Optional[float] = None
    buffer_end: Optional[float] = None
    chunk_id = 0
    overlap_text = ""

    for seg in segments:
        seg_text = seg.get("text", "").strip()
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)

        if buffer_start is None:
            buffer_start = seg_start

        buffer_text = (buffer_text + " " + seg_text).strip()
        buffer_end = seg_end

        if len(buffer_text) >= chunk_size:
            chunks.append(
                Chunk(
                    content=buffer_text[:chunk_size].strip(),
                    chunk_id=chunk_id,
                    source_file=source_file,
                    modality=modality,
                    start_timestamp=buffer_start,
                    end_timestamp=buffer_end,
                )
            )
            chunk_id += 1
            # keep tail for overlap
            overlap_text = buffer_text[chunk_size - chunk_overlap:].strip()
            buffer_text = overlap_text
            buffer_start = buffer_end  # approximate

    # Emit leftover
    if buffer_text.strip():
        chunks.append(
            Chunk(
                content=buffer_text.strip(),
                chunk_id=chunk_id,
                source_file=source_file,
                modality=modality,
                start_timestamp=buffer_start,
                end_timestamp=buffer_end,
            )
        )

    return chunks
