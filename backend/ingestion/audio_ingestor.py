"""
backend/ingestion/audio_ingestor.py
MP3, WAV, M4A → Faster-Whisper → transcript chunks with timestamps.
Uses lazy import so the module can be imported without faster_whisper installed.
"""

from pathlib import Path
from typing import List

from backend.chunking import Chunk, chunks_from_transcript
from backend.logging_config import get_logger

logger = get_logger(__name__)

_WHISPER_MODEL = None


def _get_whisper_model():
    """Load and cache the Whisper model (lazy import)."""
    global _WHISPER_MODEL
    if _WHISPER_MODEL is None:
        from faster_whisper import WhisperModel  # noqa: PLC0415

        logger.info("Loading Whisper model (small, CPU int8)")
        _WHISPER_MODEL = WhisperModel("small", device="cpu", compute_type="int8")
        logger.info("Whisper model loaded")
    return _WHISPER_MODEL


def transcribe_audio(file_path: str) -> List[dict]:
    """
    Transcribe an audio file using Faster-Whisper.
    Returns a list of segment dicts: {"text": str, "start": float, "end": float}
    """
    model = _get_whisper_model()
    segments, info = model.transcribe(file_path, beam_size=5)
    logger.info(
        "Transcription complete",
        extra={
            "source_file": Path(file_path).name,
            "language": info.language,
            "duration_s": round(info.duration, 1),
        },
    )
    return [
        {"text": seg.text.strip(), "start": seg.start, "end": seg.end}
        for seg in segments
        if seg.text.strip()
    ]


def ingest_audio(file_path: str) -> List[Chunk]:
    """Transcribe an audio file and return timestamp-preserving chunks."""
    source_file = Path(file_path).name
    ext = Path(file_path).suffix.lower().lstrip(".")

    segments = transcribe_audio(file_path)
    chunks = chunks_from_transcript(
        segments=segments,
        source_file=source_file,
        modality="audio",
    )
    for chunk in chunks:
        chunk.metadata = {"file_type": ext}

    logger.info(
        "Audio ingested",
        extra={"source_file": source_file, "chunk_count": len(chunks)},
    )
    return chunks
