"""
backend/ingestion/video_ingestor.py
MP4, AVI, MOV → extract audio (ffmpeg) → Faster-Whisper → transcript chunks.
Transcript-only: no OCR, no frame analysis, no computer vision.
Uses lazy imports for ffmpeg-python and faster_whisper.
"""

import os
import tempfile
from pathlib import Path
from typing import List

from backend.ingestion.audio_ingestor import transcribe_audio
from backend.chunking import Chunk, chunks_from_transcript
from backend.logging_config import get_logger

logger = get_logger(__name__)


def extract_audio_from_video(video_path: str, output_dir: str) -> str:
    """
    Extract the audio track from a video file to a 16 kHz mono WAV.
    Returns the path to the temporary WAV file.
    """
    import ffmpeg  # noqa: PLC0415  (lazy import)

    audio_path = os.path.join(output_dir, "extracted_audio.wav")
    (
        ffmpeg
        .input(video_path)
        .output(audio_path, ac=1, ar=16000, format="wav")
        .overwrite_output()
        .run(quiet=True)
    )
    logger.info(
        "Audio extracted from video",
        extra={"video": Path(video_path).name, "audio_path": audio_path},
    )
    return audio_path


def ingest_video(file_path: str) -> List[Chunk]:
    """Extract audio from video, transcribe, and return timestamp chunks."""
    source_file = Path(file_path).name
    ext = Path(file_path).suffix.lower().lstrip(".")

    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = extract_audio_from_video(file_path, tmpdir)
        segments = transcribe_audio(audio_path)

    chunks = chunks_from_transcript(
        segments=segments,
        source_file=source_file,
        modality="video",
    )
    for chunk in chunks:
        chunk.metadata = {"file_type": ext}

    logger.info(
        "Video ingested",
        extra={"source_file": source_file, "chunk_count": len(chunks)},
    )
    return chunks
