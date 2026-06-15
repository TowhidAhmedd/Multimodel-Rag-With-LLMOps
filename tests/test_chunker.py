"""
tests/test_chunker.py
Unit tests for the text chunker.
"""

import pytest
from backend.chunking import chunk_text, chunks_from_transcript, Chunk


class TestChunkText:
    def test_basic_chunking(self):
        text = "Hello world. " * 100
        chunks = chunk_text(text, source_file="test.txt", chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        for c in chunks:
            assert isinstance(c, Chunk)
            assert len(c.content) > 0

    def test_single_short_text(self):
        text = "Short text."
        chunks = chunk_text(text, source_file="short.txt", chunk_size=800, chunk_overlap=150)
        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_page_number_preserved(self):
        text = "Sample text. " * 20
        chunks = chunk_text(text, source_file="doc.pdf", page_number=3, modality="text", chunk_size=100, chunk_overlap=20)
        for c in chunks:
            assert c.page_number == 3

    def test_source_file_preserved(self):
        text = "Sample. " * 50
        chunks = chunk_text(text, source_file="my_file.txt", chunk_size=100, chunk_overlap=20)
        for c in chunks:
            assert c.source_file == "my_file.txt"

    def test_modality_preserved(self):
        text = "Some text. " * 30
        chunks = chunk_text(text, source_file="x.txt", modality="audio", chunk_size=100, chunk_overlap=20)
        for c in chunks:
            assert c.modality == "audio"

    def test_empty_text(self):
        chunks = chunk_text("", source_file="empty.txt")
        assert chunks == []

    def test_whitespace_only(self):
        chunks = chunk_text("   \n\t  ", source_file="ws.txt")
        assert chunks == []

    def test_chunk_ids_sequential(self):
        text = "Word " * 500
        chunks = chunk_text(text, source_file="big.txt", chunk_size=200, chunk_overlap=50)
        for i, c in enumerate(chunks):
            assert c.chunk_id == i

    def test_metadata_passed(self):
        text = "Content. " * 20
        meta = {"file_type": "pdf", "total_pages": 5}
        chunks = chunk_text(text, source_file="f.pdf", metadata=meta, chunk_size=100, chunk_overlap=20)
        for c in chunks:
            assert c.metadata == meta


class TestChunksFromTranscript:
    def _make_segments(self, n: int) -> list:
        return [
            {"text": f"Segment {i} with some content here", "start": float(i * 5), "end": float(i * 5 + 4)}
            for i in range(n)
        ]

    def test_basic_transcript_chunking(self):
        segments = self._make_segments(20)
        chunks = chunks_from_transcript(segments, source_file="audio.mp3", chunk_size=200, chunk_overlap=50)
        assert len(chunks) >= 1

    def test_timestamps_present(self):
        segments = self._make_segments(30)
        chunks = chunks_from_transcript(segments, source_file="audio.mp3", chunk_size=200, chunk_overlap=50)
        for c in chunks:
            assert c.start_timestamp is not None
            assert c.end_timestamp is not None

    def test_modality_preserved(self):
        segments = self._make_segments(10)
        chunks = chunks_from_transcript(segments, source_file="video.mp4", modality="video", chunk_size=200, chunk_overlap=50)
        for c in chunks:
            assert c.modality == "video"

    def test_empty_segments(self):
        chunks = chunks_from_transcript([], source_file="empty.mp3")
        assert chunks == []
