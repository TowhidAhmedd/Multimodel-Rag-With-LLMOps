"""
tests/test_evaluator.py
Unit tests for the evaluation module.
"""

import pytest
from backend.evaluation import compute_metrics, extract_confidence, format_source_references


class TestExtractConfidence:
    def test_high_confidence(self):
        text = "The answer is X. [Confidence: 0.92]"
        assert extract_confidence(text) == pytest.approx(0.92)

    def test_low_confidence(self):
        text = "Based on context. [Confidence: 0.3]"
        assert extract_confidence(text) == pytest.approx(0.3)

    def test_no_confidence_returns_default(self):
        text = "Some answer without a confidence score."
        assert extract_confidence(text) == 0.5

    def test_clamped_above_1(self):
        text = "[Confidence: 1.5]"
        assert extract_confidence(text) == 1.0

    def test_clamped_below_0(self):
        text = "[Confidence: -0.2]"
        assert extract_confidence(text) == 0.0

    def test_case_insensitive(self):
        text = "[CONFIDENCE: 0.75]"
        assert extract_confidence(text) == pytest.approx(0.75)


class TestComputeMetrics:
    def _make_chunks(self, n: int = 3) -> list:
        return [
            {"content": f"Chunk {i} content", "score": 0.8, "rerank_score": 0.7}
            for i in range(n)
        ]

    def test_returns_dict(self):
        metrics = compute_metrics("q", self._make_chunks(), "answer [Confidence: 0.8]", 150.0, "model-x")
        assert isinstance(metrics, dict)

    def test_retrieval_count(self):
        chunks = self._make_chunks(4)
        metrics = compute_metrics("q", chunks, "ans", 100.0, "m")
        assert metrics["retrieval_count"] == 4

    def test_confidence_extracted(self):
        metrics = compute_metrics("q", self._make_chunks(), "ans [Confidence: 0.65]", 100.0, "m")
        assert metrics["confidence_score"] == pytest.approx(0.65)

    def test_latency_recorded(self):
        metrics = compute_metrics("q", self._make_chunks(), "ans", 234.5, "m")
        assert metrics["latency_ms"] == pytest.approx(234.5)

    def test_has_sufficient_evidence_true(self):
        metrics = compute_metrics("q", self._make_chunks(), "Here is the answer.", 100.0, "m")
        assert metrics["has_sufficient_evidence"] is True

    def test_has_sufficient_evidence_false(self):
        metrics = compute_metrics("q", self._make_chunks(), "I could not find sufficient evidence in the uploaded content.", 100.0, "m")
        assert metrics["has_sufficient_evidence"] is False

    def test_empty_chunks(self):
        metrics = compute_metrics("q", [], "no answer", 50.0, "m")
        assert metrics["retrieval_count"] == 0
        assert metrics["avg_retrieval_score"] == 0.0


class TestFormatSourceReferences:
    def test_text_source(self):
        chunks = [{"source_file": "doc.pdf", "modality": "text", "page_number": 3}]
        refs = format_source_references(chunks)
        assert len(refs) == 1
        assert refs[0]["page"] == 3

    def test_audio_source(self):
        chunks = [{"source_file": "audio.mp3", "modality": "audio", "start_timestamp": 10.0, "end_timestamp": 30.0}]
        refs = format_source_references(chunks)
        assert refs[0]["start_timestamp"] == 10.0

    def test_deduplication(self):
        chunks = [
            {"source_file": "doc.pdf", "modality": "text", "page_number": 2},
            {"source_file": "doc.pdf", "modality": "text", "page_number": 2},
        ]
        refs = format_source_references(chunks)
        assert len(refs) == 1

    def test_empty_chunks(self):
        refs = format_source_references([])
        assert refs == []
