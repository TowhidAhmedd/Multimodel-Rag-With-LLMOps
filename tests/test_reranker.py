"""
tests/test_reranker.py
Unit tests for the re-ranker using mocks (no sentence_transformers needed).
"""

import pytest
from unittest.mock import patch, MagicMock

from backend.reranking.reranker import rerank


def _make_chunks(n: int = 5) -> list:
    return [
        {
            "content": f"This is chunk {i} with relevant content about the topic.",
            "source_file": "doc.pdf",
            "modality": "text",
        }
        for i in range(n)
    ]


class TestRerank:
    @patch("backend.reranking.reranker._load_reranker")
    def test_returns_list(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.3, 0.7, 0.1]
        mock_load.return_value = mock_model

        result = rerank("What is the topic?", _make_chunks(5), top_k=3)
        assert isinstance(result, list)

    @patch("backend.reranking.reranker._load_reranker")
    def test_top_k_respected(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.9, 0.5, 0.3, 0.7, 0.1]
        mock_load.return_value = mock_model

        result = rerank("Query", _make_chunks(5), top_k=3)
        assert len(result) == 3

    @patch("backend.reranking.reranker._load_reranker")
    def test_sorted_descending(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.9, 0.5]
        mock_load.return_value = mock_model

        result = rerank("Query", _make_chunks(3), top_k=3)
        scores = [r["rerank_score"] for r in result]
        assert scores == sorted(scores, reverse=True)

    @patch("backend.reranking.reranker._load_reranker")
    def test_rerank_score_added_to_chunks(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.8]
        mock_load.return_value = mock_model

        result = rerank("Query", _make_chunks(1), top_k=1)
        assert "rerank_score" in result[0]
        assert isinstance(result[0]["rerank_score"], float)

    @patch("backend.reranking.reranker._load_reranker")
    def test_score_rounded_to_4dp(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.123456789]
        mock_load.return_value = mock_model

        result = rerank("Query", _make_chunks(1), top_k=1)
        assert result[0]["rerank_score"] == 0.1235

    def test_empty_chunks_returns_empty(self):
        result = rerank("Query", [], top_k=5)
        assert result == []

    @patch("backend.reranking.reranker._load_reranker")
    def test_top_k_larger_than_chunks(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.5, 0.3]
        mock_load.return_value = mock_model

        result = rerank("Query", _make_chunks(2), top_k=10)
        assert len(result) == 2  # capped at available chunks

    @patch("backend.reranking.reranker._load_reranker")
    def test_model_called_with_pairs(self, mock_load):
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.7, 0.4]
        mock_load.return_value = mock_model

        chunks = _make_chunks(2)
        rerank("my query", chunks, top_k=2)

        call_args = mock_model.predict.call_args[0][0]
        assert len(call_args) == 2
        for pair in call_args:
            assert pair[0] == "my query"
