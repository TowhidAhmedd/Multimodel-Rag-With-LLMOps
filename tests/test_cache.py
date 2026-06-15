"""
tests/test_cache.py
Unit tests for the in-memory TTL cache.
"""

import pytest
from backend.cache import get_cached, set_cached, clear_cache, cache_stats


@pytest.fixture(autouse=True)
def reset_cache():
    clear_cache()
    yield
    clear_cache()


class TestQueryCache:
    def test_miss_returns_none(self):
        result = get_cached("what is AI?", "llama-3.3-70b-versatile")
        assert result is None

    def test_set_and_get(self):
        response = {"answer": "AI is Artificial Intelligence.", "score": 0.9}
        set_cached("what is AI?", "llama-3.3-70b-versatile", response)
        result = get_cached("what is AI?", "llama-3.3-70b-versatile")
        assert result == response

    def test_different_model_different_key(self):
        response = {"answer": "test"}
        set_cached("query", "llama-3.3-70b-versatile", response)
        result = get_cached("query", "llama-3.1-8b-instant")
        assert result is None

    def test_different_query_different_key(self):
        response = {"answer": "test"}
        set_cached("query A", "model", response)
        result = get_cached("query B", "model")
        assert result is None

    def test_source_file_affects_key(self):
        response = {"answer": "doc-specific answer"}
        set_cached("query", "model", response, source_file="doc.pdf")
        assert get_cached("query", "model") is None
        assert get_cached("query", "model", source_file="doc.pdf") == response

    def test_clear_cache(self):
        set_cached("q", "m", {"a": 1})
        cleared = clear_cache()
        assert cleared >= 1
        assert get_cached("q", "m") is None

    def test_cache_stats(self):
        stats = cache_stats()
        assert "size" in stats
        assert "maxsize" in stats
        assert "ttl_seconds" in stats

    def test_cache_size_increments(self):
        before = cache_stats()["size"]
        set_cached("unique_q_xyz", "model", {"a": 1})
        after = cache_stats()["size"]
        assert after == before + 1
