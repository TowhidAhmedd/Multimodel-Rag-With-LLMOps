"""
tests/test_prompt_builder.py
Unit tests for the prompt builder.
"""

import pytest
from backend.prompts import build_prompt, SYSTEM_PROMPT


def _make_chunk(
    content: str,
    source: str = "doc.pdf",
    modality: str = "text",
    page: int = 1,
    start_ts: float = None,
    end_ts: float = None,
) -> dict:
    return {
        "content": content,
        "source_file": source,
        "modality": modality,
        "page_number": page,
        "start_timestamp": start_ts,
        "end_timestamp": end_ts,
    }


class TestBuildPrompt:
    def test_returns_tuple(self):
        chunks = [_make_chunk("Hello world")]
        result = build_prompt("What is this about?", chunks)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_system_message_not_empty(self):
        system_msg, _ = build_prompt("Question?", [_make_chunk("content")])
        assert len(system_msg) > 0
        assert "context" in system_msg.lower() or "assistant" in system_msg.lower()

    def test_user_message_contains_query(self):
        query = "What is the revenue for Q3?"
        _, user_msg = build_prompt(query, [_make_chunk("Revenue was $10M")])
        assert query in user_msg

    def test_user_message_contains_content(self):
        content = "The annual report shows 25% growth."
        _, user_msg = build_prompt("What is the growth?", [_make_chunk(content)])
        assert content in user_msg

    def test_no_chunks_shows_fallback(self):
        _, user_msg = build_prompt("What?", [])
        assert "No relevant context" in user_msg

    def test_page_number_shown_for_text(self):
        chunk = _make_chunk("Content here", page=7)
        _, user_msg = build_prompt("?", [chunk])
        assert "Page: 7" in user_msg

    def test_timestamp_shown_for_audio(self):
        chunk = _make_chunk("Spoken content", modality="audio", page=None, start_ts=12.5, end_ts=45.0)
        _, user_msg = build_prompt("?", [chunk])
        assert "12.5" in user_msg
        assert "45.0" in user_msg

    def test_multiple_chunks_included(self):
        chunks = [_make_chunk(f"Content {i}", page=i) for i in range(3)]
        _, user_msg = build_prompt("?", chunks)
        assert "Context 1" in user_msg
        assert "Content 0" in user_msg

    def test_context_length_limit(self):
        # Very long content should be truncated
        big_content = "A" * 5000
        chunks = [_make_chunk(big_content) for _ in range(10)]
        _, user_msg = build_prompt("?", chunks, max_context_chars=6000)
        assert len(user_msg) < 15000  # Should be bounded
