"""
backend/prompts/builder.py
Constructs grounded RAG prompts from retrieved context chunks.
"""

from typing import List, Optional


SYSTEM_PROMPT = """You are a precise document assistant. Your job is to answer questions STRICTLY based on the provided context.

Rules:
1. ONLY use information from the provided context. Never add external knowledge.
2. If the context does not contain enough information, respond EXACTLY with:
   "I could not find sufficient evidence in the uploaded content."
3. Always cite your sources using [Source: filename, Page: X] for documents or [Source: filename, Time: Xs-Ys] for audio/video.
4. Provide a confidence score from 0.0 to 1.0 at the end of your response in this exact format:
   [Confidence: 0.85]
5. Be concise, accurate, and helpful.
"""


def _format_chunk(chunk: dict, index: int) -> str:
    """Format a single chunk into a readable context block."""
    source = chunk.get("source_file", "unknown")
    modality = chunk.get("modality", "text")
    lines = [f"[Context {index + 1}] Source: {source}"]

    if modality == "text" and chunk.get("page_number"):
        lines.append(f"Page: {chunk['page_number']}")
    elif modality in ("audio", "video"):
        start = chunk.get("start_timestamp")
        end = chunk.get("end_timestamp")
        if start is not None and end is not None:
            lines.append(f"Timestamp: {start:.1f}s – {end:.1f}s")

    lines.append(f"Content: {chunk['content']}")
    return "\n".join(lines)


def build_prompt(
    query: str,
    chunks: List[dict],
    max_context_chars: int = 6000,
) -> tuple[str, str]:
    """
    Build the system and user messages for the LLM.

    Returns:
        (system_message, user_message)
    """
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(chunks):
        block = _format_chunk(chunk, i)
        if total_chars + len(block) > max_context_chars:
            break
        context_parts.append(block)
        total_chars += len(block)

    if not context_parts:
        context_section = "No relevant context found."
    else:
        context_section = "\n\n---\n\n".join(context_parts)

    user_message = f"""Context from uploaded files:

{context_section}

---

Question: {query}

Answer strictly from the context above. Include source citations and a confidence score."""

    return SYSTEM_PROMPT, user_message
