"""
backend/evaluation/evaluator.py
Lightweight evaluation without external frameworks.
Tracks retrieval quality, response quality, and latency.
"""

import re
from typing import List, Optional


def extract_confidence(response_text: str) -> float:
    """
    Parse [Confidence: X.XX] from the LLM response.
    Defaults to 0.5 if not found.
    """
    match = re.search(r"\[Confidence:\s*(-?[\d.]+)\]", response_text, re.IGNORECASE)
    if match:
        try:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        except ValueError:
            pass
    return 0.5


def compute_metrics(
    query: str,
    retrieved_chunks: List[dict],
    response: str,
    latency_ms: float,
    model: str,
) -> dict:
    """
    Compute lightweight metrics for a RAG response.

    Returns:
        dict with evaluation metrics.
    """
    confidence = extract_confidence(response)
    context_length = sum(len(c.get("content", "")) for c in retrieved_chunks)
    avg_retrieval_score = (
        sum(c.get("score", 0.0) for c in retrieved_chunks) / len(retrieved_chunks)
        if retrieved_chunks else 0.0
    )
    avg_rerank_score = (
        sum(c.get("rerank_score", 0.0) for c in retrieved_chunks) / len(retrieved_chunks)
        if retrieved_chunks and "rerank_score" in retrieved_chunks[0] else None
    )

    has_sufficient_evidence = (
        "I could not find sufficient evidence" not in response
    )

    return {
        "retrieval_count": len(retrieved_chunks),
        "context_length_chars": context_length,
        "response_length_chars": len(response),
        "confidence_score": confidence,
        "avg_retrieval_score": round(avg_retrieval_score, 4),
        "avg_rerank_score": round(avg_rerank_score, 4) if avg_rerank_score is not None else None,
        "latency_ms": round(latency_ms, 2),
        "model": model,
        "has_sufficient_evidence": has_sufficient_evidence,
    }


def format_source_references(chunks: List[dict]) -> List[dict]:
    """
    Build human-readable source references from retrieved chunks.
    """
    refs = []
    seen = set()

    for chunk in chunks:
        source = chunk.get("source_file", "unknown")
        modality = chunk.get("modality", "text")
        page = chunk.get("page_number")
        start_ts = chunk.get("start_timestamp")
        end_ts = chunk.get("end_timestamp")

        if modality == "text" and page:
            key = (source, page)
            if key not in seen:
                refs.append({"source": source, "page": page, "modality": modality})
                seen.add(key)
        elif modality in ("audio", "video") and start_ts is not None:
            key = (source, round(start_ts, 0))
            if key not in seen:
                refs.append({
                    "source": source,
                    "start_timestamp": start_ts,
                    "end_timestamp": end_ts,
                    "modality": modality,
                })
                seen.add(key)
        else:
            key = source
            if key not in seen:
                refs.append({"source": source, "modality": modality})
                seen.add(key)

    return refs
