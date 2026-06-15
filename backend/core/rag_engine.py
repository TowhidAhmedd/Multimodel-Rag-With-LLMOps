"""
backend/core/rag_engine.py
Top-level RAG orchestrator.
Ties together: retrieval → prompt → LLM → evaluation → tracing.
"""

import time
import uuid
from typing import Optional

from backend.retrieval import retrieve
from backend.prompts import build_prompt
from backend.llm import GroqProvider
from backend.cache import get_cached, set_cached
from backend.observability import get_tracer
from backend.evaluation import compute_metrics, format_source_references
from backend.logging_config import get_logger, log_request
from backend.vectorstore import VectorStoreSession

logger = get_logger(__name__)


async def answer_query(
    query: str,
    session: VectorStoreSession,
    model: str,
    source_file: Optional[str] = None,
    top_k: int = 5,
) -> dict:
    """
    Execute the full RAG pipeline for a user query.
    Returns a structured response dict.
    """
    request_id = str(uuid.uuid4())
    start_time = time.monotonic()

    # --- Cache check ---
    cached = get_cached(query, model, source_file)
    if cached:
        cached["from_cache"] = True
        return cached

    # --- Retrieval ---
    chunks = await retrieve(
        query=query,
        session=session,
        top_k=top_k,
        source_file=source_file,
    )

    if not chunks:
        response_text = "I could not find sufficient evidence in the uploaded content."
        latency_ms = (time.monotonic() - start_time) * 1000
        result = {
            "answer":     response_text,
            "sources":    [],
            "chunks":     [],
            "metrics":    compute_metrics(query, [], response_text, latency_ms, model),
            "model":      model,
            "latency_ms": latency_ms,
            "trace_id":   None,
            "request_id": request_id,
            "from_cache": False,
        }
        return result

    # --- Prompt ---
    system_msg, user_msg = build_prompt(query, chunks)
    full_prompt = f"{system_msg}\n\n{user_msg}"

    # --- LLM ---
    llm = GroqProvider(model=model)
    response_text = await llm.async_generate(
        system_message=system_msg,
        user_message=user_msg,
    )

    latency_ms = (time.monotonic() - start_time) * 1000

    # --- Evaluation ---
    metrics = compute_metrics(query, chunks, response_text, latency_ms, model)
    sources = format_source_references(chunks)

    # --- Trace ---
    tracer = get_tracer()
    trace_id = tracer.trace_query(
        query=query,
        retrieved_chunks=chunks,
        prompt=full_prompt,
        response=response_text,
        model=model,
        latency_ms=latency_ms,
        metadata={"source_file": source_file, "request_id": request_id},
    )

    # --- Logging ---
    log_request(
        logger,
        query=query,
        model=model,
        latency_ms=latency_ms,
        request_id=request_id,
        trace_id=trace_id,
        source_file=source_file,
    )

    result = {
        "answer":  response_text,
        "sources": sources,
        "chunks": [
            {
                "content":         c["content"][:300] + "..." if len(c["content"]) > 300 else c["content"],
                "source_file":     c["source_file"],
                "modality":        c["modality"],
                "page_number":     c.get("page_number"),
                "start_timestamp": c.get("start_timestamp"),
                "end_timestamp":   c.get("end_timestamp"),
                "score":           c.get("score"),
                "rerank_score":    c.get("rerank_score"),
            }
            for c in chunks
        ],
        "metrics":    metrics,
        "model":      model,
        "latency_ms": round(latency_ms, 2),
        "trace_id":   trace_id,
        "request_id": request_id,
        "from_cache": False,
    }

    set_cached(query, model, result, source_file)
    return result
