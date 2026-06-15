"""
backend/observability/langsmith_tracker.py
LangSmith tracing integration using the langsmith SDK.
Each RAG request gets a traceable run that appears in the LangSmith UI.
"""

import os
import uuid
import datetime
from typing import Any, Dict, List, Optional

from backend.config import get_settings
from backend.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()


def _configure_langsmith() -> bool:
    """Set env vars required by the LangSmith SDK. Return True if configured."""
    if not settings.langsmith_api_key:
        logger.warning("LANGSMITH_API_KEY not set — tracing disabled")
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    return True


_LANGSMITH_ENABLED = _configure_langsmith()


class RAGTracer:
    """
    Thin wrapper around the LangSmith Client.
    Creates one run per RAG query and records inputs, outputs, and latency.
    """

    def __init__(self) -> None:
        self.enabled = _LANGSMITH_ENABLED
        self._client = None

        if self.enabled:
            try:
                from langsmith import Client  # lazy import
                self._client = Client(api_key=settings.langsmith_api_key)
                logger.info("LangSmith tracing enabled",
                            extra={"project": settings.langsmith_project})
            except Exception as exc:
                logger.warning("LangSmith init failed — tracing disabled",
                               extra={"error": str(exc)})
                self.enabled = False

    def trace_query(
        self,
        query: str,
        retrieved_chunks: List[dict],
        prompt: str,
        response: str,
        model: str,
        latency_ms: float,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> str:
        """
        Create a LangSmith run for one RAG request.
        Returns a trace_id string regardless of whether tracing succeeded.
        """
        trace_id = str(uuid.uuid4())

        if not self.enabled or self._client is None:
            return trace_id

        try:
            run_id = uuid.UUID(trace_id)
            now = datetime.datetime.utcnow()

            self._client.create_run(
                id=run_id,
                name="multimodal_rag_query",
                run_type="chain",
                project_name=settings.langsmith_project,
                start_time=now,
                inputs={
                    "query": query,
                    "num_retrieved_chunks": len(retrieved_chunks),
                    "prompt_chars": len(prompt),
                    "model": model,
                },
                outputs={
                    "response": response,
                    "latency_ms": round(latency_ms, 2),
                    "error": error,
                },
                tags=["multimodal-rag", model],
                extra={"metadata": metadata or {}},
            )

            # Close the run immediately (non-streaming)
            self._client.update_run(
                run_id,
                end_time=datetime.datetime.utcnow(),
                error=error,
            )

            logger.info("LangSmith run created", extra={"trace_id": trace_id})

        except Exception as exc:
            logger.warning("LangSmith trace failed — continuing without trace",
                           extra={"error": str(exc)})

        return trace_id


# ── Module-level singleton ────────────────────────────────────────────────────

_tracer: Optional[RAGTracer] = None


def get_tracer() -> RAGTracer:
    global _tracer
    if _tracer is None:
        _tracer = RAGTracer()
    return _tracer
