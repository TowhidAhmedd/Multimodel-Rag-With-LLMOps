"""
backend/logging_config/logger.py
Structured JSON logging for the application.
"""

import logging
import json
import sys
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Format log records as JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Attach any extra fields passed via extra={}
        for key, value in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "created", "filename", "funcName",
                "levelname", "levelno", "lineno", "module", "msecs",
                "message", "pathname", "process", "processName",
                "relativeCreated", "stack_info", "thread", "threadName",
                "exc_info", "exc_text",
            ):
                log_obj[key] = value

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj, default=str)


def get_logger(name: str) -> logging.Logger:
    """Return a JSON-structured logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger


def log_request(
    logger: logging.Logger,
    query: str,
    model: str,
    latency_ms: float,
    request_id: Optional[str] = None,
    trace_id: Optional[str] = None,
    source_file: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Emit a structured log entry for a RAG request."""
    logger.info(
        "rag_request",
        extra={
            "request_id": request_id or str(uuid.uuid4()),
            "trace_id": trace_id,
            "query": query[:200],
            "model": model,
            "source_file": source_file,
            "latency_ms": round(latency_ms, 2),
            "error": error,
        },
    )
