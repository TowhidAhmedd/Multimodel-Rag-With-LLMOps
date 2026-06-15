"""
backend/api/schemas.py
Pydantic models for FastAPI request and response validation.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ─── Upload ───────────────────────────────────────────────────────────────────

class UploadResponse(BaseModel):
    status: str
    source_file: str
    chunk_count: int
    modality: str
    message: Optional[str] = None


# ─── Query ────────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    model: str = Field("llama-3.3-70b-versatile")
    source_file: Optional[str] = None
    top_k: int = Field(5, ge=1, le=20)


class SourceReference(BaseModel):
    source: str
    modality: str
    page: Optional[int] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None


class RetrievedChunk(BaseModel):
    content: str
    source_file: str
    modality: str
    page_number: Optional[int] = None
    start_timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None


class EvaluationMetrics(BaseModel):
    retrieval_count: int
    context_length_chars: int
    response_length_chars: int
    confidence_score: float
    avg_retrieval_score: float
    avg_rerank_score: Optional[float]
    latency_ms: float
    model: str
    has_sufficient_evidence: bool


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    chunks: List[Dict[str, Any]]
    metrics: Dict[str, Any]
    model: str
    latency_ms: float
    trace_id: Optional[str] = None
    request_id: str
    from_cache: bool


# ─── Health / Metrics ────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    database: str
    embedding_model: str
    default_model: str


class MetricsResponse(BaseModel):
    cache_size: int
    cache_maxsize: int
    cache_ttl_seconds: int
    supported_models: List[str]
    embedding_model: str
