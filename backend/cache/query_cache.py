"""
backend/cache/query_cache.py
Simple in-memory TTL cache for query → response pairs.
Uses cachetools.TTLCache (thread-safe).
"""

import hashlib
import json
from typing import Optional

from cachetools import TTLCache
import threading

from backend.config import get_settings
from backend.logging_config import get_logger

logger = get_logger(__name__)
settings = get_settings()

# Thread-safe TTL cache
_cache: TTLCache = TTLCache(
    maxsize=settings.cache_max_size,
    ttl=settings.cache_ttl_seconds,
)
_lock = threading.Lock()


def _make_key(query: str, model: str, source_file: Optional[str] = None) -> str:
    """Generate a deterministic cache key."""
    payload = json.dumps(
        {"query": query.strip().lower(), "model": model, "source": source_file},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def get_cached(
    query: str,
    model: str,
    source_file: Optional[str] = None,
) -> Optional[dict]:
    """Return cached response dict or None."""
    key = _make_key(query, model, source_file)
    with _lock:
        result = _cache.get(key)
    if result:
        logger.info("Cache hit", extra={"key": key[:16]})
    return result


def set_cached(
    query: str,
    model: str,
    response: dict,
    source_file: Optional[str] = None,
) -> None:
    """Store a response in the cache."""
    key = _make_key(query, model, source_file)
    with _lock:
        _cache[key] = response
    logger.info("Cache set", extra={"key": key[:16]})


def clear_cache() -> int:
    """Clear all cached entries and return the count cleared."""
    with _lock:
        count = len(_cache)
        _cache.clear()
    logger.info("Cache cleared", extra={"cleared": count})
    return count


def cache_stats() -> dict:
    """Return current cache statistics."""
    with _lock:
        return {
            "size": len(_cache),
            "maxsize": _cache.maxsize,
            "ttl_seconds": settings.cache_ttl_seconds,
        }
