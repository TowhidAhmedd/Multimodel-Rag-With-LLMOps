"""
backend/config/settings.py
Central configuration – pydantic-settings v2 compatible.
"""


from functools import lru_cache
# pyrefly: ignore [missing-import]
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Groq ---
    groq_api_key: str = "changeme"
    default_llm_model: str = "llama-3.3-70b-versatile"
    llm_timeout: int = 200  # seconds

    # --- LangSmith ---
    langsmith_api_key: str = ""
    langsmith_project: str = "multimodal-rag"
    langsmith_tracing_v2: str = "true"

    # --- Pinecone ---
    pinecone_api_key: str = "changeme"
    pinecone_index_name: str = "multimodal-rag"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Embeddings ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    # --- File Upload ---
    max_upload_size_mb: int = 200
    upload_dir: str = "storage/uploads"

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 150

    # --- Retrieval ---
    top_k: int = 5

    # --- Cache ---
    cache_max_size: int = 256
    cache_ttl_seconds: int = 3600

    # --- Logging ---
    log_level: str = "INFO"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
