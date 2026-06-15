"""
tests/conftest.py
Shared pytest configuration.
Sets required environment variables before any module is imported.
"""

import os

os.environ.setdefault("GROQ_API_KEY",          "test_key_placeholder")
os.environ.setdefault("LANGSMITH_API_KEY",     "")
os.environ.setdefault("LANGSMITH_PROJECT",     "test-project")
os.environ.setdefault("PINECONE_API_KEY",      "test_pinecone_key")
os.environ.setdefault("PINECONE_INDEX_NAME",   "test-index")
os.environ.setdefault("PINECONE_CLOUD",        "aws")
os.environ.setdefault("PINECONE_REGION",       "us-east-1")
os.environ.setdefault("EMBEDDING_MODEL",       "BAAI/bge-small-en-v1.5")
os.environ.setdefault("DEFAULT_LLM_MODEL",     "llama-3.3-70b-versatile")
os.environ.setdefault("UPLOAD_DIR",            "/tmp/rag_test_uploads")
