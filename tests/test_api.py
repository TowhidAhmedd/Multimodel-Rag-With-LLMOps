"""
tests/test_api.py
API endpoint tests using FastAPI TestClient.
All Pinecone, ML, and external service dependencies are mocked.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """
    Build a TestClient with Pinecone and init_db mocked out so
    no real network calls are made during tests.
    """
    # Mock the Pinecone client and index before any import triggers it
    mock_index = MagicMock()
    mock_index.describe_index_stats.return_value = {"total_vector_count": 0}
    mock_index.query.return_value = {"matches": []}
    mock_index.upsert.return_value = None

    with patch("backend.vectorstore.database._get_client") as mock_client_fn, \
         patch("backend.vectorstore.database.init_db", new_callable=AsyncMock), \
         patch("backend.vectorstore.database._pinecone_index", mock_index):

        mock_pc = MagicMock()
        mock_pc.has_index.return_value = True
        mock_pc.Index.return_value = mock_index
        mock_client_fn.return_value = mock_pc

        from fastapi.testclient import TestClient
        from backend.main import app

        with TestClient(app, raise_server_exceptions=False) as c:
            yield c


# ── /metrics ──────────────────────────────────────────────────────────────────

class TestMetricsEndpoint:
    def test_returns_200(self, client):
        assert client.get("/metrics").status_code == 200

    def test_response_schema(self, client):
        data = client.get("/metrics").json()
        assert "cache_size" in data
        assert "cache_maxsize" in data
        assert "supported_models" in data
        assert "embedding_model" in data
        assert isinstance(data["supported_models"], list)
        assert len(data["supported_models"]) == 3

    def test_content_type_json(self, client):
        assert client.get("/metrics").headers["content-type"].startswith("application/json")


# ── DELETE /cache ─────────────────────────────────────────────────────────────

class TestCacheEndpoint:
    def test_returns_200(self, client):
        assert client.delete("/cache").status_code == 200

    def test_response_body(self, client):
        data = client.delete("/cache").json()
        assert "cleared" in data
        assert data["status"] == "ok"


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_returns_json(self, client):
        resp = client.get("/health")
        assert resp.headers["content-type"].startswith("application/json")

    def test_status_code_valid(self, client):
        assert client.get("/health").status_code in (200, 500, 503)


# ── POST /query – schema validation ──────────────────────────────────────────

class TestQueryValidation:
    def test_empty_query_rejected(self, client):
        resp = client.post("/query", json={"query": "", "model": "llama-3.3-70b-versatile"})
        assert resp.status_code == 422

    def test_missing_query_rejected(self, client):
        resp = client.post("/query", json={"model": "llama-3.3-70b-versatile"})
        assert resp.status_code == 422

    def test_invalid_model_rejected(self, client):
        resp = client.post("/query", json={"query": "What is AI?", "model": "gpt-99"})
        assert resp.status_code == 400

    def test_invalid_model_error_message(self, client):
        resp = client.post("/query", json={"query": "Hello", "model": "fake-model"})
        assert "not supported" in resp.json()["detail"].lower()

    def test_top_k_zero_rejected(self, client):
        resp = client.post("/query", json={"query": "Hello", "model": "llama-3.3-70b-versatile", "top_k": 0})
        assert resp.status_code == 422

    def test_top_k_too_large_rejected(self, client):
        resp = client.post("/query", json={"query": "Hello", "model": "llama-3.3-70b-versatile", "top_k": 999})
        assert resp.status_code == 422

    def test_valid_models_pass_schema(self, client):
        for model in ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"]:
            resp = client.post("/query", json={"query": "Hello", "model": model})
            assert resp.status_code != 422, f"Model '{model}' failed schema validation"


# ── POST /upload – extension validation ──────────────────────────────────────

class TestUploadValidation:
    def test_no_file_rejected(self, client):
        assert client.post("/upload").status_code == 422

    def test_unsupported_extension_exe(self, client):
        resp = client.post("/upload", files={"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")})
        assert resp.status_code == 415

    def test_unsupported_extension_zip(self, client):
        resp = client.post("/upload", files={"file": ("archive.zip", b"PK\x03\x04", "application/zip")})
        assert resp.status_code == 415

    def test_error_message_mentions_not_supported(self, client):
        resp = client.post("/upload", files={"file": ("bad.xyz", b"data", "application/octet-stream")})
        assert "not supported" in resp.json()["detail"].lower()

    def test_supported_pdf_passes_extension_check(self, client):
        resp = client.post("/upload", files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")})
        assert resp.status_code != 415

    def test_supported_txt_passes_extension_check(self, client):
        resp = client.post("/upload", files={"file": ("note.txt", b"Hello world", "text/plain")})
        assert resp.status_code != 415

    def test_supported_mp3_passes_extension_check(self, client):
        resp = client.post("/upload", files={"file": ("audio.mp3", b"\xff\xfb fake", "audio/mpeg")})
        assert resp.status_code != 415
