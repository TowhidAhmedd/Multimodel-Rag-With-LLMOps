
# 🎥 Demo Video

[![Watch Demo](https://img.youtube.com/vi/tHPU-L5qwq8/maxresdefault.jpg)](https://www.youtube.com/watch?v=tHPU-L5qwq8)


# 🧠 Multimodal RAG with LangSmith & pgvector

A production-style **Multimodal Retrieval-Augmented Generation (RAG)** application that lets you upload documents, audio, and video files, then ask grounded questions answered **strictly from your uploaded content** — no hallucinations.

---
## Architecture
```mermaid

flowchart TD

    UI[Streamlit Frontend] --> API[FastAPI Gateway]

    API --> AUTH[Auth / Rate Limiter]
    AUTH --> ORCH[RAG Orchestrator]

    ORCH --> ING[Ingestion Service]
    ORCH --> RET[Retrieval Service]
    ORCH --> LLM[LLM Service]
    ORCH --> OBS[Observability Service]

    ING --> CHUNK[Chunking Engine]
    CHUNK --> EMB[Embedding Service - BGE-small]
    EMB --> DB[(Pinecone)]

    RET --> QEMB[Query Embedding]
    QEMB --> SEARCH[Vector Search]
    SEARCH --> DB
    SEARCH --> RERANK[Cross Encoder Re-ranker]

    RERANK --> PROMPT[Prompt Builder]
    PROMPT --> LLM_API[Groq LLM API]

    LLM_API --> OBS

    OBS --> LANGSMITH[LangSmith Tracing]
    OBS --> METRICS[Evaluation Metrics Engine]

    LLM_API --> RESPONSE[Structured Response API]

    RESPONSE --> UI

    
```


<!--
## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI (8501)                   │
│         Upload Files │ Select Model │ Ask Question       │
└──────────────────────────┬──────────────────────────────┘
                           │ HTTP
┌──────────────────────────▼──────────────────────────────┐
│                   FastAPI Backend (8000)                 │
│    POST /upload  │  POST /query  │  GET /health         │
└──┬──────────────────────┬──────────────────────────┬────┘
   │                      │                          │
   ▼                      ▼                          ▼
Ingestion             Retrieval                  Evaluation
Pipeline              Pipeline                   & Logging
   │                      │
   ├─ PDF (pdfplumber)     ├─ Embed Query
   ├─ DOCX (python-docx)  │   (BGE-small)
   ├─ TXT                 │
   ├─ Audio (Whisper)     ├─ pgvector Cosine Search
   └─ Video (ffmpeg+      │
      Whisper)            ├─ Cross-Encoder Re-rank
                          │   (MiniLM)
           │              │
           ▼              ▼
     ┌─────────────────────────┐
     │  PostgreSQL + pgvector  │
     │  (embeddings + chunks)  │
     └─────────────────────────┘
                          │
                          ▼
                   Prompt Builder
                          │
                          ▼
                    Groq LLM API
              (llama-3.3-70b-versatile)
                          │
                          ▼
                  LangSmith Tracing
                          │
                          ▼
                 Structured Response
          (Answer + Sources + Metrics)
```
-->

---

## ✨ Features

| Feature | Details |
|---|---|
| **Document support** | PDF (page numbers), DOCX, TXT |
| **Audio support** | MP3, WAV, M4A → Whisper → timestamps |
| **Video support** | MP4, AVI, MOV → audio extract → Whisper → timestamps |
| **Embeddings** | BAAI/bge-small-en-v1.5 (384-dim, fast, accurate) |
| **Vector DB** | PostgreSQL + pgvector (cosine similarity) |
| **Re-ranking** | Cross-Encoder MiniLM for precision boost |
| **LLM** | Groq API (llama-3.3-70b, llama-3.1-8b, mixtral) |
| **Observability** | LangSmith trace per request |
| **Caching** | In-memory TTL cache (cachetools) |
| **Grounded answers** | Strict context-only prompting |
| **Confidence scores** | Model self-reports 0.0–1.0 |

---

## 📁 Project Structure

```
multimodal_rag/
├── frontend/
│   └── streamlit_app.py          # Streamlit UI
│
├── backend/
│   ├── main.py                   # FastAPI app entry point
│   ├── api/
│   │   ├── routes.py             # API endpoints
│   │   └── schemas.py            # Pydantic models
│   ├── config/
│   │   └── settings.py           # Centralised config (pydantic-settings)
│   ├── core/
│   │   └── rag_engine.py         # Top-level RAG orchestrator
│   ├── ingestion/
│   │   ├── pipeline.py           # Ingestion orchestrator
│   │   ├── document_ingestor.py  # PDF / DOCX / TXT
│   │   ├── audio_ingestor.py     # MP3 / WAV / M4A
│   │   └── video_ingestor.py     # MP4 / AVI / MOV
│   ├── chunking/
│   │   └── chunker.py            # Sliding-window chunker
│   ├── embeddings/
│   │   └── embedder.py           # BGE-small sentence-transformer
│   ├── vectorstore/
│   │   ├── database.py           # SQLAlchemy async + pgvector ORM
│   │   └── store.py              # Insert / search / delete ops
│   ├── retrieval/
│   │   └── retriever.py          # Embed → search → rerank pipeline
│   ├── reranking/
│   │   └── reranker.py           # Cross-encoder MiniLM
│   ├── prompts/
│   │   └── builder.py            # Grounded prompt construction
│   ├── llm/
│   │   └── groq_provider.py      # Groq API wrapper + retry
│   ├── cache/
│   │   └── query_cache.py        # cachetools TTLCache
│   ├── observability/
│   │   └── langsmith_tracker.py  # LangSmith run creation
│   ├── evaluation/
│   │   └── evaluator.py          # Lightweight metrics
│   └── logging_config/
│       └── logger.py             # Structured JSON logging
│
├── storage/uploads/              # Uploaded files
├── tests/                        # Pytest test suite
├── docker/
│   ├── Dockerfile.backend
│   └── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
├── pytest.ini
├── .env.example
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- A [Groq API key](https://console.groq.com)
- A [LangSmith API key](https://smith.langchain.com) _(optional but recommended)_

### 1. Clone & configure

```bash
git clone <your-repo>
cd multimodal_rag
cp .env.example .env
```

Edit `.env`:

```env
GROQ_API_KEY=gsk_your_key_here
LANGSMITH_API_KEY=ls__your_key_here   # optional
LANGSMITH_PROJECT=multimodal-rag
```

### 2. Run with Docker

```bash
docker compose up --build
```

- **Streamlit UI** → http://localhost:8501
- **FastAPI docs** → http://localhost:8000/docs
- **Health check** → http://localhost:8000/health

---

## 🖥️ Running Locally (without Docker)

### Prerequisites

- Python 3.11+
- PostgreSQL 14+ with pgvector extension
- ffmpeg installed (`sudo apt install ffmpeg` / `brew install ffmpeg`)

### PostgreSQL Setup

```bash
# Install PostgreSQL
sudo apt install postgresql postgresql-contrib

# Connect and create database
psql -U postgres
```

```sql
CREATE USER raguser WITH PASSWORD 'ragpassword';
CREATE DATABASE ragdb OWNER raguser;
\c ragdb
CREATE EXTENSION vector;
\q
```

### pgvector Setup

If pgvector is not bundled with your PostgreSQL:

```bash
# Ubuntu/Debian
sudo apt install postgresql-16-pgvector

# macOS with Homebrew
brew install pgvector
```

### Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Set environment variables

```bash
cp .env.example .env
# Edit .env with your DATABASE_URL pointing to local postgres
# DATABASE_URL=postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb
```

### Start the backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### Start the frontend

```bash
BACKEND_URL=http://localhost:8000 streamlit run frontend/streamlit_app.py
```

---

## 🔑 API Keys Setup

### Groq API

1. Visit https://console.groq.com
2. Create an account → API Keys → Create Key
3. Copy the key to `GROQ_API_KEY` in `.env`

**Supported models:**

| Model | Speed | Quality |
|---|---|---|
| `llama-3.3-70b-versatile` | Medium | Best |
| `llama-3.1-8b-instant` | Fast | Good |
| `mixtral-8x7b-32768` | Medium | Good |

### LangSmith Setup

1. Visit https://smith.langchain.com
2. Sign up → Settings → API Keys → Create
3. Copy to `LANGSMITH_API_KEY` in `.env`
4. Set `LANGSMITH_PROJECT=multimodal-rag`

Every query automatically creates a trace in your LangSmith project. The Trace ID is displayed in the UI.

---

## 📡 API Documentation

### `POST /upload`

Upload a file for ingestion.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@document.pdf"
```

**Response:**
```json
{
  "status": "success",
  "source_file": "document.pdf",
  "chunk_count": 42,
  "modality": "text"
}
```

---

### `POST /query`

Ask a question against uploaded content.

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main findings?",
    "model": "llama-3.3-70b-versatile",
    "top_k": 5
  }'
```

**Response:**
```json
{
  "answer": "The main findings indicate... [Source: report.pdf, Page: 3] [Confidence: 0.87]",
  "sources": [{"source": "report.pdf", "page": 3, "modality": "text"}],
  "chunks": [...],
  "metrics": {
    "retrieval_count": 5,
    "confidence_score": 0.87,
    "latency_ms": 1234.5
  },
  "model": "llama-3.3-70b-versatile",
  "latency_ms": 1234.5,
  "trace_id": "uuid-here",
  "request_id": "uuid-here",
  "from_cache": false
}
```

**Optional fields:**
- `source_file` — filter search to one specific uploaded file
- `top_k` — number of chunks to retrieve (1–20, default 5)

---

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "database": "connected",
  "embedding_model": "BAAI/bge-small-en-v1.5",
  "default_model": "llama-3.3-70b-versatile"
}
```

---

### `GET /metrics`

```bash
curl http://localhost:8000/metrics
```

```json
{
  "cache_size": 12,
  "cache_maxsize": 256,
  "cache_ttl_seconds": 3600,
  "supported_models": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"],
  "embedding_model": "BAAI/bge-small-en-v1.5"
}
```

---

### `DELETE /cache`

Clear the in-memory query cache.

```bash
curl -X DELETE http://localhost:8000/cache
```

---

## 🧪 Testing

```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_chunker.py -v
pytest tests/test_evaluator.py -v
pytest tests/test_cache.py -v
pytest tests/test_reranker.py -v
pytest tests/test_api.py -v
```

---

## ⚙️ Configuration Reference

| Variable | Default | Description |
|---|---|---|
| `GROQ_API_KEY` | required | Groq API key |
| `LANGSMITH_API_KEY` | optional | LangSmith API key |
| `LANGSMITH_PROJECT` | `multimodal-rag` | LangSmith project name |
| `DATABASE_URL` | postgres URL | Async SQLAlchemy URL |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Sentence transformer model |
| `DEFAULT_LLM_MODEL` | `llama-3.3-70b-versatile` | Default Groq model |
| `MAX_UPLOAD_SIZE_MB` | `200` | Max file upload size |
| `CHUNK_SIZE` | `800` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `5` | Chunks to retrieve |
| `CACHE_MAX_SIZE` | `256` | Max cached responses |
| `CACHE_TTL_SECONDS` | `3600` | Cache TTL (1 hour) |

---

## 🐛 Troubleshooting

### Backend won't start

```bash
# Check logs
docker compose logs backend

# Common causes:
# - Missing GROQ_API_KEY in .env
# - PostgreSQL not ready yet (wait 30s after compose up)
```

### pgvector extension missing

```sql
-- Connect to your database and run:
CREATE EXTENSION IF NOT EXISTS vector;
```

### Whisper model download fails

The Whisper "small" model downloads automatically on first use (~244 MB). Ensure internet access from the container. Pre-downloaded in Dockerfile to avoid cold starts.

### Out of memory errors

The embedding model and re-ranker are CPU-only by default. On machines with <4GB RAM, switch to a smaller Whisper model in `audio_ingestor.py`:

```python
_WHISPER_MODEL = WhisperModel("tiny", device="cpu", compute_type="int8")
```

### LangSmith traces not appearing

1. Verify `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT` in `.env`
2. Check `LANGSMITH_TRACING_V2=true`
3. Confirm your project name matches what you see at smith.langchain.com

### File upload 413 error

Increase `MAX_UPLOAD_SIZE_MB` in `.env` and rebuild.

### Slow first query

Models are loaded lazily on first request. Allow 30–60s for:
- BGE-small embedding model
- MiniLM cross-encoder
- Whisper (audio/video only)

Subsequent requests will be fast.

---

## 🏅 Interview Talking Points

This project demonstrates:

1. **Multimodal RAG pipeline** — unified ingestion across text, audio, and video
2. **pgvector** — production-grade vector search without a separate vector DB service
3. **Cross-encoder re-ranking** — two-stage retrieval for precision
4. **Async FastAPI** — non-blocking I/O with SQLAlchemy async
5. **LangSmith observability** — every request traced end-to-end
6. **Structured logging** — JSON logs with request_id and trace_id correlation
7. **TTL caching** — reduces latency and LLM costs for repeated queries
8. **Grounded prompting** — strict context-only answering with confidence scores
9. **Docker Compose** — single command deployment
10. **Modular architecture** — each component independently testable

---

## 📄 License

MIT License — use freely for learning and production.
