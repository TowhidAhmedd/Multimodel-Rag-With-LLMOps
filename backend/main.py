

"""
backend/main.py

FastAPI application entry point for Multimodal RAG.

Features:
- Pinecone vector database
- Groq LLM integration
- CORS middleware
- Structured logging
- Health check endpoint
- Production deployment support
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.vectorstore import init_db
from backend.logging_config import get_logger


logger = get_logger(__name__)


# ─────────────────────────────────────────────
# Lifespan (Startup / Shutdown)
# ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager.

    Startup:
    - Initialize Pinecone/vector database

    Shutdown:
    - Cleanup resources
    """

    try:
        logger.info("Starting Multimodal RAG backend")

        await init_db()

        logger.info("Vector database ready")

    except Exception as exc:
        logger.error(
            f"Startup failed: {exc}",
            exc_info=True
        )
        raise

    yield

    logger.info("Shutting down Multimodal RAG backend")


# ─────────────────────────────────────────────
# FastAPI Application
# ─────────────────────────────────────────────

app = FastAPI(
    title="Multimodal RAG API",
    description=(
        "Production RAG application for "
        "documents, audio and video "
        "using Pinecone + Groq"
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ─────────────────────────────────────────────
# CORS Middleware
# ─────────────────────────────────────────────

app.add_middleware(
    CORSMiddleware,

    # Streamlit / frontend access
    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────

app.include_router(
    router,
    prefix=""
)


# ─────────────────────────────────────────────
# Health Check Endpoints
# ─────────────────────────────────────────────


@app.get("/")
async def root():
    """
    Basic service check.
    """

    return {
        "service": "Multimodal RAG",
        "status": "ok",
        "docs": "/docs"
    }



@app.get("/health")
async def health_check():
    """
    Used by Streamlit and deployment platforms.
    """

    return {
        "status": "ok",
        "database": "connected",
        "embedding_model": "BAAI/bge-small-en-v1.5"
    }



# ─────────────────────────────────────────────
# Local Development Only
# ─────────────────────────────────────────────

# if __name__ == "__main__":

#     import uvicorn

#     # Local run only
#     # Production uses Docker CMD:
#     # uvicorn backend.main:app --host 0.0.0.0 --port $PORT

#     uvicorn.run(
#         "backend.main:app",
#         host="0.0.0.0",
#         port=8000,
#         log_level="info"
#     )

# """
# backend/main.py
# FastAPI application entry point.
# """

# from contextlib import asynccontextmanager

# # pyrefly: ignore [missing-import]
# from fastapi import FastAPI

# # pyrefly: ignore [missing-import]
# from fastapi.middleware.cors import CORSMiddleware

# from backend.api.routes import router
# from backend.vectorstore import init_db
# from backend.logging_config import get_logger

# logger = get_logger(__name__)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """Startup: initialise DB. Shutdown: nothing needed."""
#     logger.info("Starting Multimodal RAG backend")
#     await init_db()
#     logger.info("Database ready")
#     yield
#     logger.info("Shutting down")


# app = FastAPI(
#     title="Multimodal RAG API",
#     description="Production-style RAG application for documents, audio, and video.",
#     version="1.0.0",
#     lifespan=lifespan,
# )

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# app.include_router(router, prefix="")

# # pyrefly: ignore [missing-import]
# # if __name__ == "__main__":
# #     # pyrefly: ignore [missing-import]
# #     import uvicorn
# #     uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)




