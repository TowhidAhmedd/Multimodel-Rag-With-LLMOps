"""
backend/main.py
FastAPI application entry point.
"""

from contextlib import asynccontextmanager

# pyrefly: ignore [missing-import]
from fastapi import FastAPI

# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import router
from backend.vectorstore import init_db
from backend.logging_config import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: initialise DB. Shutdown: nothing needed."""
    logger.info("Starting Multimodal RAG backend")
    await init_db()
    logger.info("Database ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Multimodal RAG API",
    description="Production-style RAG application for documents, audio, and video.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="")

# pyrefly: ignore [missing-import]
# if __name__ == "__main__":
#     # pyrefly: ignore [missing-import]
#     import uvicorn
#     uvicorn.run("backend.main:app", host="0.0.0.0", port=8000, reload=True)




