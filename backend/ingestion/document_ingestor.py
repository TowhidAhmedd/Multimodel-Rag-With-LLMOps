"""
backend/ingestion/document_ingestor.py
PDF, DOCX, TXT ingestion with lazy imports for heavy parsing libraries.
"""

from pathlib import Path
from typing import List

from backend.chunking import Chunk, chunk_text
from backend.logging_config import get_logger

logger = get_logger(__name__)


def ingest_pdf(file_path: str) -> List[Chunk]:
    """Extract text page-by-page from a PDF and produce overlapping chunks."""
    import pdfplumber  # noqa: PLC0415

    source_file = Path(file_path).name
    chunks: List[Chunk] = []

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if not text.strip():
                continue
            page_chunks = chunk_text(
                text=text,
                source_file=source_file,
                modality="text",
                page_number=page_num,
                metadata={"file_type": "pdf", "total_pages": total_pages},
            )
            chunks.extend(page_chunks)

    logger.info("PDF ingested", extra={"source_file": source_file, "chunks": len(chunks)})
    return chunks


def ingest_docx(file_path: str) -> List[Chunk]:
    """Extract text from a DOCX file and produce overlapping chunks."""
    import docx as python_docx  # noqa: PLC0415

    source_file = Path(file_path).name
    doc = python_docx.Document(file_path)
    full_text = "\n".join(
        p.text for p in doc.paragraphs if p.text.strip()
    )
    chunks = chunk_text(
        text=full_text,
        source_file=source_file,
        modality="text",
        metadata={"file_type": "docx"},
    )
    logger.info("DOCX ingested", extra={"source_file": source_file, "chunks": len(chunks)})
    return chunks


def ingest_txt(file_path: str) -> List[Chunk]:
    """Read a plain-text file and produce overlapping chunks."""
    source_file = Path(file_path).name
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        full_text = f.read()
    chunks = chunk_text(
        text=full_text,
        source_file=source_file,
        modality="text",
        metadata={"file_type": "txt"},
    )
    logger.info("TXT ingested", extra={"source_file": source_file, "chunks": len(chunks)})
    return chunks


def ingest_document(file_path: str) -> List[Chunk]:
    """Dispatch to the correct ingestor based on file extension."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return ingest_pdf(file_path)
    elif ext == ".docx":
        return ingest_docx(file_path)
    elif ext == ".txt":
        return ingest_txt(file_path)
    raise ValueError(f"Unsupported document format: {ext}")
