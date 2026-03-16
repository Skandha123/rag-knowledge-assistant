"""
RAG Pipeline — top-level orchestrator that ties together:
  document loading → chunking → embedding → storage
  and:
  query → retrieval → generation

This module is the single entry point used by main.py.
"""

import logging
import uuid
from pathlib import Path
from typing import AsyncIterator, List, Optional

from langchain_core.documents import Document

from chunking import chunk_documents, get_chunk_stats
from config import settings
from document_loader import load_document
from llm_generator import extract_citations, generate, generate_stream, generate_sync
from retriever import retrieve
from vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Ingestion
# ─────────────────────────────────────────────────────────────────────────────

class IngestResult:
    def __init__(
        self,
        doc_id: str,
        filename: str,
        chunks_stored: int,
        chunk_stats: dict,
    ):
        self.doc_id = doc_id
        self.filename = filename
        self.chunks_stored = chunks_stored
        self.chunk_stats = chunk_stats

    def to_dict(self) -> dict:
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "chunks_stored": self.chunks_stored,
            "chunk_stats": self.chunk_stats,
            "status": "success",
        }


def ingest_document(file_path: str | Path, doc_id: Optional[str] = None) -> IngestResult:
    """
    Full ingestion pipeline for a single document.

    Steps:
        1. Load document (PDF / DOCX / TXT).
        2. Split into chunks.
        3. Embed and store in ChromaDB.

    Args:
        file_path: Path to the uploaded file.
        doc_id:    Optional caller-supplied ID (a UUID is generated otherwise).

    Returns:
        IngestResult with doc_id, filename, chunk count, and stats.
    """
    path = Path(file_path)
    doc_id = doc_id or str(uuid.uuid4())

    logger.info("Ingesting document: %s (doc_id=%s)", path.name, doc_id)

    # Step 1 — Load
    raw_docs: List[Document] = load_document(path, doc_id=doc_id)

    # Step 2 — Chunk
    chunks = chunk_documents(raw_docs)
    stats = get_chunk_stats(chunks)

    # Step 3 — Embed + Store
    store = get_vector_store()
    chunk_ids = store.add_chunks(chunks)

    result = IngestResult(
        doc_id=doc_id,
        filename=path.name,
        chunks_stored=len(chunk_ids),
        chunk_stats=stats,
    )
    logger.info(
        "Ingestion complete: %d chunks stored for '%s'.", len(chunk_ids), path.name
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Query (streaming)
# ─────────────────────────────────────────────────────────────────────────────

async def query_stream(
    question: str,
    filter_doc_ids: Optional[List[str]] = None,
) -> AsyncIterator[str]:
    """
    Full RAG pipeline with streaming output.

    Yields SSE-formatted strings ending with "data: [DONE]\n\n".
    """
    logger.info("Streaming query: %s", question[:80])

    # Retrieve context (uses LLM for query expansion if available)
    try:
        llm_fn = generate_sync if settings.QUERY_EXPANSION_ENABLED else None
        chunks = retrieve(question, llm_fn=llm_fn, filter_doc_ids=filter_doc_ids)
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        chunks = []

    # Stream the answer
    async for chunk in generate_stream(question, chunks):
        yield chunk


# ─────────────────────────────────────────────────────────────────────────────
# Query (non-streaming)
# ─────────────────────────────────────────────────────────────────────────────

async def query(
    question: str,
    filter_doc_ids: Optional[List[str]] = None,
) -> dict:
    """
    Full RAG pipeline — returns a complete answer dict.

    Returns:
        {
            "answer":        str,
            "citations":     list[dict],
            "context_used":  int,
        }
    """
    logger.info("Query: %s", question[:80])

    try:
        llm_fn = generate_sync if settings.QUERY_EXPANSION_ENABLED else None
        chunks = retrieve(question, llm_fn=llm_fn, filter_doc_ids=filter_doc_ids)
    except Exception as exc:
        logger.error("Retrieval error: %s", exc)
        chunks = []

    return await generate(question, chunks)


# ─────────────────────────────────────────────────────────────────────────────
# Document management helpers (thin wrappers over vector store)
# ─────────────────────────────────────────────────────────────────────────────

def list_documents() -> List[dict]:
    """Return metadata for all ingested documents."""
    store = get_vector_store()
    return store.get_all_documents_meta()


def delete_document(doc_id: str) -> dict:
    """Remove all chunks for doc_id from the vector store."""
    store = get_vector_store()
    deleted = store.delete_document(doc_id)
    return {"doc_id": doc_id, "chunks_deleted": deleted}
