"""
Enterprise RAG Knowledge Assistant — FastAPI Backend

Endpoints:
  POST   /upload               Upload and ingest a document
  POST   /ask                  Ask a question (non-streaming)
  POST   /ask/stream           Ask a question (SSE streaming)
  GET    /documents            List all ingested documents
  DELETE /documents/{doc_id}   Remove a document from the knowledge base
  GET    /health               Health check
"""

import logging
import os
import shutil
import time
import uuid
from pathlib import Path
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from rag_pipeline import delete_document, ingest_document, list_documents, query, query_stream

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Production-ready RAG system with query expansion, hybrid retrieval, "
        "re-ranking, context compression, and citation generation."
    ),
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ──────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    doc_ids: Optional[List[str]] = Field(
        default=None,
        description="Restrict search to specific document IDs. Leave empty for all docs.",
    )
    stream: bool = Field(default=False)


class AskResponse(BaseModel):
    answer: str
    citations: List[dict]
    context_used: int
    elapsed_ms: int


class UploadResponse(BaseModel):
    doc_id: str
    filename: str
    chunks_stored: int
    chunk_stats: dict
    status: str


class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    extension: str
    total_chunks: int


class DeleteResponse(BaseModel):
    doc_id: str
    chunks_deleted: int
    status: str


# ── Helpers ────────────────────────────────────────────────────────────────────

def _validate_file(file: UploadFile) -> None:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )


def _save_upload(file: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "upload").suffix.lower()
    unique_name = f"{uuid.uuid4().hex}{suffix}"
    dest_path = dest_dir / unique_name
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return dest_path


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
def health_check():
    """Simple health probe."""
    from vector_store import get_vector_store
    store = get_vector_store()
    return {
        "status": "ok",
        "version": settings.APP_VERSION,
        "llm_provider": settings.LLM_PROVIDER,
        "embedding_model": settings.EMBEDDING_MODEL,
        "total_chunks_indexed": store.count(),
    }


@app.post("/upload", response_model=UploadResponse, tags=["Documents"])
async def upload_document(
    file: UploadFile = File(...),
    doc_id: Optional[str] = Form(default=None),
):
    """
    Upload a PDF, DOCX, TXT, or Markdown file.
    The document is parsed, chunked, embedded, and stored in ChromaDB.
    """
    _validate_file(file)

    # Check file size
    file.file.seek(0, 2)
    size_mb = file.file.tell() / (1024 * 1024)
    file.file.seek(0)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_FILE_SIZE_MB} MB",
        )

    saved_path = _save_upload(file, settings.UPLOAD_DIR)

    try:
        result = ingest_document(saved_path, doc_id=doc_id)
    except Exception as exc:
        saved_path.unlink(missing_ok=True)
        logger.error("Ingestion failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}")

    return UploadResponse(**result.to_dict())


@app.post("/ask", response_model=AskResponse, tags=["Query"])
async def ask_question(req: AskRequest):
    """
    Ask a question. The system retrieves relevant document chunks and
    generates a grounded answer with citations.
    """
    start = time.perf_counter()
    try:
        result = await query(req.question, filter_doc_ids=req.doc_ids)
    except Exception as exc:
        logger.error("Query failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return AskResponse(
        answer=result["answer"],
        citations=result["citations"],
        context_used=result["context_used"],
        elapsed_ms=elapsed_ms,
    )


@app.post("/ask/stream", tags=["Query"])
async def ask_question_stream(req: AskRequest):
    """
    Ask a question with Server-Sent Events streaming.
    Each event is a JSON object with a 'type' field:
      - type="token"     → streaming answer token
      - type="citations" → citation list (sent after the full answer)
      - type="error"     → error message
    Ends with "data: [DONE]"
    """
    return StreamingResponse(
        query_stream(req.question, filter_doc_ids=req.doc_ids),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/documents", response_model=List[DocumentInfo], tags=["Documents"])
def get_documents():
    """List all documents currently in the knowledge base."""
    try:
        return list_documents()
    except Exception as exc:
        logger.error("list_documents failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/documents/{doc_id}", response_model=DeleteResponse, tags=["Documents"])
def remove_document(doc_id: str):
    """Remove a document and all its chunks from the knowledge base."""
    try:
        result = delete_document(doc_id)
        if result["chunks_deleted"] == 0:
            raise HTTPException(
                status_code=404, detail=f"Document '{doc_id}' not found."
            )
        return DeleteResponse(status="deleted", **result)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("delete_document failed: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# ── Dev runner ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
