"""
Chunking — splits LangChain Documents into overlapping text segments
ready for embedding and storage.

Strategy:
  1. Recursive character splitting (respects paragraph/sentence boundaries).
  2. Each chunk inherits parent document metadata plus its own chunk_id,
     chunk_index, and char_offset.
"""

import hashlib
import logging
import re
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

_SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    length_function=len,
    is_separator_regex=False,
)


def _chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    """Deterministic, short ID for a chunk."""
    digest = hashlib.md5(f"{doc_id}:{chunk_index}:{text[:64]}".encode()).hexdigest()[:10]
    return f"{doc_id[:8]}_{chunk_index:04d}_{digest}"


def _clean_chunk(text: str) -> str:
    """Remove leading/trailing noise while preserving internal structure."""
    text = re.sub(r"^\s*[-–—•*]\s*", "", text)   # strip leading bullet
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Split a list of Documents into smaller, overlapping chunks.

    Each output chunk's metadata includes:
        chunk_id         — globally unique identifier
        chunk_index      — position within the parent document
        total_chunks     — total chunks derived from the same source
        char_count       — number of characters in the chunk
        word_count       — approximate word count
        ... (all parent metadata is preserved)

    Args:
        documents: Raw documents from document_loader.load_document().

    Returns:
        List of chunked Document objects.
    """
    if not documents:
        return []

    # Split each section independently so metadata is preserved cleanly.
    all_chunks: List[Document] = []
    for doc in documents:
        sub_chunks = _SPLITTER.split_documents([doc])
        for chunk in sub_chunks:
            text = _clean_chunk(chunk.page_content)
            if len(text) < settings.MIN_CHUNK_LENGTH:
                logger.debug("Skipping short chunk (%d chars).", len(text))
                continue
            all_chunks.append(Document(page_content=text, metadata=chunk.metadata.copy()))

    # Assign chunk IDs and counts within each doc_id group.
    doc_id_to_chunks: dict[str, List[int]] = {}
    for i, c in enumerate(all_chunks):
        did = c.metadata.get("doc_id", "unknown")
        doc_id_to_chunks.setdefault(did, []).append(i)

    for did, indices in doc_id_to_chunks.items():
        total = len(indices)
        for local_idx, global_idx in enumerate(indices):
            chunk = all_chunks[global_idx]
            text = chunk.page_content
            chunk.metadata.update(
                {
                    "chunk_id": _chunk_id(did, local_idx, text),
                    "chunk_index": local_idx,
                    "total_chunks": total,
                    "char_count": len(text),
                    "word_count": len(text.split()),
                }
            )

    logger.info(
        "Chunked %d document section(s) into %d chunks.",
        len(documents),
        len(all_chunks),
    )
    return all_chunks


def get_chunk_stats(chunks: List[Document]) -> dict:
    """Return summary statistics for a list of chunks."""
    if not chunks:
        return {}
    sizes = [c.metadata.get("char_count", len(c.page_content)) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "avg_chars": round(sum(sizes) / len(sizes), 1),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
    }
