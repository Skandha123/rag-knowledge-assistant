"""
Vector Store — ChromaDB-backed storage for document embeddings.

Provides:
  • VectorStore — manage collections, add / delete / query chunks.
  • get_vector_store() — singleton factory.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document

from config import settings
from embeddings import get_embedding_model

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Persistent ChromaDB vector store with helper methods for
    adding, querying, and deleting document chunks.
    """

    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=settings.CHROMA_DB_PATH,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        self._embedder = get_embedding_model()
        logger.info(
            "ChromaDB ready — collection '%s' has %d documents.",
            settings.CHROMA_COLLECTION_NAME,
            self._collection.count(),
        )

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_chunks(self, chunks: List[Document]) -> List[str]:
        """
        Embed and store a list of chunks.

        Returns:
            List of stored chunk IDs.
        """
        if not chunks:
            return []

        texts = [c.page_content for c in chunks]
        embeddings = self._embedder.embed_batch(texts)
        ids = [c.metadata.get("chunk_id") or str(uuid.uuid4()) for c in chunks]
        metadatas = [self._serialise_meta(c.metadata) for c in chunks]

        # ChromaDB upserts in a single call
        self._collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        logger.info("Stored %d chunks in ChromaDB.", len(chunks))
        return ids

    # ── Query ──────────────────────────────────────────────────────────────────

    def similarity_search(
        self,
        query: str,
        top_k: int = settings.RETRIEVAL_TOP_K,
        filter_doc_ids: Optional[List[str]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Cosine similarity search.

        Returns:
            List of (Document, score) tuples, sorted by descending score.
        """
        query_embedding = self._embedder.embed_text(query)

        where_clause = None
        if filter_doc_ids:
            where_clause = {"doc_id": {"$in": filter_doc_ids}}

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, max(self._collection.count(), 1)),
            where=where_clause,
            include=["documents", "metadatas", "distances"],
        )

        docs_and_scores: List[Tuple[Document, float]] = []
        for text, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = 1.0 - dist  # cosine distance → similarity
            if score < settings.SIMILARITY_THRESHOLD:
                continue
            docs_and_scores.append(
                (Document(page_content=text, metadata=meta), round(score, 4))
            )

        docs_and_scores.sort(key=lambda x: x[1], reverse=True)
        return docs_and_scores

    def get_all_documents_meta(self) -> List[Dict[str, Any]]:
        """
        Return deduplicated document-level metadata (one entry per doc_id).
        """
        if self._collection.count() == 0:
            return []

        results = self._collection.get(include=["metadatas"])
        seen: Dict[str, Dict] = {}
        for meta in results["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            if doc_id not in seen:
                seen[doc_id] = {
                    "doc_id": doc_id,
                    "filename": meta.get("filename", "unknown"),
                    "extension": meta.get("extension", ""),
                    "total_chunks": meta.get("total_chunks", 0),
                    "total_sections": meta.get("total_sections", 0),
                }
        return list(seen.values())

    # ── Delete ─────────────────────────────────────────────────────────────────

    def delete_document(self, doc_id: str) -> int:
        """
        Remove all chunks belonging to doc_id.

        Returns:
            Number of chunks deleted.
        """
        results = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids_to_delete = results["ids"]
        if ids_to_delete:
            self._collection.delete(ids=ids_to_delete)
            logger.info("Deleted %d chunks for doc_id='%s'.", len(ids_to_delete), doc_id)
        return len(ids_to_delete)

    def count(self) -> int:
        return self._collection.count()

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _serialise_meta(meta: Dict[str, Any]) -> Dict[str, Any]:
        """ChromaDB only accepts str / int / float / bool values in metadata."""
        clean: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif v is None:
                clean[k] = ""
            else:
                clean[k] = str(v)
        return clean


# ─────────────────────────────────────────────────────────────────────────────
# Singleton
# ─────────────────────────────────────────────────────────────────────────────

_vector_store: Optional[VectorStore] = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
