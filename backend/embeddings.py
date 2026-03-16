"""
Embeddings — wraps SentenceTransformers for document and query embedding.

Provides:
  • EmbeddingModel — singleton wrapper around a SentenceTransformer model.
  • LangChain-compatible Embeddings class for use with ChromaDB.
"""

import logging
from functools import lru_cache
from typing import List

from langchain_core.embeddings import Embeddings

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class EmbeddingModel:
    """
    Thin wrapper around a SentenceTransformer model.
    The model is loaded once and reused (singleton via get_embedding_model()).
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        logger.info("Loading embedding model: %s", model_name)
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.model_name = model_name
            self.dimension = self._model.get_sentence_embedding_dimension()
            logger.info(
                "Embedding model loaded — dimension: %d", self.dimension
            )
        except ImportError:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            )

    def embed_text(self, text: str) -> List[float]:
        """Embed a single string."""
        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def embed_batch(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Embed a list of strings in batches."""
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 100,
        )
        return [v.tolist() for v in vectors]


@lru_cache(maxsize=1)
def get_embedding_model() -> EmbeddingModel:
    """Return the global singleton EmbeddingModel."""
    return EmbeddingModel()


# ─────────────────────────────────────────────────────────────────────────────
# LangChain-compatible Embeddings class
# ─────────────────────────────────────────────────────────────────────────────

class SentenceTransformerEmbeddings(Embeddings):
    """
    LangChain Embeddings implementation backed by SentenceTransformers.
    Pass an instance of this class directly to Chroma() or other vector stores.
    """

    def __init__(self, model_name: str = settings.EMBEDDING_MODEL):
        self._model = EmbeddingModel(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.embed_batch(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_text(text)


@lru_cache(maxsize=1)
def get_langchain_embeddings() -> SentenceTransformerEmbeddings:
    """Return the singleton LangChain-compatible embeddings instance."""
    return SentenceTransformerEmbeddings()
