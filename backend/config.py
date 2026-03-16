"""
Centralized configuration for the RAG Knowledge Assistant.
All settings are loaded from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = "Enterprise RAG Knowledge Assistant"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # ── API Keys ──────────────────────────────────────────────────────────────
    OPENAI_API_KEY: Optional[str] = None

    # ── LLM Settings ──────────────────────────────────────────────────────────
    LLM_PROVIDER: str = "openai"          # "openai" | "ollama"
    OPENAI_MODEL: str = "gpt-4o-mini"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    STREAMING_ENABLED: bool = True

    # ── Embedding Settings ─────────────────────────────────────────────────────
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"   # SentenceTransformers model
    EMBEDDING_DIMENSION: int = 384

    # ── Chunking Settings ─────────────────────────────────────────────────────
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64
    MIN_CHUNK_LENGTH: int = 50

    # ── Retrieval Settings ─────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 15            # Initial retrieval count before re-ranking
    RERANK_TOP_N: int = 5               # Final chunks after re-ranking
    SIMILARITY_THRESHOLD: float = 0.3   # Min similarity score to include a chunk
    KEYWORD_WEIGHT: float = 0.3         # Weight for BM25 in hybrid retrieval
    VECTOR_WEIGHT: float = 0.7          # Weight for vector similarity in hybrid retrieval

    # ── Query Expansion ────────────────────────────────────────────────────────
    QUERY_EXPANSION_ENABLED: bool = True
    QUERY_EXPANSION_COUNT: int = 3      # Number of expanded queries to generate

    # ── Context Compression ────────────────────────────────────────────────────
    CONTEXT_COMPRESSION_ENABLED: bool = True
    MAX_CONTEXT_TOKENS: int = 3000

    # ── Storage Paths ──────────────────────────────────────────────────────────
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CHROMA_DB_PATH: str = str(BASE_DIR / "data" / "chroma_db")
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    CHROMA_COLLECTION_NAME: str = "rag_knowledge_base"

    # ── CORS ───────────────────────────────────────────────────────────────────
    ALLOWED_ORIGINS: list = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # ── File Upload ────────────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = [".pdf", ".docx", ".txt", ".md"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"

    def setup_dirs(self):
        """Ensure all required directories exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        Path(self.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.setup_dirs()
