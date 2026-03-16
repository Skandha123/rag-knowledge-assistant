"""
Retriever — implements the full advanced RAG retrieval pipeline:

  1. Query Expansion   — rewrite the query into N sub-queries.
  2. Hybrid Retrieval  — vector similarity + BM25 keyword matching.
  3. De-duplication    — merge results from multiple sub-queries.
  4. Re-ranking        — cross-encoder style scoring to select best N chunks.
  5. Context Compression — trim each chunk to relevant sentences.
"""

import logging
import re
from typing import List, Optional, Tuple

from langchain_core.documents import Document

from config import settings
from vector_store import get_vector_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Query Expansion
# ─────────────────────────────────────────────────────────────────────────────

def expand_query(query: str, llm_fn) -> List[str]:
    """
    Use the LLM to generate multiple search queries from the original query.
    Falls back to the original query if expansion fails.

    Args:
        query:  Original user query.
        llm_fn: Callable(prompt: str) -> str  (synchronous).

    Returns:
        List of query strings (original + expanded).
    """
    if not settings.QUERY_EXPANSION_ENABLED:
        return [query]

    prompt = (
        f"You are an expert at reformulating search queries to improve document retrieval.\n"
        f"Generate {settings.QUERY_EXPANSION_COUNT} alternative search queries for the "
        f"following question. Each alternative should capture a different angle or "
        f"phrasing. Return ONLY the queries, one per line, no numbering.\n\n"
        f"Original question: {query}\n\nAlternative queries:"
    )

    try:
        raw = llm_fn(prompt).strip()
        expanded = [q.strip() for q in raw.splitlines() if q.strip()]
        # Keep orignal first, de-duplicate
        queries = [query] + [q for q in expanded if q.lower() != query.lower()]
        logger.info("Query expanded into %d variants.", len(queries))
        return queries[: settings.QUERY_EXPANSION_COUNT + 1]
    except Exception as exc:
        logger.warning("Query expansion failed (%s); using original query.", exc)
        return [query]


# ─────────────────────────────────────────────────────────────────────────────
# 2. BM25 Keyword Scorer (lightweight, no index needed)
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def bm25_score(query: str, doc_text: str, k1: float = 1.5, b: float = 0.75) -> float:
    """Simple BM25-like TF-IDF scoring (no corpus stats needed for re-ranking)."""
    query_tokens = set(_tokenize(query))
    doc_tokens = _tokenize(doc_text)
    doc_len = len(doc_tokens)
    avg_len = 200.0  # assumed average chunk length

    tf: dict[str, int] = {}
    for t in doc_tokens:
        tf[t] = tf.get(t, 0) + 1

    score = 0.0
    for token in query_tokens:
        if token not in tf:
            continue
        freq = tf[token]
        numerator = freq * (k1 + 1)
        denominator = freq + k1 * (1 - b + b * doc_len / avg_len)
        score += numerator / denominator
    return score


# ─────────────────────────────────────────────────────────────────────────────
# 3. Hybrid Retrieval
# ─────────────────────────────────────────────────────────────────────────────

def hybrid_retrieve(
    queries: List[str],
    top_k: int = settings.RETRIEVAL_TOP_K,
    filter_doc_ids: Optional[List[str]] = None,
) -> List[Tuple[Document, float]]:
    """
    For each expanded query, perform vector similarity search then combine
    with BM25 scores using a weighted sum.

    Returns:
        Deduplicated list of (Document, combined_score) sorted desc.
    """
    store = get_vector_store()
    seen: dict[str, Tuple[Document, float]] = {}

    for query in queries:
        vector_results = store.similarity_search(
            query, top_k=top_k, filter_doc_ids=filter_doc_ids
        )
        for doc, vec_score in vector_results:
            bm25 = bm25_score(query, doc.page_content)
            # Normalise BM25 to [0,1] range with a soft cap
            bm25_norm = min(bm25 / 10.0, 1.0)
            combined = (
                settings.VECTOR_WEIGHT * vec_score
                + settings.KEYWORD_WEIGHT * bm25_norm
            )
            chunk_id = doc.metadata.get("chunk_id", doc.page_content[:32])
            if chunk_id not in seen or combined > seen[chunk_id][1]:
                seen[chunk_id] = (doc, round(combined, 4))

    results = sorted(seen.values(), key=lambda x: x[1], reverse=True)
    return results[:top_k]


# ─────────────────────────────────────────────────────────────────────────────
# 4. Re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def rerank(
    query: str,
    candidates: List[Tuple[Document, float]],
    top_n: int = settings.RERANK_TOP_N,
) -> List[Tuple[Document, float]]:
    """
    Re-rank candidates using a cross-attention-inspired scoring function
    (without requiring a heavy cross-encoder model by default).

    The score blends:
      • Original retrieval score (50 %)
      • BM25 score against the original query (30 %)
      • Chunk length penalty — prefer medium-length chunks (20 %)
    """
    rescored: List[Tuple[Document, float]] = []
    for doc, base_score in candidates:
        bm25 = min(bm25_score(query, doc.page_content) / 10.0, 1.0)
        length = len(doc.page_content)
        # Gaussian length preference around CHUNK_SIZE characters
        ideal = settings.CHUNK_SIZE
        length_score = max(0.0, 1.0 - abs(length - ideal) / ideal)
        final = 0.5 * base_score + 0.3 * bm25 + 0.2 * length_score
        rescored.append((doc, round(final, 4)))

    rescored.sort(key=lambda x: x[1], reverse=True)
    logger.info(
        "Re-ranked %d candidates → selected top %d.",
        len(rescored),
        min(top_n, len(rescored)),
    )
    return rescored[:top_n]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Context Compression
# ─────────────────────────────────────────────────────────────────────────────

def compress_context(
    query: str,
    chunks: List[Tuple[Document, float]],
    max_chars: int = settings.MAX_CONTEXT_TOKENS * 4,  # ~4 chars per token
) -> List[Tuple[Document, float]]:
    """
    Trim each chunk to the most relevant sentences and enforce a global
    character budget to keep the LLM context manageable.
    """
    if not settings.CONTEXT_COMPRESSION_ENABLED:
        return chunks

    query_tokens = set(_tokenize(query))
    compressed: List[Tuple[Document, float]] = []
    total_chars = 0

    for doc, score in chunks:
        if total_chars >= max_chars:
            break
        sentences = re.split(r"(?<=[.!?])\s+", doc.page_content)
        ranked_sents = sorted(
            sentences,
            key=lambda s: sum(1 for t in _tokenize(s) if t in query_tokens),
            reverse=True,
        )
        # Keep top sentences up to remaining budget
        remaining = max_chars - total_chars
        selected: List[str] = []
        for sent in ranked_sents:
            if sum(len(s) for s in selected) + len(sent) > remaining:
                break
            selected.append(sent)

        if not selected:
            selected = [sentences[0]] if sentences else [doc.page_content[:300]]

        # Re-order selected sentences in their original order
        original_order = {s: i for i, s in enumerate(sentences)}
        selected.sort(key=lambda s: original_order.get(s, 999))

        compressed_text = " ".join(selected)
        total_chars += len(compressed_text)
        new_doc = Document(
            page_content=compressed_text,
            metadata={**doc.metadata, "compressed": True},
        )
        compressed.append((new_doc, score))

    return compressed


# ─────────────────────────────────────────────────────────────────────────────
# Public pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    llm_fn=None,
    filter_doc_ids: Optional[List[str]] = None,
) -> List[Tuple[Document, float]]:
    """
    Full retrieval pipeline:
        expand → hybrid retrieve → re-rank → compress

    Args:
        query:          User's original question.
        llm_fn:         Optional callable for query expansion.
        filter_doc_ids: Optional list of doc IDs to restrict search.

    Returns:
        List of (Document, score) tuples — best chunks for the LLM context.
    """
    # Step 1 — Query expansion
    queries = expand_query(query, llm_fn) if llm_fn else [query]

    # Step 2 — Hybrid retrieval across all expanded queries
    candidates = hybrid_retrieve(
        queries,
        top_k=settings.RETRIEVAL_TOP_K,
        filter_doc_ids=filter_doc_ids,
    )
    if not candidates:
        logger.warning("No candidates found for query: %s", query)
        return []

    # Step 3 — Re-rank
    reranked = rerank(query, candidates, top_n=settings.RERANK_TOP_N)

    # Step 4 — Context compression
    final = compress_context(query, reranked)

    logger.info(
        "Retrieval pipeline: %d expanded queries → %d candidates → %d final chunks.",
        len(queries),
        len(candidates),
        len(final),
    )
    return final
