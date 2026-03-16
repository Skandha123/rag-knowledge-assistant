"""
LLM Generator — builds the answer from retrieved context.

Supports:
  • OpenAI (gpt-4o-mini, gpt-4o, …)
  • Ollama (local open-source models)
  • Streaming and non-streaming responses
  • Source citation extraction
"""

import json
import logging
from typing import AsyncIterator, List, Optional, Tuple

from langchain_core.documents import Document

from config import settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert knowledge assistant. Answer questions accurately and concisely based ONLY on the provided document context. 

Rules:
- Only use information from the provided context.
- If the context doesn't contain enough information, say so clearly.
- Always cite the source document(s) you used to answer.
- Be precise, structured, and helpful.
- Format your response with clear paragraphs.
- At the end of your answer, include a "Sources:" section listing the documents used."""


def _build_context_block(chunks: List[Tuple[Document, float]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts = []
    for i, (doc, score) in enumerate(chunks, 1):
        filename = doc.metadata.get("filename", "Unknown")
        page = doc.metadata.get("page", "")
        chunk_idx = doc.metadata.get("chunk_index", "")
        page_info = f" | Page {page}" if page else ""
        chunk_info = f" | Chunk {chunk_idx}" if chunk_idx != "" else ""
        header = f"[{i}] {filename}{page_info}{chunk_info} (relevance: {score:.2f})"
        parts.append(f"{header}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def _build_messages(query: str, context: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context from documents:\n\n{context}\n\n"
                f"---\n\nQuestion: {query}\n\nAnswer:"
            ),
        },
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Citation extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_citations(chunks: List[Tuple[Document, float]]) -> List[dict]:
    """Return structured citation info from the retrieved chunks."""
    seen = set()
    citations = []
    for doc, score in chunks:
        meta = doc.metadata
        doc_id = meta.get("doc_id", "unknown")
        if doc_id in seen:
            continue
        seen.add(doc_id)
        citations.append(
            {
                "doc_id": doc_id,
                "filename": meta.get("filename", "Unknown"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "relevance_score": score,
                "excerpt": doc.page_content[:200] + ("…" if len(doc.page_content) > 200 else ""),
            }
        )
    return citations


# ─────────────────────────────────────────────────────────────────────────────
# LLM client factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_openai_client():
    try:
        from openai import OpenAI, AsyncOpenAI
        sync_client = OpenAI(api_key=settings.OPENAI_API_KEY)
        async_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        return sync_client, async_client
    except ImportError:
        raise ImportError("openai package is required: pip install openai")


# ─────────────────────────────────────────────────────────────────────────────
# Synchronous generation (used for query expansion)
# ─────────────────────────────────────────────────────────────────────────────

def generate_sync(prompt: str) -> str:
    """
    Simple synchronous LLM call used internally (e.g., query expansion).
    """
    if settings.LLM_PROVIDER == "openai":
        sync_client, _ = _get_openai_client()
        response = sync_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=256,
        )
        return response.choices[0].message.content or ""

    elif settings.LLM_PROVIDER == "ollama":
        import httpx
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")


# ─────────────────────────────────────────────────────────────────────────────
# Async streaming generation
# ─────────────────────────────────────────────────────────────────────────────

async def generate_stream(
    query: str,
    chunks: List[Tuple[Document, float]],
) -> AsyncIterator[str]:
    """
    Stream the LLM answer token-by-token as Server-Sent Events (SSE) lines.

    Yields:
        SSE-formatted strings: "data: <json>\n\n"
        Final event:           "data: [DONE]\n\n"
    """
    if not chunks:
        no_context_msg = (
            "I couldn't find relevant information in the uploaded documents "
            "to answer your question. Please try rephrasing or upload more documents."
        )
        yield f"data: {json.dumps({'type': 'token', 'content': no_context_msg})}\n\n"
        yield "data: [DONE]\n\n"
        return

    context = _build_context_block(chunks)
    messages = _build_messages(query, context)
    citations = extract_citations(chunks)

    try:
        if settings.LLM_PROVIDER == "openai":
            _, async_client = _get_openai_client()
            stream = await async_client.chat.completions.create(
                model=settings.OPENAI_MODEL,
                messages=messages,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.LLM_MAX_TOKENS,
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield f"data: {json.dumps({'type': 'token', 'content': delta})}\n\n"

        elif settings.LLM_PROVIDER == "ollama":
            import httpx
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream(
                    "POST",
                    f"{settings.OLLAMA_BASE_URL}/api/chat",
                    json={
                        "model": settings.OLLAMA_MODEL,
                        "messages": messages,
                        "stream": True,
                        "options": {
                            "temperature": settings.LLM_TEMPERATURE,
                            "num_predict": settings.LLM_MAX_TOKENS,
                        },
                    },
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            data = json.loads(line)
                            content = data.get("message", {}).get("content", "")
                            if content:
                                yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"

        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")

    except Exception as exc:
        logger.error("LLM generation error: %s", exc)
        yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    # Send citations as a final event before [DONE]
    yield f"data: {json.dumps({'type': 'citations', 'content': citations})}\n\n"
    yield "data: [DONE]\n\n"


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming generation (for simpler / batch use)
# ─────────────────────────────────────────────────────────────────────────────

async def generate(
    query: str,
    chunks: List[Tuple[Document, float]],
) -> dict:
    """
    Non-streaming generation — returns the full answer at once.

    Returns:
        {
            "answer":    str,
            "citations": list[dict],
            "context_used": int,
        }
    """
    if not chunks:
        return {
            "answer": (
                "I couldn't find relevant information in the uploaded documents. "
                "Please try rephrasing or upload more documents."
            ),
            "citations": [],
            "context_used": 0,
        }

    context = _build_context_block(chunks)
    messages = _build_messages(query, context)
    citations = extract_citations(chunks)
    answer = ""

    if settings.LLM_PROVIDER == "openai":
        _, async_client = _get_openai_client()
        response = await async_client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            messages=messages,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
        )
        answer = response.choices[0].message.content or ""

    elif settings.LLM_PROVIDER == "ollama":
        import httpx
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{settings.OLLAMA_BASE_URL}/api/chat",
                json={
                    "model": settings.OLLAMA_MODEL,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": settings.LLM_TEMPERATURE},
                },
            )
            resp.raise_for_status()
            answer = resp.json()["message"]["content"]

    return {
        "answer": answer,
        "citations": citations,
        "context_used": len(chunks),
    }
