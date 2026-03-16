# Enterprise RAG Knowledge Assistant

A production-ready Retrieval Augmented Generation (RAG) system that lets you upload documents and ask questions about them using advanced AI retrieval techniques.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        React Frontend (Vite)                        │
│  Sidebar: Upload/Manage Docs  │  Chat: Streaming Answers + Citations │
└────────────────────────┬────────────────────────────────────────────┘
                         │ HTTP / SSE
┌────────────────────────▼────────────────────────────────────────────┐
│                      FastAPI Backend                                 │
│                                                                     │
│  POST /upload          POST /ask/stream      GET /documents         │
│       │                       │                                     │
│  ┌────▼──────────┐    ┌───────▼──────────────────────────────────┐  │
│  │ Ingestion     │    │ RAG Pipeline                              │  │
│  │ Pipeline      │    │                                           │  │
│  │               │    │  1. Query Expansion (LLM)                 │  │
│  │ load_document │    │     └─ Rewrite query into N variants      │  │
│  │ chunk_text    │    │                                           │  │
│  │ embed_chunks  │    │  2. Hybrid Retrieval                      │  │
│  │ store_vectors │    │     ├─ Vector Similarity (ChromaDB)       │  │
│  └───────────────┘    │     └─ BM25 Keyword Matching             │  │
│                       │                                           │  │
│                       │  3. Re-ranking (top 15 → top 5)          │  │
│                       │                                           │  │
│                       │  4. Context Compression                   │  │
│                       │     └─ Keep relevant sentences only       │  │
│                       │                                           │  │
│                       │  5. LLM Generation (OpenAI / Ollama)     │  │
│                       │     └─ Grounded answer + citations        │  │
│                       └───────────────────────────────────────────┘  │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
          ┌─────────────────────▼──────────────────────┐
          │           ChromaDB (local persistent)       │
          │     Cosine similarity · HNSW index          │
          └────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Python 3.11 · FastAPI · Uvicorn |
| **AI / RAG** | LangChain · OpenAI GPT-4o-mini (or Ollama) |
| **Embeddings** | SentenceTransformers `all-MiniLM-L6-v2` |
| **Vector DB** | ChromaDB (persistent local storage) |
| **Document Parsing** | pypdf · python-docx |
| **Frontend** | React 18 · Vite · Tailwind CSS |
| **Containerisation** | Docker · Docker Compose · Nginx |

---

## Advanced RAG Features

### 1. Query Expansion
Before retrieval, the user's query is rewritten into 3 alternative formulations by the LLM. All variants are used for retrieval and deduplicated, dramatically improving recall for ambiguous or short queries.

### 2. Hybrid Retrieval
Each query is searched using both:
- **Vector similarity** — cosine distance in ChromaDB (weight: 70%)
- **BM25 keyword matching** — TF-IDF style scoring (weight: 30%)

The scores are combined into a single ranked list.

### 3. Re-ranking
The top 15 candidates are re-scored using a blended function:
- Base retrieval score (50%)
- BM25 against the *original* query (30%)
- Chunk length optimality (20%)

Only the top 5 chunks survive to the generation step.

### 4. Context Compression
Each selected chunk is trimmed to only the sentences most relevant to the query, staying within a token budget. This reduces noise in the LLM context and lowers costs.

### 5. Citation Generation
Every answer includes structured citations: filename, page number, chunk index, relevance score, and an excerpt from the source passage.

---

## Setup

### Prerequisites
- Python 3.11+
- Node.js 18+
- An OpenAI API key **or** a local [Ollama](https://ollama.com) installation

### 1. Clone and configure

```bash
git clone <repo-url>
cd rag-knowledge-assistant
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY (or switch LLM_PROVIDER=ollama)
```

### 2. Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI will be available at `http://localhost:5173`

---

## Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Frontend: http://localhost:3000
# Backend:  http://localhost:8000
```

Data (ChromaDB + uploads) is persisted in `./data/` on the host.

---

## API Reference

### `POST /upload`
Upload a document for ingestion.

**Request:** `multipart/form-data` with `file` field (PDF / DOCX / TXT / MD)

**Response:**
```json
{
  "doc_id": "abc123",
  "filename": "report.pdf",
  "chunks_stored": 42,
  "chunk_stats": { "total_chunks": 42, "avg_chars": 487 },
  "status": "success"
}
```

---

### `POST /ask`
Ask a question (non-streaming).

**Request:**
```json
{
  "question": "What are the key findings?",
  "doc_ids": ["abc123"]   // optional: filter to specific docs
}
```

**Response:**
```json
{
  "answer": "The key findings are...",
  "citations": [
    {
      "doc_id": "abc123",
      "filename": "report.pdf",
      "page": 3,
      "relevance_score": 0.87,
      "excerpt": "The analysis found that..."
    }
  ],
  "context_used": 5,
  "elapsed_ms": 1240
}
```

---

### `POST /ask/stream`
Ask a question with SSE streaming.

**Events:**
```
data: {"type": "token",     "content": "The "}
data: {"type": "token",     "content": "key "}
data: {"type": "citations", "content": [...]}
data: [DONE]
```

---

### `GET /documents`
List all ingested documents.

---

### `DELETE /documents/{doc_id}`
Remove a document from the knowledge base.

---

### `GET /health`
System health check including chunk count and model info.

---

## Configuration

All settings live in `.env` (see `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_PROVIDER` | `openai` | `openai` or `ollama` |
| `OPENAI_API_KEY` | — | Required for OpenAI |
| `OPENAI_MODEL` | `gpt-4o-mini` | OpenAI model name |
| `OLLAMA_MODEL` | `llama3` | Ollama model name |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | SentenceTransformers model |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | `64` | Overlap between chunks |
| `RETRIEVAL_TOP_K` | `15` | Candidates before re-ranking |
| `RERANK_TOP_N` | `5` | Chunks sent to LLM |
| `QUERY_EXPANSION_ENABLED` | `true` | Enable query expansion |
| `CONTEXT_COMPRESSION_ENABLED` | `true` | Enable context compression |

---

## Example Queries

Once you've uploaded a document, try:

```
"Summarise the executive summary"
"What are the main risks identified?"
"List all action items or recommendations"
"What methodology was used?"
"Compare the results in section 3 with section 5"
"What conclusions does the author draw?"
```

---

## Project Structure

```
rag-knowledge-assistant/
├── backend/
│   ├── main.py            # FastAPI app + all endpoints
│   ├── rag_pipeline.py    # Top-level pipeline orchestrator
│   ├── document_loader.py # PDF / DOCX / TXT loading
│   ├── chunking.py        # Text splitting with overlap
│   ├── embeddings.py      # SentenceTransformers wrapper
│   ├── vector_store.py    # ChromaDB management
│   ├── retriever.py       # Hybrid retrieval + re-ranking
│   ├── llm_generator.py   # LLM generation + streaming
│   └── config.py          # Centralised settings
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── api/client.js  # All API calls
│   │   ├── hooks/         # useDocuments, useChat
│   │   └── components/    # Sidebar, ChatWindow, ChatMessage, …
│   ├── package.json
│   └── vite.config.js
├── docker/
│   ├── Dockerfile.backend
│   ├── Dockerfile.frontend
│   └── nginx.conf
├── data/                  # Gitignored — ChromaDB + uploads
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Using Ollama (free, local)

```bash
# Install Ollama: https://ollama.com
ollama pull llama3

# In .env:
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3
```

No API key required. Works fully offline.

---

## License

MIT
