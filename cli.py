#!/usr/bin/env python3
"""
RAG Knowledge Assistant — CLI
Usage:
  python cli.py upload <file> [<file> ...]
  python cli.py list
  python cli.py ask "<question>"
  python cli.py delete <doc_id>
  python cli.py stats
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Ensure backend is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent / "backend"))


# ── Helpers ────────────────────────────────────────────────────────────────

BOLD   = "\033[1m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
CYAN   = "\033[36m"
DIM    = "\033[2m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}✓{RESET} {msg}")
def warn(msg):  print(f"{YELLOW}⚠{RESET} {msg}")
def err(msg):   print(f"{RED}✗{RESET} {msg}", file=sys.stderr)
def info(msg):  print(f"{CYAN}ℹ{RESET} {msg}")
def dim(msg):   print(f"{DIM}{msg}{RESET}")


def _bar(pct: int, width: int = 30) -> str:
    filled = int(width * pct / 100)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {pct:3d}%"


# ── Commands ───────────────────────────────────────────────────────────────

def cmd_upload(files: list[str]):
    from rag_pipeline import ingest_document
    print(f"\n{BOLD}Uploading {len(files)} file(s)…{RESET}\n")

    for path_str in files:
        path = Path(path_str)
        if not path.exists():
            err(f"File not found: {path}")
            continue

        print(f"  {BOLD}{path.name}{RESET}", end="  ", flush=True)
        try:
            result = ingest_document(path)
            print(f"{GREEN}✓{RESET} {result.chunks_stored} chunks  "
                  f"{DIM}(doc_id: {result.doc_id[:12]}…){RESET}")
            stats = result.chunk_stats
            dim(f"      avg {stats.get('avg_chars', '?')} chars/chunk  "
                f"min {stats.get('min_chars', '?')}  max {stats.get('max_chars', '?')}")
        except Exception as exc:
            print(f"{RED}✗{RESET} {exc}")

    print()


def cmd_list():
    from rag_pipeline import list_documents
    docs = list_documents()

    if not docs:
        warn("No documents in the knowledge base.")
        return

    print(f"\n{BOLD}Knowledge Base — {len(docs)} document(s){RESET}\n")
    fmt = f"  {{:<36}}  {{:<30}}  {{:>8}}  {{:>5}}"
    print(fmt.format("doc_id", "filename", "chunks", "ext"))
    print("  " + "─" * 85)
    for d in docs:
        print(fmt.format(
            d["doc_id"][:36],
            d["filename"][:30],
            d["total_chunks"],
            d["extension"],
        ))
    print()


def cmd_ask(question: str):
    from rag_pipeline import query

    print(f"\n{BOLD}Question:{RESET} {question}\n")
    print(f"{DIM}{'─' * 60}{RESET}")

    async def _run():
        return await query(question)

    try:
        result = asyncio.run(_run())
    except Exception as exc:
        err(f"Query failed: {exc}")
        return

    print(f"\n{BOLD}Answer:{RESET}")
    print(result["answer"])

    if result["citations"]:
        print(f"\n{BOLD}Sources:{RESET}")
        for c in result["citations"]:
            page_info = f" p.{c['page']}" if c.get("page") else ""
            print(f"  {CYAN}•{RESET} {c['filename']}{page_info}  "
                  f"{DIM}(relevance: {c['relevance_score']:.0%}){RESET}")
    else:
        warn("No citations (no relevant documents found).")

    print(f"\n{DIM}Context chunks used: {result['context_used']}{RESET}\n")


def cmd_delete(doc_id: str):
    from rag_pipeline import delete_document
    result = delete_document(doc_id)
    if result["chunks_deleted"] == 0:
        warn(f"No document found with id: {doc_id}")
    else:
        ok(f"Deleted {result['chunks_deleted']} chunks for doc_id '{doc_id}'")


def cmd_stats():
    from vector_store import get_vector_store
    from config import settings
    store = get_vector_store()
    docs = store.get_all_documents_meta()

    print(f"\n{BOLD}System Statistics{RESET}\n")
    print(f"  LLM provider  : {settings.LLM_PROVIDER}")
    print(f"  Embedding     : {settings.EMBEDDING_MODEL}")
    print(f"  ChromaDB path : {settings.CHROMA_DB_PATH}")
    print(f"  Total chunks  : {store.count()}")
    print(f"  Documents     : {len(docs)}")
    print(f"  Chunk size    : {settings.CHUNK_SIZE} chars  (overlap: {settings.CHUNK_OVERLAP})")
    print(f"  Retrieval     : top-{settings.RETRIEVAL_TOP_K} → re-rank → top-{settings.RERANK_TOP_N}")
    print(f"  Query expand  : {'on' if settings.QUERY_EXPANSION_ENABLED else 'off'}")
    print(f"  Compression   : {'on' if settings.CONTEXT_COMPRESSION_ENABLED else 'off'}")
    print()


def cmd_interactive():
    """Simple REPL for interactive Q&A."""
    from rag_pipeline import list_documents, query

    docs = list_documents()
    if not docs:
        warn("No documents in the knowledge base. Upload some files first.")
        return

    print(f"\n{BOLD}Interactive mode{RESET} — {len(docs)} document(s) loaded")
    print(f"{DIM}Type your question and press Enter. Ctrl+C to exit.{RESET}\n")

    async def _ask(q):
        return await query(q)

    while True:
        try:
            q = input(f"{CYAN}>{RESET} ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break
        if not q:
            continue
        try:
            result = asyncio.run(_ask(q))
            print(f"\n{result['answer']}\n")
            if result["citations"]:
                for c in result["citations"]:
                    print(f"  {DIM}— {c['filename']} ({c['relevance_score']:.0%}){RESET}")
            print()
        except Exception as exc:
            err(str(exc))


# ── Entry point ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="rag-cli",
        description="RAG Knowledge Assistant — command-line interface",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # upload
    p_up = sub.add_parser("upload", help="Upload and ingest one or more documents")
    p_up.add_argument("files", nargs="+", help="Path(s) to PDF / DOCX / TXT files")

    # list
    sub.add_parser("list", help="List all documents in the knowledge base")

    # ask
    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", help="The question to ask")

    # delete
    p_del = sub.add_parser("delete", help="Remove a document by doc_id")
    p_del.add_argument("doc_id", help="Document ID to remove")

    # stats
    sub.add_parser("stats", help="Show system statistics and configuration")

    # interactive
    sub.add_parser("interactive", help="Start an interactive Q&A session")

    args = parser.parse_args()

    dispatch = {
        "upload":      lambda: cmd_upload(args.files),
        "list":        cmd_list,
        "ask":         lambda: cmd_ask(args.question),
        "delete":      lambda: cmd_delete(args.doc_id),
        "stats":       cmd_stats,
        "interactive": cmd_interactive,
    }

    try:
        dispatch[args.command]()
    except KeyboardInterrupt:
        print("\nInterrupted.")
    except Exception as exc:
        err(f"Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
