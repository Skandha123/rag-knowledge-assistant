#!/usr/bin/env bash
# =============================================================================
#  RAG Knowledge Assistant — Automated Setup Script
#  Run this from the project root: bash setup.sh
# =============================================================================

set -e  # Exit on any error

# ── Colors ────────────────────────────────────────────────────────────────────
BOLD="\033[1m"; GREEN="\033[32m"; YELLOW="\033[33m"; RED="\033[31m"; CYAN="\033[36m"; RESET="\033[0m"

ok()   { echo -e "${GREEN}✓${RESET} $1"; }
warn() { echo -e "${YELLOW}⚠${RESET}  $1"; }
err()  { echo -e "${RED}✗${RESET}  $1"; exit 1; }
info() { echo -e "${CYAN}ℹ${RESET}  $1"; }
step() { echo -e "\n${BOLD}$1${RESET}"; }

# ── Banner ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║   Enterprise RAG Knowledge Assistant — Setup    ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════╝${RESET}"
echo ""

# ── Check we're in the right directory ────────────────────────────────────────
if [ ! -f "requirements.txt" ] || [ ! -d "backend" ]; then
  err "Run this script from the project root (where requirements.txt lives)."
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Python version check
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 1 — Checking Python version"
PYTHON=""
for cmd in python3.12 python3.11 python3.10 python3 python; do
  if command -v "$cmd" &>/dev/null; then
    VER=$($cmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    MAJOR=$(echo $VER | cut -d. -f1)
    MINOR=$(echo $VER | cut -d. -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 10 ]; then
      PYTHON="$cmd"
      ok "Found $cmd (Python $VER)"
      break
    fi
  fi
done

if [ -z "$PYTHON" ]; then
  err "Python 3.10+ is required but not found. Install from https://python.org/downloads/"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Node.js version check
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 2 — Checking Node.js version"
if ! command -v node &>/dev/null; then
  err "Node.js 18+ is required but not found. Install from https://nodejs.org/"
fi

NODE_VER=$(node -v | sed 's/v//' | cut -d. -f1)
if [ "$NODE_VER" -lt 18 ]; then
  err "Node.js 18+ required, found v$(node -v). Upgrade at https://nodejs.org/"
fi
ok "Found Node.js $(node -v)"

if ! command -v npm &>/dev/null; then
  err "npm not found. It should come with Node.js."
fi
ok "Found npm $(npm -v)"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 3 — Create Python virtual environment
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 3 — Creating Python virtual environment"
if [ -d ".venv" ]; then
  warn ".venv already exists — skipping creation"
else
  $PYTHON -m venv .venv
  ok "Virtual environment created at .venv/"
fi

# Activate
source .venv/bin/activate
ok "Virtual environment activated"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 4 — Install Python dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 4 — Installing Python dependencies (this may take 2–5 minutes)"
info "Installing: FastAPI, LangChain, ChromaDB, SentenceTransformers, OpenAI…"

pip install --upgrade pip --quiet
pip install -r requirements.txt

ok "Python dependencies installed"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 5 — Download embedding model
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 5 — Pre-downloading the embedding model (all-MiniLM-L6-v2)"
info "This downloads ~90MB from HuggingFace (one-time only)…"

python -c "
from sentence_transformers import SentenceTransformer
print('  Downloading all-MiniLM-L6-v2…')
model = SentenceTransformer('all-MiniLM-L6-v2')
dim = model.get_sentence_embedding_dimension()
print(f'  Model ready — embedding dimension: {dim}')
"
ok "Embedding model downloaded and cached"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 6 — Create .env file
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 6 — Configuring environment"
if [ -f ".env" ]; then
  warn ".env already exists — skipping. Edit it manually if needed."
else
  cp .env.example .env
  ok ".env created from .env.example"

  echo ""
  echo -e "  ${YELLOW}ACTION REQUIRED:${RESET} Open .env and set your OpenAI API key:"
  echo -e "  ${BOLD}  OPENAI_API_KEY=sk-...${RESET}"
  echo ""
  echo "  Or to use Ollama (free, local):"
  echo "    LLM_PROVIDER=ollama"
  echo "    OLLAMA_MODEL=llama3"
fi

# ─────────────────────────────────────────────────────────────────────────────
# STEP 7 — Create data directories
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 7 — Creating data directories"
mkdir -p data/chroma_db data/uploads
ok "data/chroma_db and data/uploads created"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 8 — Install frontend dependencies
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 8 — Installing frontend dependencies (npm install)"
cd frontend
npm install --silent
cd ..
ok "Frontend dependencies installed"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 9 — Quick smoke test (import check)
# ─────────────────────────────────────────────────────────────────────────────
step "STEP 9 — Running import smoke test"
python -c "
import sys
sys.path.insert(0, 'backend')
errors = []
modules = [
    ('fastapi',              'FastAPI'),
    ('chromadb',             'ChromaDB'),
    ('sentence_transformers','SentenceTransformers'),
    ('langchain',            'LangChain'),
    ('openai',               'OpenAI'),
    ('pypdf',                'pypdf'),
    ('docx',                 'python-docx'),
]
for mod, name in modules:
    try:
        __import__(mod)
        print(f'  ✓ {name}')
    except ImportError as e:
        print(f'  ✗ {name}: {e}')
        errors.append(name)
if errors:
    print(f'\nMissing: {errors}')
    sys.exit(1)
"
ok "All imports successful"

# ─────────────────────────────────────────────────────────────────────────────
# Done
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔═══════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║          Setup complete!  🎉              ║${RESET}"
echo -e "${GREEN}${BOLD}╚═══════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  ${BOLD}Next steps:${RESET}"
echo ""
echo -e "  1. ${YELLOW}Edit .env and add your OPENAI_API_KEY${RESET}"
echo ""
echo -e "  2. Start the backend:"
echo -e "     ${CYAN}source .venv/bin/activate${RESET}"
echo -e "     ${CYAN}uvicorn backend.main:app --reload${RESET}"
echo ""
echo -e "  3. In a new terminal, start the frontend:"
echo -e "     ${CYAN}cd frontend && npm run dev${RESET}"
echo ""
echo -e "  4. Open your browser:"
echo -e "     ${BOLD}http://localhost:5173${RESET}  (UI)"
echo -e "     ${BOLD}http://localhost:8000/docs${RESET}  (API docs)"
echo ""
