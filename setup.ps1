# =============================================================================
#  RAG Knowledge Assistant — Windows Setup Script (PowerShell)
#  Run: Right-click > "Run with PowerShell" OR: powershell -ExecutionPolicy Bypass -File setup.ps1
# =============================================================================

$ErrorActionPreference = "Stop"

function ok($msg)   { Write-Host "  [OK] $msg" -ForegroundColor Green }
function warn($msg) { Write-Host "  [!!] $msg" -ForegroundColor Yellow }
function err($msg)  { Write-Host "  [ERR] $msg" -ForegroundColor Red; exit 1 }
function info($msg) { Write-Host "  [..] $msg" -ForegroundColor Cyan }
function step($msg) { Write-Host "`n=== $msg ===" -ForegroundColor White }

Write-Host ""
Write-Host "  Enterprise RAG Knowledge Assistant - Windows Setup" -ForegroundColor White
Write-Host ""

# Check project root
if (-not (Test-Path "requirements.txt") -or -not (Test-Path "backend")) {
    err "Run this script from the project root (where requirements.txt lives)."
}

# ── STEP 1: Python ─────────────────────────────────────────────────────────
step "STEP 1 - Checking Python"
$pythonCmd = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "Python 3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 10) { $pythonCmd = $cmd; ok "Found: $ver"; break }
        }
    } catch {}
}
if (-not $pythonCmd) { err "Python 3.10+ not found. Download from https://python.org/downloads/" }

# ── STEP 2: Node.js ────────────────────────────────────────────────────────
step "STEP 2 - Checking Node.js"
try {
    $nodeVer = & node --version
    $major = [int]($nodeVer -replace 'v(\d+)\..*','$1')
    if ($major -lt 18) { err "Node.js 18+ required. Got $nodeVer. Download from https://nodejs.org/" }
    ok "Found Node.js $nodeVer"
    ok "Found npm $(npm --version)"
} catch { err "Node.js not found. Download from https://nodejs.org/" }

# ── STEP 3: Virtual environment ────────────────────────────────────────────
step "STEP 3 - Creating virtual environment"
if (Test-Path ".venv") {
    warn ".venv already exists - skipping"
} else {
    & $pythonCmd -m venv .venv
    ok "Virtual environment created"
}
& .\.venv\Scripts\Activate.ps1
ok "Virtual environment activated"

# ── STEP 4: Python deps ────────────────────────────────────────────────────
step "STEP 4 - Installing Python dependencies (2-5 minutes)"
info "Installing FastAPI, LangChain, ChromaDB, SentenceTransformers..."
pip install --upgrade pip --quiet
pip install -r requirements.txt
ok "Python dependencies installed"

# ── STEP 5: Embedding model ────────────────────────────────────────────────
step "STEP 5 - Downloading embedding model (90MB, one-time)"
python -c @"
from sentence_transformers import SentenceTransformer
print('  Downloading all-MiniLM-L6-v2...')
model = SentenceTransformer('all-MiniLM-L6-v2')
print(f'  Ready - dimension: {model.get_sentence_embedding_dimension()}')
"@
ok "Embedding model ready"

# ── STEP 6: .env ───────────────────────────────────────────────────────────
step "STEP 6 - Configuring environment"
if (Test-Path ".env") {
    warn ".env already exists - skipping"
} else {
    Copy-Item ".env.example" ".env"
    ok ".env created"
    Write-Host ""
    Write-Host "  ACTION: Open .env and set your OPENAI_API_KEY=sk-..." -ForegroundColor Yellow
}

# ── STEP 7: Data dirs ─────────────────────────────────────────────────────
step "STEP 7 - Creating data directories"
New-Item -ItemType Directory -Force -Path "data\chroma_db" | Out-Null
New-Item -ItemType Directory -Force -Path "data\uploads"   | Out-Null
ok "data directories created"

# ── STEP 8: Frontend ──────────────────────────────────────────────────────
step "STEP 8 - Installing frontend (npm install)"
Set-Location frontend
npm install --silent
Set-Location ..
ok "Frontend dependencies installed"

# ── Done ──────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "  NEXT STEPS:" -ForegroundColor White
Write-Host "  1. Edit .env -> set OPENAI_API_KEY=sk-..." -ForegroundColor Yellow
Write-Host ""
Write-Host "  2. Backend (Terminal 1):"
Write-Host "     .\.venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "     uvicorn backend.main:app --reload" -ForegroundColor Cyan
Write-Host ""
Write-Host "  3. Frontend (Terminal 2):"
Write-Host "     cd frontend" -ForegroundColor Cyan
Write-Host "     npm run dev" -ForegroundColor Cyan
Write-Host ""
Write-Host "  4. Open browser:"
Write-Host "     http://localhost:5173   (UI)" -ForegroundColor Cyan
Write-Host "     http://localhost:8000/docs  (API)" -ForegroundColor Cyan
Write-Host ""
