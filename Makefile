.PHONY: help install dev-backend dev-frontend dev test lint clean docker-up docker-down

# ── Default ────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "  Enterprise RAG Knowledge Assistant"
	@echo ""
	@echo "  make install        Install all dependencies (Python + Node)"
	@echo "  make dev-backend    Run FastAPI dev server (port 8000)"
	@echo "  make dev-frontend   Run Vite dev server (port 5173)"
	@echo "  make dev            Run both servers in parallel"
	@echo "  make test           Run Python test suite"
	@echo "  make test-verbose   Run tests with full output"
	@echo "  make lint           Run ruff linter on backend"
	@echo "  make docker-up      Build and start Docker Compose stack"
	@echo "  make docker-down    Stop Docker Compose stack"
	@echo "  make clean          Remove __pycache__, .pytest_cache, etc."
	@echo ""

# ── Install ────────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt
	cd frontend && npm install

# ── Dev servers ────────────────────────────────────────────────────────────
dev-backend:
	uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

dev:
	@echo "Starting both servers — Ctrl+C to stop both"
	$(MAKE) dev-backend & $(MAKE) dev-frontend & wait

# ── Tests ──────────────────────────────────────────────────────────────────
test:
	pytest -q

test-verbose:
	pytest -v --tb=short

test-coverage:
	pytest --cov=backend --cov-report=term-missing -q

# ── Lint ───────────────────────────────────────────────────────────────────
lint:
	@command -v ruff >/dev/null 2>&1 && ruff check backend/ || echo "ruff not installed: pip install ruff"

# ── Docker ─────────────────────────────────────────────────────────────────
docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

docker-clean:
	docker-compose down -v --rmi local

# ── CLI ────────────────────────────────────────────────────────────────────
cli-upload:
	@read -p "File path: " f; python cli.py upload "$$f"

cli-ask:
	@read -p "Question: " q; python cli.py ask "$$q"

cli-list:
	python cli.py list

cli-stats:
	python cli.py stats

# ── Clean ──────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	find . -name ".coverage" -delete
	rm -rf frontend/dist frontend/.vite
	@echo "Clean complete."
