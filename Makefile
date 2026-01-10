# Citizen Support Assistant - Makefile
# Common development and deployment commands

.PHONY: help install run run-ui run-api docker-build docker-up docker-down test lint format clean

# Default target
help:
	@echo "🏛️ Citizen Support Assistant"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@echo "  install       Install dependencies in virtual environment"
	@echo "  run-ui        Run Streamlit frontend (recommended)"
	@echo "  run-api       Run FastAPI backend only"
	@echo "  test          Run tests"
	@echo "  lint          Run linting"
	@echo "  format        Format code with black and isort"
	@echo ""
	@echo "Docker:"
	@echo "  docker-ui     Build and run Streamlit with Docker"
	@echo "  docker-api    Build and run FastAPI with Docker"
	@echo "  docker-all    Build and run both services"
	@echo "  docker-down   Stop all Docker services"
	@echo "  docker-logs   View Docker logs"
	@echo ""
	@echo "Utilities:"
	@echo "  clean         Remove generated files"
	@echo "  setup-env     Create .env from template"
	@echo "  check-ollama  Verify Ollama is running"

# Check Ollama
check-ollama:
	@echo "Checking Ollama..."
	@curl -s http://localhost:11434/api/tags > /dev/null 2>&1 && echo "✅ Ollama is running" || echo "❌ Ollama is not running. Start with: ollama serve"

# Development
install:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	@echo "✅ Installation complete. Activate with: source venv/bin/activate"

run-ui: check-ollama
	@echo "🚀 Starting Streamlit UI..."
	. venv/bin/activate && cd streamlit_app && streamlit run app.py --server.port 8501

run-api: check-ollama
	@echo "🚀 Starting FastAPI..."
	. venv/bin/activate && uvicorn app.main:app --reload --port 8000

# Alias for run-ui
run: run-ui

test:
	. venv/bin/activate && pytest tests/ -v

test-cov:
	. venv/bin/activate && pytest tests/ --cov=app --cov-report=html

lint:
	. venv/bin/activate && ruff check app/ streamlit_app/ tests/ || true
	. venv/bin/activate && mypy app/ || true

format:
	. venv/bin/activate && black app/ streamlit_app/ tests/
	. venv/bin/activate && isort app/ streamlit_app/ tests/

# Docker
docker-ui:
	docker compose up streamlit --build

docker-api:
	docker compose --profile api up api --build

docker-all:
	docker compose --profile api up --build

docker-down:
	docker compose --profile api down

docker-logs:
	docker compose logs -f

# Utilities
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .coverage htmlcov
	rm -rf data/tts_output/*.mp3
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

setup-env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✅ Created .env from template."; \
	else \
		echo "ℹ️ .env already exists"; \
	fi

# Create required directories
dirs:
	mkdir -p data/chroma data/tts_output data models/whisper knowledge
