.PHONY: help install dev test lint format clean docker-build docker-up docker-down migrate backup

# Default target
help:
	@echo "PredictIQ - Available Commands"
	@echo "================================"
	@echo "  make install       - Install all dependencies"
	@echo "  make dev           - Start development servers"
	@echo "  make test          - Run test suite with coverage"
	@echo "  make lint          - Run linters (flake8, oxlint)"
	@echo "  make format        - Format code (black, isort)"
	@echo "  make pipeline      - Run full ML pipeline"
	@echo "  make clean         - Clean temporary files"
	@echo ""
	@echo "Docker Commands:"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker stack"
	@echo "  make docker-down   - Stop Docker stack"
	@echo "  make docker-logs   - View Docker logs"
	@echo ""
	@echo "Database Commands:"
	@echo "  make migrate       - Run database migrations"
	@echo "  make backup        - Backup database"
	@echo ""

# Installation
install:
	@echo "Installing backend dependencies..."
	cd backend && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "✓ All dependencies installed"

# Development
dev:
	@echo "Starting development servers..."
	@echo "Backend will run on http://localhost:8000"
	@echo "Frontend will run on http://localhost:5173"
	@make -j 2 dev-backend dev-frontend

dev-backend:
	cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-frontend:
	cd frontend && npm run dev

# Testing
test:
	@echo "Running backend tests with coverage..."
	cd backend && pytest tests/ -v --cov=. --cov-report=html --cov-report=term
	@echo "✓ Tests completed. Open backend/htmlcov/index.html for coverage report"

test-api:
	@echo "Running API tests only..."
	cd backend && pytest tests/test_api.py -v

test-pipeline:
	@echo "Running pipeline tests only..."
	cd backend && pytest tests/test_pipeline.py -v

# Code Quality
lint:
	@echo "Running backend linters..."
	cd backend && flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
	cd backend && flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics
	@echo "Running frontend linter..."
	cd frontend && npm run lint
	@echo "✓ Linting completed"

format:
	@echo "Formatting backend code..."
	cd backend && black . && isort .
	@echo "✓ Code formatted"

# ML Pipeline
pipeline:
	@echo "Running full ML pipeline..."
	cd backend && python pipeline/preprocessing.py
	cd backend && python pipeline/rfm_features.py
	cd backend && python pipeline/segmentation.py
	cd backend && python pipeline/model_training.py
	cd backend && python pipeline/drift_detection.py
	@echo "✓ Pipeline completed"

pipeline-fast:
	@echo "Running pipeline (skip preprocessing)..."
	cd backend && python pipeline/rfm_features.py
	cd backend && python pipeline/segmentation.py
	cd backend && python pipeline/model_training.py
	@echo "✓ Fast pipeline completed"

# Docker
docker-build:
	@echo "Building Docker images..."
	docker-compose build
	@echo "✓ Docker images built"

docker-up:
	@echo "Starting Docker stack..."
	docker-compose up -d
	@echo "✓ Docker stack started"
	@echo "Backend API: http://localhost:8000"
	@echo "Frontend: http://localhost:3000"
	@echo "Grafana: http://localhost:3001"
	@echo "Prometheus: http://localhost:9090"

docker-down:
	@echo "Stopping Docker stack..."
	docker-compose down
	@echo "✓ Docker stack stopped"

docker-logs:
	docker-compose logs -f

docker-restart:
	@echo "Restarting Docker stack..."
	docker-compose restart
	@echo "✓ Docker stack restarted"

docker-clean:
	@echo "Cleaning Docker resources..."
	docker-compose down -v --remove-orphans
	docker system prune -f
	@echo "✓ Docker resources cleaned"

# Database
migrate:
	@echo "Running database migrations..."
	cd backend && alembic upgrade head
	@echo "✓ Migrations completed"

migrate-create:
	@echo "Creating new migration..."
	@read -p "Migration message: " msg; \
	cd backend && alembic revision --autogenerate -m "$$msg"

backup:
	@echo "Creating database backup..."
	@mkdir -p backups
	docker-compose exec -T postgres pg_dump -U predictiq_user predictiq | gzip > backups/backup_$$(date +%Y%m%d_%H%M%S).sql.gz
	@echo "✓ Backup created in backups/ directory"

restore:
	@echo "Available backups:"
	@ls -lh backups/*.sql.gz
	@read -p "Enter backup filename: " backup; \
	gunzip -c backups/$$backup | docker-compose exec -T postgres psql -U predictiq_user predictiq
	@echo "✓ Database restored"

# Cleanup
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name ".DS_Store" -delete
	rm -rf backend/htmlcov backend/.coverage
	rm -rf frontend/dist frontend/node_modules/.cache
	@echo "✓ Cleanup completed"

# Production
deploy:
	@echo "Deploying to production..."
	@echo "1. Building images..."
	docker-compose build
	@echo "2. Running migrations..."
	docker-compose run --rm backend alembic upgrade head
	@echo "3. Starting services..."
	docker-compose up -d
	@echo "✓ Deployment completed"

health-check:
	@echo "Checking system health..."
	@curl -s http://localhost:8000/api/health | python -m json.tool
	@echo ""
	@docker-compose ps

# Monitoring
logs-backend:
	docker-compose logs -f backend

logs-frontend:
	docker-compose logs -f frontend

logs-db:
	docker-compose logs -f postgres

stats:
	docker stats

# Quick shortcuts
run: docker-up
stop: docker-down
restart: docker-restart
build: docker-build
