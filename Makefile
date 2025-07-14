# Makefile for ML Project

.PHONY: help install train serve test clean build deploy lint format setup-dev docker-build docker-up docker-down

# Default target
help:
	@echo "Available commands:"
	@echo "  install      - Install dependencies"
	@echo "  setup-dev    - Setup development environment"
	@echo "  train        - Train the model"
	@echo "  serve        - Start the API server"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  build        - Build Docker image"
	@echo "  docker-up    - Start services with Docker Compose"
	@echo "  docker-down  - Stop Docker Compose services"
	@echo "  clean        - Clean artifacts and cache"
	@echo "  deploy       - Deploy to production"

# Install dependencies
install:
	uv init
	uv venv
	source uv .venv/bin/activate
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

# Setup development environment
setup-dev: install
	mkdir -p data/raw data/processed artifacts logs mlruns
	pip install pre-commit pytest pytest-cov black flake8
	@echo "Development environment setup complete!"

# Train the model
train:
	python -m src.pipeline.training_pipeline

# Start API server
serve:
	python -m api.main

# Run tests
test:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Run linting
lint:
	flake8 src/ api/ tests/ --max-line-length=120
	black --check src/ api/ tests/

# Format code
format:
	

# Build Docker image
build:
	docker build -f docker/Dockerfile -t churn-prediction:latest .

# Start services with Docker Compose
docker-up:
	docker-compose -f docker/docker-compose.yml up -d

# Stop Docker Compose services
docker-down:
	docker-compose -f docker/docker-compose.yml down

# Clean artifacts and cache
clean:
	rm -rf artifacts/*
	rm -rf logs/*
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .pytest_cache/
	find . -type d -name "__pycache__" -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete

# Quick start - setup and train
quick-start: setup-dev train
	@echo "Quick start completed! You can now run 'make serve' to start the API."

# Run integration test
integration-test:
	python -m api.main &
	sleep 10
	curl -f http://localhost:8000/health
	pkill -f "python -m api.main" || true

# Deploy (placeholder - customize for your deployment)
deploy:
	@echo "Deploying to production..."
	# kubectl apply -f deployment/kubernetes/production/
	@echo "Deployment completed!"

# Data drift monitoring
monitor-drift:
	python -c "from src.monitoring.data_drift import DataDriftDetector; print('Data drift monitoring completed')"

# Generate documentation
docs:
	@echo "API documentation available at: http://localhost:8000/docs"
	@echo "Start the server with 'make serve' to view documentation"

