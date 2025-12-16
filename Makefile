# Makefile for F1 Analytics Platform
# Usage: make <target>

.PHONY: help install dev test lint format type-check clean run

help:
	@echo "Available commands:"
	@echo "  make install     Install production dependencies"
	@echo "  make dev         Install development dependencies"
	@echo "  make test        Run test suite"
	@echo "  make lint        Run linter (ruff)"
	@echo "  make format      Format code (ruff)"
	@echo "  make type-check  Run type checker (mypy)"
	@echo "  make clean       Remove cache and build artifacts"
	@echo "  make run         Start Streamlit app"

install:
	pip install -r requirements.txt

dev:
	pip install -e ".[dev]"
	pre-commit install

test:
	pytest tests/ -v --cov=src --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/
	ruff check --fix src/ tests/

type-check:
	mypy src/ --ignore-missing-imports

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ 2>/dev/null || true

run:
	streamlit run src/app.py

ci: lint type-check test
	@echo "All CI checks passed"
