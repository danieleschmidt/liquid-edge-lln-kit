.PHONY: help install install-dev test test-cov test-fast lint fmt typecheck security benchmark clean docker build docs

# Default target
help:  ## Show this help message
	@echo "Liquid Edge LLN Kit - Development Commands"
	@echo "=========================================="
	@echo ""
	@echo "Setup:"
	@echo "  install      Install package in editable mode"
	@echo "  install-dev  Install with development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  test         Run full test suite"
	@echo "  test-cov     Run tests with coverage report"
	@echo "  test-fast    Run tests with fail-fast"
	@echo "  benchmark    Run performance benchmarks"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         Run code linting (ruff)"
	@echo "  fmt          Format code (black + ruff --fix)"
	@echo "  typecheck    Run type checking (mypy)"
	@echo "  security     Run security scans"
	@echo ""
	@echo "Build & Deploy:"
	@echo "  build        Build package distributions"
	@echo "  docker       Build development container"
	@echo "  docs         Build documentation"
	@echo ""
	@echo "Utilities:"
	@echo "  clean        Clean build artifacts"

# Installation targets  
install:  ## Install package in development mode
	pip install -e .

install-dev:  ## Install with development dependencies
	pip install -e ".[dev,deployment,ros2]"
	pre-commit install

# Testing targets
test:  ## Run full test suite
	python -m pytest tests/ -v

test-cov:  ## Run tests with coverage report
	python -m pytest tests/ --cov=liquid_edge --cov-report=html --cov-report=term-missing

test-fast:  ## Run tests with fail-fast
	python -m pytest tests/ -x -vvs

test-unit:  ## Run only unit tests
	python -m pytest tests/ -m "not integration and not hardware" -v

# Benchmarking
benchmark:  ## Run performance benchmarks
	python scripts/benchmark.py

# Code quality targets
lint:  ## Run code linting
	ruff check src/ tests/ scripts/

fmt:  ## Format code
	black src/ tests/ scripts/
	ruff check --fix src/ tests/ scripts/

typecheck:  ## Run type checking
	mypy src/liquid_edge/

# Security scanning
security:  ## Run security scans
	python scripts/security_scan.py

# Build targets
build:  ## Build package distributions
	python -m build

# Container targets
docker:  ## Build development container
	docker build -f Dockerfile.dev -t liquid-edge-dev .

docker-compose:  ## Start development environment
	docker-compose up -d

# Documentation
docs:  ## Build documentation
	cd docs && python -m http.server 8080

# Development utilities
clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

pre-commit-run:  ## Run pre-commit hooks
	pre-commit run --all-files

# Development workflow shortcuts
dev-setup: install-dev  ## Setup development environment
	@echo "Development environment setup complete!"

dev-check: lint typecheck test-unit security  ## Quick development checks
	@echo "Development checks passed!"

pr-ready: fmt lint typecheck test security  ## Full validation before PR
	@echo "Pull request validation complete!"

release:  ## Release checklist
	@echo "Release checklist:"
	@echo "1. Update version in pyproject.toml"
	@echo "2. Update CHANGELOG.md" 
	@echo "3. Run: make pr-ready"
	@echo "4. Run: make build"
	@echo "5. Git tag and push"
	@echo "6. Upload to PyPI: twine upload dist/*"