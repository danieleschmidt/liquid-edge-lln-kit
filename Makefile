.PHONY: help install test lint format type-check clean build docs

help:  ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in development mode
	pip install -e ".[dev]"
	pre-commit install

test:  ## Run tests
	pytest

test-cov:  ## Run tests with coverage
	pytest --cov=liquid_edge --cov-report=term-missing --cov-report=html

lint:  ## Run linting
	ruff check src/ tests/ examples/

format:  ## Format code
	black src/ tests/ examples/
	ruff check --fix src/ tests/ examples/

type-check:  ## Run type checking
	mypy src/

quality:  ## Run all quality checks
	make format
	make lint  
	make type-check
	make test

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build distribution packages
	python -m build

docs:  ## Build documentation
	@echo "Documentation build not yet implemented"
	@echo "See docs/ directory for markdown documentation"

release:  ## Release checklist
	@echo "Release checklist:"
	@echo "1. Update version in pyproject.toml"
	@echo "2. Update CHANGELOG.md" 
	@echo "3. Run: make quality"
	@echo "4. Run: make build"
	@echo "5. Git tag and push"
	@echo "6. Upload to PyPI: twine upload dist/*"