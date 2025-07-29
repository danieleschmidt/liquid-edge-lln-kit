# Required GitHub Workflows

This document outlines the essential GitHub Actions workflows needed for this repository's CI/CD pipeline.

## 1. Main CI Workflow (.github/workflows/ci.yml)

```yaml
name: CI
on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
          
      - name: Run tests
        run: pytest --cov=liquid_edge --cov-report=xml
        
      - name: Upload coverage
        uses: codecov/codecov-action@v3

  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      - run: pip install -e ".[dev]"
      - run: ruff check .
      - run: black --check .
      - run: mypy src/
```

## 2. Security Scanning (.github/workflows/security.yml)

```yaml
name: Security
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 0 * * 1'  # Weekly

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
        
      - name: Run bandit
        run: bandit -r src/ -f json -o bandit-report.json
        
      - name: Run safety
        run: safety check --json --output safety-report.json
        
      - name: Run CodeQL
        uses: github/codeql-action/analyze@v2
        with:
          languages: python
```

## 3. Release Workflow (.github/workflows/release.yml) 

```yaml
name: Release
on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
          
      - name: Build package
        run: |
          pip install build
          python -m build
          
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@v1.8.10
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

## 4. Docker Build (.github/workflows/docker.yml)

```yaml
name: Docker
on:
  push:
    branches: [main]
    tags: ['v*']
  pull_request:
    branches: [main]

jobs:
  docker:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3
        
      - name: Login to registry
        if: github.event_name != 'pull_request'
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          file: Dockerfile.dev
          push: ${{ github.event_name != 'pull_request' }}
          tags: ghcr.io/${{ github.repository }}:latest
```

## Setup Instructions

1. Create the workflow files in `.github/workflows/`
2. Add required secrets to repository settings:
   - `PYPI_API_TOKEN` for PyPI publishing
   - `CODECOV_TOKEN` for coverage reporting (optional)
3. Enable GitHub Actions in repository settings
4. Configure branch protection rules requiring CI checks