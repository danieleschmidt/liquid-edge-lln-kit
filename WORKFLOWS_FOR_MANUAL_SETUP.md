# GitHub Actions Workflows for Manual Setup

Due to GitHub security restrictions, the following 6 comprehensive workflows need to be manually reviewed and committed by a repository administrator with appropriate permissions.

## Workflow Files to Create

### 1. `.github/workflows/ci.yml` - Continuous Integration

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

### 2. `.github/workflows/security.yml` - Security Scanning

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

  container-security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      
      - name: Build Docker image
        run: docker build -f Dockerfile.dev -t liquid-edge:test .
        
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'liquid-edge:test'
          format: 'sarif'
          output: 'trivy-results.sarif'
          
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### 3. `.github/workflows/release.yml` - SLSA Level 3 Releases

```yaml
name: Release

on:
  push:
    tags:
      - 'v*'

jobs:
  build:
    runs-on: ubuntu-latest
    outputs:
      hashes: ${{ steps.hash.outputs.hashes }}
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
          
      - name: Build package
        run: |
          pip install build
          python -m build
          
      - name: Generate hashes for SLSA
        shell: bash
        id: hash
        run: |
          cd dist && echo "hashes=$(sha256sum * | base64 -w0)" >> $GITHUB_OUTPUT
          
      - name: Upload build artifacts
        uses: actions/upload-artifact@v3
        with:
          name: built-packages
          path: dist/
          
  generate-provenance:
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.build.outputs.hashes }}"
      upload-assets: true

  publish:
    needs: [build, generate-provenance]
    runs-on: ubuntu-latest
    permissions:
      id-token: write
    steps:
      - name: Download build artifacts
        uses: actions/download-artifact@v3
        with:
          name: built-packages
          path: dist/
          
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          attestations: true
```

### 4. `.github/workflows/docker.yml` - Container Builds

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
      
      - name: Extract metadata
        id: meta
        uses: docker/metadata-action@v5
        with:
          images: ghcr.io/${{ github.repository }}
          tags: |
            type=ref,event=branch
            type=ref,event=pr
            type=semver,pattern={{version}}
            type=semver,pattern={{major}}.{{minor}}
            
      - name: Build and push
        uses: docker/build-push-action@v5
        with:
          context: .
          file: Dockerfile.dev
          push: ${{ github.event_name != 'pull_request' }}
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
          cache-from: type=gha
          cache-to: type=gha,mode=max
```

### 5. `.github/workflows/performance.yml` - Performance Testing

```yaml
name: Performance

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"
      
      - name: Install dependencies
        run: pip install -e ".[dev]"
        
      - name: Run benchmarks
        run: pytest tests/ -m benchmark --benchmark-json=benchmark.json
        
      - name: Store benchmark result
        uses: benchmark-action/github-action-benchmark@v1
        with:
          tool: 'pytest'
          output-file-path: benchmark.json
          github-token: ${{ secrets.GITHUB_TOKEN }}
          auto-push: true
          comment-on-alert: true
          alert-threshold: '200%'
          fail-on-alert: true
```

### 6. `.github/workflows/sbom.yml` - Software Bill of Materials

```yaml
name: SBOM Generation

on:
  push:
    branches: [main]
  release:
    types: [published]

jobs:
  generate-sbom:
    runs-on: ubuntu-latest
    permissions:
      contents: write
      id-token: write
    steps:
      - uses: actions/checkout@v4
      
      - name: Generate SBOM
        uses: anchore/sbom-action@v0
        with:
          path: .
          format: spdx-json
          output-file: liquid-edge-sbom.spdx.json
          
      - name: Upload SBOM
        uses: actions/upload-artifact@v3
        with:
          name: sbom
          path: liquid-edge-sbom.spdx.json
          
      - name: Attest SBOM
        uses: actions/attest-sbom@v1
        with:
          subject-path: liquid-edge-sbom.spdx.json
          sbom-path: liquid-edge-sbom.spdx.json
```

## Setup Instructions

### Prerequisites
1. Repository administrator permissions
2. Enable GitHub Actions in repository settings
3. Configure required secrets

### Required Repository Secrets
- `PYPI_API_TOKEN`: For automated PyPI publishing
- `CODECOV_TOKEN`: For coverage reporting (optional but recommended)

### Implementation Steps
1. **Create workflow directory**: `mkdir -p .github/workflows`
2. **Copy workflow files**: Create each of the 6 YAML files above
3. **Commit workflows**: `git add .github/workflows && git commit -m "feat: add comprehensive CI/CD workflows"`
4. **Configure branch protection**: Require CI checks to pass before merging
5. **Test workflows**: Create a test pull request to validate all pipelines

### Branch Protection Configuration
Enable the following required status checks:
- `test` (from ci.yml)
- `lint` (from ci.yml)  
- `security` (from security.yml)
- `container-security` (from security.yml)

## Workflow Benefits

### CI/CD Pipeline Features
- **Multi-Python testing**: Python 3.10, 3.11, 3.12 compatibility
- **Code quality**: Automated linting, formatting, and type checking
- **Security scanning**: Dependency vulnerabilities, container security, static analysis
- **Performance testing**: Automated benchmarks with regression detection
- **SLSA compliance**: Supply chain security with cryptographic attestations
- **Container registry**: Automated Docker builds with caching optimization

### Production Readiness
These workflows provide enterprise-grade automation for:
- Continuous integration and testing
- Security vulnerability management
- Automated releases with provenance
- Performance regression detection  
- Container security scanning
- Software bill of materials generation

The complete pipeline ensures safe, secure, and reliable deployment of liquid neural networks for edge computing applications.