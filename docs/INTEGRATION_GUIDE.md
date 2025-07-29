# Integration Guide

## Overview

This guide provides comprehensive instructions for integrating the Liquid Edge LLN Kit into existing projects, CI/CD pipelines, and development workflows.

## Quick Integration Checklist

- [ ] Environment setup complete
- [ ] Dependencies installed
- [ ] CI/CD workflows configured
- [ ] Security scanning enabled
- [ ] Monitoring integrated
- [ ] Documentation updated
- [ ] Team training completed

## Development Environment Integration

### IDE Configuration

#### Visual Studio Code
```json
// .vscode/settings.json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": false,
    "python.linting.ruffEnabled": true,
    "python.formatting.provider": "black",
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": ["tests/"],
    "files.associations": {
        "*.liquid": "json"
    }
}

// .vscode/extensions.json
{
    "recommendations": [
        "ms-python.python",
        "ms-python.black-formatter",
        "charliermarsh.ruff",
        "ms-python.pytest",
        "redhat.vscode-yaml",
        "ms-vscode.makefile-tools"
    ]
}
```

#### PyCharm Configuration
```xml
<!-- .idea/liquid-edge.iml -->
<module type="PYTHON_MODULE" version="4">
  <component name="NewModuleRootManager">
    <content url="file://$MODULE_DIR$">
      <sourceFolder url="file://$MODULE_DIR$/src" isTestSource="false" />
      <excludeFolder url="file://$MODULE_DIR$/build" />
      <excludeFolder url="file://$MODULE_DIR$/dist" />
      <excludeFolder url="file://$MODULE_DIR$/.pytest_cache" />
    </content>
    <orderEntry type="jdk" jdkName="Python 3.10 (liquid-edge)" jdkType="Python SDK" />
    <orderEntry type="sourceFolder" forTests="false" />
  </component>
</module>
```

### Development Container Setup

#### DevContainer Configuration
```json
// .devcontainer/devcontainer.json
{
    "name": "Liquid Edge Development",
    "image": "python:3.10-slim",
    "features": {
        "ghcr.io/devcontainers/features/docker-in-docker:2": {},
        "ghcr.io/devcontainers/features/node:1": {"version": "18"}
    },
    "postCreateCommand": "make install-dev",
    "customizations": {
        "vscode": {
            "extensions": [
                "ms-python.python",
                "ms-python.black-formatter",
                "charliermarsh.ruff"
            ]
        }
    },
    "forwardPorts": [8000, 8080],
    "remoteUser": "vscode"
}
```

#### Dockerfile for Development
```dockerfile
# .devcontainer/Dockerfile
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    make \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -s /bin/bash vscode
USER vscode
WORKDIR /workspace

# Install Python dependencies
COPY requirements.txt .
RUN pip install --user -r requirements.txt

# Configure git
RUN git config --global --add safe.directory /workspace
```

## CI/CD Pipeline Integration

### GitHub Actions Integration

#### Workflow Triggers
```yaml
# Example: Integration with existing workflows
name: Integrate Liquid Edge

on:
  workflow_call:
    inputs:
      python-version:
        required: false
        type: string
        default: "3.10"
    secrets:
      PYPI_API_TOKEN:
        required: false

jobs:
  liquid-edge-tests:
    uses: ./.github/workflows/ci.yml
    with:
      python-version: ${{ inputs.python-version }}
    secrets: inherit
```

#### Custom Workflow Extension
```yaml
# .github/workflows/custom-integration.yml
name: Custom Integration

on:
  push:
    paths: ['models/**', 'configs/**']

jobs:
  model-validation:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Liquid Edge
        uses: ./actions/setup-liquid-edge
        with:
          python-version: "3.10"
          install-extras: "deployment,ros2"
      
      - name: Validate Models
        run: |
          liquid-lln validate models/
          liquid-lln benchmark --model-dir models/
```

### GitLab CI Integration

#### .gitlab-ci.yml Configuration
```yaml
stages:
  - setup
  - test
  - security
  - deploy

variables:
  PYTHON_VERSION: "3.10"
  PIP_CACHE_DIR: "$CI_PROJECT_DIR/.cache/pip"

cache:
  paths:
    - .cache/pip
    - venv/

setup:liquid-edge:
  stage: setup
  image: python:${PYTHON_VERSION}
  script:
    - python -m venv venv
    - source venv/bin/activate
    - pip install -e ".[dev,deployment]"
  artifacts:
    paths:
      - venv/
    expire_in: 1 hour

test:liquid-edge:
  stage: test
  dependencies:
    - setup:liquid-edge
  script:
    - source venv/bin/activate
    - pytest tests/ --cov=liquid_edge --cov-report=xml
  coverage: '/TOTAL.*\s+(\d+%)$/'
  artifacts:
    reports:
      coverage_report:
        coverage_format: cobertura
        path: coverage.xml

security:bandit:
  stage: security
  dependencies:
    - setup:liquid-edge
  script:
    - source venv/bin/activate
    - bandit -r src/ -f json -o bandit-report.json
  artifacts:
    reports:
      sast: bandit-report.json
```

### Jenkins Integration

#### Jenkinsfile
```groovy
pipeline {
    agent any
    
    environment {
        PYTHON_VERSION = '3.10'
        VIRTUAL_ENV = "${WORKSPACE}/venv"
    }
    
    stages {
        stage('Setup') {
            steps {
                sh """
                    python${PYTHON_VERSION} -m venv ${VIRTUAL_ENV}
                    . ${VIRTUAL_ENV}/bin/activate
                    pip install -e ".[dev,deployment]"
                """
            }
        }
        
        stage('Test') {
            steps {
                sh """
                    . ${VIRTUAL_ENV}/bin/activate
                    pytest tests/ --junitxml=test-results.xml --cov=liquid_edge --cov-report=xml
                """
            }
            post {
                always {
                    junit 'test-results.xml'
                    publishCoverage(
                        adapters: [coberturaAdapter('coverage.xml')],
                        sourceFileResolver: sourceFiles('STORE_LAST_BUILD')
                    )
                }
            }
        }
        
        stage('Security') {
            steps {
                sh """
                    . ${VIRTUAL_ENV}/bin/activate
                    bandit -r src/ -f json -o bandit-report.json
                    safety check --json --output safety-report.json
                """
            }
            post {
                always {
                    archiveArtifacts artifacts: '*-report.json', fingerprint: true
                }
            }
        }
        
        stage('Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh """
                    . ${VIRTUAL_ENV}/bin/activate
                    python -m build
                    twine upload dist/* --repository-url ${PYPI_REPOSITORY_URL}
                """
            }
        }
    }
}
```

## Docker Integration

### Multi-stage Dockerfile
```dockerfile
# Dockerfile.production
FROM python:3.10-slim as builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

RUN pip install build && python -m build

FROM python:3.10-slim as runtime

LABEL org.opencontainers.image.title="Liquid Edge LLN Kit"
LABEL org.opencontainers.image.description="Tiny liquid neural networks for edge robotics"
LABEL org.opencontainers.image.source="https://github.com/liquid-edge/liquid-edge-lln-kit"

# Create non-root user
RUN groupadd -r liquid && useradd -r -g liquid liquid

WORKDIR /app
COPY --from=builder /app/dist/*.whl .

RUN pip install *.whl && rm *.whl

USER liquid
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD liquid-lln health-check || exit 1

CMD ["liquid-lln", "serve", "--host", "0.0.0.0", "--port", "8080"]
```

### Docker Compose Integration
```yaml
# docker-compose.production.yml
version: '3.8'

services:
  liquid-edge:
    build:
      context: .
      dockerfile: Dockerfile.production
    ports:
      - "8080:8080"
    environment:
      - LIQUID_EDGE_CONFIG=/app/config/production.yaml
      - OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
    volumes:
      - ./config:/app/config:ro
      - ./models:/app/models:ro
    healthcheck:
      test: ["CMD", "liquid-lln", "health-check"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped
    depends_on:
      - prometheus
      - jaeger

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - ./monitoring/grafana:/var/lib/grafana/dashboards:ro

  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "14268:14268"
      - "16686:16686"
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

## Kubernetes Integration

### Deployment Configuration
```yaml
# k8s/deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: liquid-edge-lln
  labels:
    app: liquid-edge-lln
spec:
  replicas: 3
  selector:
    matchLabels:
      app: liquid-edge-lln
  template:
    metadata:
      labels:
        app: liquid-edge-lln
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      containers:
      - name: liquid-edge-lln
        image: ghcr.io/liquid-edge/liquid-edge-lln-kit:latest
        ports:
        - containerPort: 8080
          name: http
        env:
        - name: OTEL_EXPORTER_OTLP_ENDPOINT
          value: "http://jaeger-collector:14268/api/traces"
        - name: LIQUID_EDGE_CONFIG
          value: "/etc/liquid-edge/config.yaml"
        resources:
          requests:
            memory: "256Mi"
            cpu: "100m"
          limits:
            memory: "512Mi"
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
        volumeMounts:
        - name: config
          mountPath: /etc/liquid-edge
          readOnly: true
        - name: models
          mountPath: /app/models
          readOnly: true
      volumes:
      - name: config
        configMap:
          name: liquid-edge-config
      - name: models
        persistentVolumeClaim:
          claimName: liquid-edge-models

---
apiVersion: v1
kind: Service
metadata:
  name: liquid-edge-lln-service
  labels:
    app: liquid-edge-lln
spec:
  selector:
    app: liquid-edge-lln
  ports:
  - port: 80
    targetPort: 8080
    name: http
  type: LoadBalancer

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: liquid-edge-config
data:
  config.yaml: |
    server:
      host: 0.0.0.0
      port: 8080
    model:
      cache_size: 100
      default_device: cpu
    monitoring:
      enabled: true
      metrics_port: 8080
    logging:
      level: INFO
      format: json
```

### Helm Chart Integration
```yaml
# charts/liquid-edge-lln/values.yaml
replicaCount: 3

image:
  repository: ghcr.io/liquid-edge/liquid-edge-lln-kit
  tag: "latest"
  pullPolicy: IfNotPresent

service:
  type: LoadBalancer
  port: 80
  targetPort: 8080

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
  hosts:
  - host: liquid-edge.example.com
    paths:
    - path: /
      pathType: Prefix
  tls:
  - secretName: liquid-edge-tls
    hosts:
    - liquid-edge.example.com

resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "512Mi"
    cpu: "500m"

monitoring:
  prometheus:
    enabled: true
    port: 8080
    path: /metrics
  
  grafana:
    enabled: true
    dashboards:
      enabled: true

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80
```

## Monitoring Integration

### Prometheus Configuration
```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "liquid_edge_rules.yml"

alerting:
  alertmanagers:
  - static_configs:
    - targets:
      - alertmanager:9093

scrape_configs:
  - job_name: 'liquid-edge-lln'
    static_configs:
    - targets: ['liquid-edge:8080']
    metrics_path: '/metrics'
    scrape_interval: 10s
    
  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
    - role: pod
    relabel_configs:
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
      action: keep
      regex: true
    - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_path]
      action: replace
      target_label: __metrics_path__
      regex: (.+)
```

### Grafana Dashboard Provisioning
```yaml
# monitoring/grafana/provisioning/dashboards/liquid-edge.yaml
apiVersion: 1

providers:
  - name: 'liquid-edge-dashboards'
    orgId: 1
    folder: 'Liquid Edge'
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /var/lib/grafana/dashboards
```

## Security Integration

### Security Scanning Integration
```bash
#!/bin/bash
# scripts/security-check.sh

set -e

echo "Running comprehensive security checks..."

# Dependency vulnerability scanning
echo "1. Scanning dependencies..."
safety check --json --output safety-report.json

# Code security analysis
echo "2. Running static analysis..."
bandit -r src/ -f json -o bandit-report.json

# Secret detection
echo "3. Scanning for secrets..."
trufflehog filesystem . --json --output secrets-report.json

# Container security (if Docker is available)
if command -v docker &> /dev/null; then
    echo "4. Scanning container images..."
    docker build -t liquid-edge-security-scan -f Dockerfile.dev .
    trivy image liquid-edge-security-scan --format json --output container-scan.json
fi

# SBOM generation
echo "5. Generating Software Bill of Materials..."
cyclonedx-py -o sbom.json

echo "Security scan complete. Check reports for findings."
```

### Pre-commit Security Hooks
```yaml
# .pre-commit-config.yaml (security additions)
repos:
  - repo: https://github.com/PyCQA/bandit
    rev: '1.7.5'
    hooks:
      - id: bandit
        args: ['-c', '.bandit']
  
  - repo: https://github.com/gitguardian/ggshield
    rev: v1.21.0
    hooks:
      - id: ggshield
        language: python
        stages: [commit]
  
  - repo: https://github.com/Yelp/detect-secrets
    rev: 'v1.4.0'
    hooks:
      - id: detect-secrets
        args: ['--baseline', '.secrets.baseline']
```

## Testing Integration

### Test Configuration Template
```python
# tests/conftest.py integration example
import pytest
import tempfile
import os
from pathlib import Path

@pytest.fixture(scope="session")
def integration_config():
    """Provide integration test configuration"""
    return {
        "test_data_dir": Path(__file__).parent / "data",
        "model_cache_dir": Path(tempfile.mkdtemp()),
        "api_timeout": 30,
        "performance_threshold": 0.1
    }

@pytest.fixture
def mock_hardware_platform():
    """Mock hardware platform for testing"""
    with patch('liquid_edge.deploy.get_platform_info') as mock:
        mock.return_value = {
            "platform": "test",
            "memory_mb": 1024,
            "cpu_cores": 4
        }
        yield mock

@pytest.fixture
def integration_client():
    """HTTP client for integration testing"""
    from liquid_edge.api import create_app
    app = create_app(testing=True)
    with app.test_client() as client:
        yield client
```

## Team Integration

### Training Materials
```markdown
# Team Onboarding Checklist

## Prerequisites
- [ ] Python 3.10+ installed
- [ ] Git and GitHub access configured
- [ ] Docker installed (optional)
- [ ] IDE configured (VS Code or PyCharm)

## Setup Tasks
- [ ] Clone repository: `git clone <repo-url>`
- [ ] Create virtual environment: `python -m venv venv`
- [ ] Install dependencies: `make install-dev`
- [ ] Run tests: `make test`
- [ ] Review architecture documentation
- [ ] Complete security training

## Development Workflow
- [ ] Create feature branch from `develop`
- [ ] Write tests first (TDD approach)
- [ ] Implement feature with type hints
- [ ] Run quality checks: `make pr-ready`
- [ ] Create pull request with description
- [ ] Address review feedback
- [ ] Merge after approvals

## Code Review Guidelines  
- [ ] Security considerations reviewed
- [ ] Performance impact assessed
- [ ] Documentation updated
- [ ] Tests provide adequate coverage
- [ ] Breaking changes documented
```

### Documentation Integration
```python
# docs/conf.py (Sphinx configuration)
import sys
import os
sys.path.insert(0, os.path.abspath('../../src'))

project = 'Liquid Edge LLN Kit'
copyright = '2025, Liquid Edge Team'
author = 'Liquid Edge Team'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx_rtd_theme',
    'myst_parser'
]

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Auto-document from docstrings
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}
```

## Troubleshooting

### Common Integration Issues

#### Import Errors
```bash
# Solution: Check Python path and virtual environment
python -c "import sys; print(sys.path)"
pip list | grep liquid-edge
```

#### Build Failures
```bash
# Solution: Clean and rebuild
make clean
pip install --upgrade pip setuptools wheel
make install-dev
```

#### Test Failures
```bash
# Solution: Run tests with verbose output
pytest -xvs tests/
pytest --lf  # Run only last failed tests
```

#### Security Scan Issues
```bash
# Solution: Update security tools and run individually
pip install --upgrade bandit safety
bandit -r src/
safety check
```

For additional support, see:
- [GitHub Issues](https://github.com/liquid-edge/liquid-edge-lln-kit/issues)
- [Discussions](https://github.com/liquid-edge/liquid-edge-lln-kit/discussions)
- [Documentation](https://liquid-edge.readthedocs.io)