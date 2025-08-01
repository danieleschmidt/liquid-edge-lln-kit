# Production container for Liquid Edge LLN Kit
FROM python:3.10-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash --uid 1000 appuser

# ============================================================================
# Builder stage - Install dependencies and build wheels
# ============================================================================
FROM base as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up Python environment
WORKDIR /build
COPY pyproject.toml ./
COPY src/ ./src/

# Install dependencies and build wheel
RUN pip install --upgrade pip setuptools wheel build && \
    pip wheel --no-deps --wheel-dir /build/wheels . && \
    pip wheel --wheel-dir /build/wheels .[deployment]

# ============================================================================
# Runtime stage - Minimal production image
# ============================================================================
FROM base as runtime

# Copy wheels from builder
COPY --from=builder /build/wheels /tmp/wheels

# Install application
RUN pip install --find-links /tmp/wheels liquid-edge-lln && \
    rm -rf /tmp/wheels

# Switch to non-root user
USER appuser
WORKDIR /home/appuser

# Copy application files
COPY --chown=appuser:appuser examples/ ./examples/
COPY --chown=appuser:appuser scripts/ ./scripts/

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import liquid_edge; print('Container healthy')" || exit 1

# Default command
CMD ["liquid-lln", "--help"]

# ============================================================================
# Development stage - Full development environment
# ============================================================================
FROM builder as development

# Install development dependencies
RUN pip install -e ".[dev,deployment,ros2,monitoring,docs]"

# Install additional development tools
RUN pip install \
    ipython \
    jupyter \
    matplotlib \
    seaborn

# Switch to non-root user
USER appuser
WORKDIR /workspace

# Expose ports for development
EXPOSE 8888 8000 3000

CMD ["bash"]

# ============================================================================
# Testing stage - Optimized for CI/CD
# ============================================================================
FROM development as testing

# Copy all source code
COPY --chown=appuser:appuser . /workspace/

# Install pre-commit hooks
RUN pre-commit install

# Run tests by default
CMD ["python", "-m", "pytest", "tests/", "-v"]

# ============================================================================
# Documentation stage - For building docs
# ============================================================================
FROM development as docs

# Install documentation dependencies
RUN pip install -e ".[docs]"

# Copy documentation source
COPY --chown=appuser:appuser docs/ /workspace/docs/

WORKDIR /workspace/docs

# Expose documentation port
EXPOSE 8080

CMD ["python", "-m", "http.server", "8080"]