# Production Dockerfile for Liquid Edge LLN
FROM python:3.12-slim as builder

# Build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    cmake \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.12-slim

# Security: Create non-root user
RUN groupadd -r liquiduser && useradd -r -g liquiduser liquiduser

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libblas3 \
    liblapack3 \
    && rm -rf /var/lib/apt/lists/*

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
WORKDIR /app
COPY src/ ./src/
COPY quantum_leap_enhancement.py .
COPY production_robustness_system.py .
COPY quantum_scaling_system.py .

# Set security and performance optimizations
RUN chown -R liquiduser:liquiduser /app
USER liquiduser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import sys; sys.exit(0)"

# Expose port
EXPOSE 8080

# Production command with optimizations
CMD ["python3", "-O", "-u", "quantum_scaling_system.py"]
