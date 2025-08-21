# Production Dockerfile for Liquid Edge LLN Kit
FROM python:3.10-slim

# Set production environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV ENVIRONMENT=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash liquid

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/
COPY examples/ ./examples/
COPY *.py ./

# Set ownership
RUN chown -R liquid:liquid /app
USER liquid

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python3 -c "import src.liquid_edge; print('OK')" || exit 1

# Expose port
EXPOSE 8080

# Default command
CMD ["python3", "scaled_generation3_demo.py"]
