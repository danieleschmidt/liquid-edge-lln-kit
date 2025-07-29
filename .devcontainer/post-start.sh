#!/bin/bash
# Post-start script for Liquid Edge LLN Kit development container

set -e

echo "🌟 Starting Liquid Edge LLN Kit development session..."

# Activate the conda environment
source /opt/conda/bin/activate liquid-edge-dev

# Set up environment variables
export PYTHONPATH="/workspace/src:$PYTHONPATH"
export LIQUID_EDGE_ENV="development"
export WANDB_MODE="offline"  # Default to offline mode for privacy

# Check if this is the first run
if [ ! -f /workspace/.dev_initialized ]; then
    echo "🎯 First-time setup detected, running additional initialization..."
    
    # Create .env file for development
    cat > /workspace/.env << EOF
# Liquid Edge Development Environment Variables
LIQUID_EDGE_ENV=development
PYTHONPATH=/workspace/src
WANDB_MODE=offline
CUDA_VISIBLE_DEVICES=0
LOG_LEVEL=DEBUG
MONITORING_ENABLED=true
SIMULATION_MODE=true
EOF
    
    # Mark as initialized
    touch /workspace/.dev_initialized
    echo "✅ First-time initialization completed"
fi

# Start background services if needed
echo "🔧 Starting development services..."

# Check if Jupyter should auto-start (optional)
if [ "${AUTO_START_JUPYTER:-false}" = "true" ]; then
    echo "🪐 Starting Jupyter Lab..."
    nohup jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser \
        --notebook-dir=/workspace > /workspace/logs/jupyter.log 2>&1 &
    echo "✅ Jupyter Lab started on port 8888"
fi

# Check if development server should auto-start (optional)
if [ "${AUTO_START_SERVER:-false}" = "true" ]; then
    echo "🚀 Starting development server..."
    nohup liquid-lln serve --config examples/configs/dev_config.yaml --debug \
        > /workspace/logs/dev_server.log 2>&1 &
    echo "✅ Development server started on port 8000"
fi

# Display helpful information
echo ""
echo "🎉 Development environment is ready!"
echo ""
echo "📋 Quick commands:"
echo "  liquid-dev       - Navigate to workspace and activate environment"
echo "  liquid-test      - Run test suite"
echo "  liquid-serve     - Start development server"
echo "  liquid-jupyter   - Start Jupyter Lab"
echo "  liquid-benchmark - Run performance benchmarks"
echo ""
echo "📁 Key directories:"
echo "  /workspace/src/          - Source code"
echo "  /workspace/examples/     - Examples and tutorials"
echo "  /workspace/data/         - Development data"
echo "  /workspace/models/       - Model files"
echo ""
echo "🔍 Monitoring:"
if [ "${AUTO_START_JUPYTER:-false}" = "true" ]; then
    echo "  Jupyter Lab: http://localhost:8888"
fi
if [ "${AUTO_START_SERVER:-false}" = "true" ]; then
    echo "  Dev Server:  http://localhost:8000"
    echo "  Health:      http://localhost:8081/health"
    echo "  Metrics:     http://localhost:8080/metrics"
fi
echo ""

# Check system resources
echo "💾 System resources:"
echo "  Memory: $(free -h | awk 'NR==2{printf \"%.1fG/%.1fG (%.0f%%)\", $3/1024/1024, $2/1024/1024, $3*100/$2}')"
echo "  Disk:   $(df -h /workspace | awk 'NR==2{printf \"%s/%s (%s)\", $3, $2, $5}')"
echo "  CPU:    $(nproc) cores available"

# Check for GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "  GPU:    $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
else
    echo "  GPU:    Not available (CPU-only development)"
fi

echo ""
echo "Happy coding! 🚀"