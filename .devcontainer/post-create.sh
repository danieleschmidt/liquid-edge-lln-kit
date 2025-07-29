#!/bin/bash
# Post-create script for Liquid Edge LLN Kit development container

set -e

echo "🚀 Setting up Liquid Edge LLN Kit development environment..."

# Activate conda environment
source /opt/conda/bin/activate liquid-edge-dev

# Install the package in editable mode with all extras
echo "📦 Installing Liquid Edge LLN Kit in development mode..."
pip install -e ".[dev,deployment,ros2]"

# Set up pre-commit hooks
echo "🔧 Setting up pre-commit hooks..."
pre-commit install
pre-commit install --hook-type commit-msg

# Create development directories
echo "📁 Creating development directories..."
mkdir -p /workspace/{data,models,experiments,logs,artifacts,hardware_tests}
mkdir -p /workspace/data/{raw,processed,synthetic}
mkdir -p /workspace/models/{checkpoints,deployed,archives}
mkdir -p /workspace/experiments/{training,benchmarks,ablations}
mkdir -p /workspace/hardware_tests/{simulation,real_device}

# Download sample data and models
echo "📥 Setting up sample data and models..."
if [ ! -f "/workspace/data/sample_robot_data.npz" ]; then
    echo "Creating synthetic sample data..."
    python -c "
import numpy as np
import os

# Create synthetic robot sensor data
np.random.seed(42)
sensor_data = np.random.randn(1000, 4)  # 1000 samples, 4 sensors
motor_commands = np.random.randn(1000, 2)  # 1000 samples, 2 motors

np.savez('/workspace/data/sample_robot_data.npz',
         sensor_data=sensor_data,
         motor_commands=motor_commands)

print('✅ Sample data created')
"
fi

# Create example configuration files
echo "⚙️ Creating example configuration files..."
cat > /workspace/examples/configs/dev_config.yaml << EOF
# Development configuration for Liquid Edge LLN Kit
environment: development
debug: true

model:
  input_dim: 4
  hidden_dim: 16
  output_dim: 2
  tau_min: 10.0
  tau_max: 100.0
  use_sparse: true
  sparsity: 0.3

training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  validation_split: 0.2
  
logging:
  level: DEBUG
  structured: true
  
monitoring:
  enabled: true
  metrics_port: 8080
  wandb_project: liquid-edge-dev

hardware:
  simulation_mode: true
  target_device: generic
EOF

cat > /workspace/examples/configs/benchmark_config.yaml << EOF
# Benchmark configuration
benchmark:
  iterations: 1000
  batch_sizes: [1, 8, 16, 32]
  model_sizes: [8, 16, 32, 64]
  devices: [cpu]
  
profiling:
  memory: true
  energy: true
  latency: true
  
output:
  format: json
  directory: /workspace/experiments/benchmarks
EOF

# Set up Jupyter kernels
echo "📓 Setting up Jupyter kernels..."
python -m ipykernel install --user --name liquid-edge-dev --display-name "Liquid Edge Development"

# Create useful aliases
echo "🔗 Setting up development aliases..."
cat >> ~/.bashrc << EOF

# Liquid Edge Development Aliases
alias liquid-dev='cd /workspace && source /opt/conda/bin/activate liquid-edge-dev'
alias liquid-test='pytest tests/ -v'
alias liquid-lint='ruff check src/ tests/'
alias liquid-format='black src/ tests/ examples/'
alias liquid-serve='liquid-lln serve --config examples/configs/dev_config.yaml --debug'
alias liquid-jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser'
alias liquid-benchmark='python scripts/benchmark_performance.py'
alias liquid-security='python scripts/security_scan.py'

# Useful development shortcuts
alias ll='ls -la'
alias la='ls -A'
alias l='ls -CF'
alias grep='grep --color=auto'
alias ..='cd ..'
alias ...='cd ../..'

# Git shortcuts
alias gs='git status'
alias ga='git add'
alias gc='git commit'
alias gp='git push'
alias gl='git log --oneline'
alias gd='git diff'

EOF

# Set up zsh if available
if [ -f ~/.zshrc ]; then
    cat >> ~/.zshrc << EOF

# Liquid Edge Development Aliases
alias liquid-dev='cd /workspace && source /opt/conda/bin/activate liquid-edge-dev'
alias liquid-test='pytest tests/ -v'
alias liquid-lint='ruff check src/ tests/'
alias liquid-format='black src/ tests/ examples/'
alias liquid-serve='liquid-lln serve --config examples/configs/dev_config.yaml --debug'
alias liquid-jupyter='jupyter lab --ip=0.0.0.0 --port=8888 --allow-root --no-browser'
alias liquid-benchmark='python scripts/benchmark_performance.py'
alias liquid-security='python scripts/security_scan.py'

EOF
fi

# Initialize git hooks
echo "🔄 Initializing git configuration..."
git config --global --add safe.directory /workspace
git config --global init.defaultBranch main
git config --global pull.rebase false

# Create example scripts
echo "📝 Creating example development scripts..."
mkdir -p /workspace/examples/scripts

cat > /workspace/examples/scripts/quick_train.py << 'EOF'
#!/usr/bin/env python3
"""Quick training script for development and testing."""

import jax
import jax.numpy as jnp
import numpy as np
from liquid_edge import LiquidNN, LiquidConfig

def main():
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(1000, 4).astype(np.float32)
    y = np.random.randn(1000, 2).astype(np.float32)
    
    # Create model
    config = LiquidConfig(
        input_dim=4,
        hidden_dim=16,
        output_dim=2,
        tau_min=10.0,
        tau_max=100.0
    )
    
    model = LiquidNN(config)
    
    # Initialize parameters
    key = jax.random.PRNGKey(0)
    params = model.init(key, X[:1])
    
    print(f"✅ Model initialized with {sum(x.size for x in jax.tree_util.tree_leaves(params))} parameters")
    
    # Quick training loop
    import optax
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    
    @jax.jit
    def train_step(params, opt_state, batch_x, batch_y):
        def loss_fn(params):
            pred = model.apply(params, batch_x)
            return jnp.mean((pred - batch_y) ** 2)
        
        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    
    # Training
    batch_size = 32
    for epoch in range(10):
        for i in range(0, len(X), batch_size):
            batch_x = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            params, opt_state, loss = train_step(params, opt_state, batch_x, batch_y)
        
        print(f"Epoch {epoch+1}/10, Loss: {loss:.4f}")
    
    print("🎉 Quick training completed!")

if __name__ == "__main__":
    main()
EOF

chmod +x /workspace/examples/scripts/quick_train.py

# Create development README
cat > /workspace/DEVELOPMENT.md << 'EOF'
# Development Environment

Welcome to the Liquid Edge LLN Kit development environment! This container is set up with all the tools you need for development, testing, and deployment.

## Quick Start

```bash
# Activate development environment
liquid-dev

# Run tests
liquid-test

# Start development server
liquid-serve

# Start Jupyter Lab
liquid-jupyter
```

## Available Tools

- **Python 3.11** with conda environment management
- **JAX/Flax** for neural network development
- **Jupyter Lab** for interactive development
- **VS Code** extensions for Python, embedded development
- **ARM cross-compilation** tools for MCU deployment
- **ESP-IDF** for ESP32 development
- **PlatformIO** for embedded systems
- **Hardware simulation** with QEMU

## Directory Structure

```
/workspace/
├── src/                    # Source code
├── tests/                  # Test files
├── examples/               # Example scripts and configs
├── data/                   # Development data
├── models/                 # Model files and checkpoints
├── experiments/            # Experiment results
├── hardware_tests/         # Hardware testing
└── docs/                   # Documentation
```

## Development Workflow

1. **Code**: Edit files in VS Code with full IntelliSense
2. **Test**: Run `liquid-test` for comprehensive testing
3. **Format**: Use `liquid-format` to format code
4. **Lint**: Use `liquid-lint` to check code quality
5. **Commit**: Pre-commit hooks ensure quality
6. **Deploy**: Test on simulated hardware first

## Hardware Development

- **MCU Simulation**: Use QEMU for ARM Cortex-M testing
- **ESP32**: Full ESP-IDF environment available
- **Cross-compilation**: ARM GCC toolchain included
- **Debugging**: GDB and OpenOCD for hardware debugging

## Useful Aliases

- `liquid-dev`: Activate development environment
- `liquid-test`: Run test suite
- `liquid-serve`: Start development server
- `liquid-jupyter`: Launch Jupyter Lab
- `liquid-benchmark`: Run performance benchmarks
- `liquid-security`: Run security scans

Happy coding! 🚀
EOF

# Set file permissions
chmod +x /workspace/.devcontainer/post-create.sh
chmod +x /workspace/.devcontainer/post-start.sh 2>/dev/null || true

# Run initial tests to verify setup
echo "🧪 Running initial setup verification..."
if python -c "import jax; import liquid_edge; print('✅ Core imports successful')"; then
    echo "✅ Development environment setup completed successfully!"
else
    echo "⚠️  Warning: Some imports failed, but continuing setup..."
fi

echo ""
echo "🎉 Liquid Edge LLN Kit development environment is ready!"
echo ""
echo "Next steps:"
echo "  1. Open VS Code and start coding"
echo "  2. Run 'liquid-test' to verify everything works"
echo "  3. Try 'python examples/scripts/quick_train.py' for a quick test"
echo "  4. Launch Jupyter with 'liquid-jupyter' for interactive development"
echo ""
echo "Happy coding! 🚀"