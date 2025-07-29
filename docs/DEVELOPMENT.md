# Development Guide

## Getting Started

### Prerequisites
- Python 3.10+
- Git
- ARM toolchain (for MCU deployment)
- ESP-IDF (for ESP32 support)

### Initial Setup

```bash
# Clone repository
git clone https://github.com/liquid-edge/liquid-edge-lln-kit.git
cd liquid-edge-lln-kit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
python -c "import liquid_edge; print('Success!')"
```

### Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Write code with tests**
   ```bash
   # Add functionality
   vim src/liquid_edge/your_module.py
   
   # Add tests
   vim tests/test_your_module.py
   ```

3. **Run quality checks**
   ```bash
   # Format code
   black src/ tests/
   
   # Lint code
   ruff check src/ tests/
   
   # Type checking
   mypy src/
   
   # Run tests
   pytest
   ```

4. **Commit and push**
   ```bash
   git add .
   git commit -m "feat: add your feature description"
   git push origin feature/your-feature-name
   ```

## Testing Strategy

### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Focus on edge cases and error conditions

### Integration Tests
- Test component interactions
- Use real JAX computations
- Validate mathematical correctness

### Hardware Tests
- Deploy to real MCU hardware
- Measure actual energy consumption
- Validate real-time performance

### Example Test Structure
```python
import pytest
import jax.numpy as jnp
from liquid_edge import LiquidNN, LiquidConfig

def test_liquid_nn_forward_pass():
    config = LiquidConfig(input_dim=4, hidden_dim=8, output_dim=2)
    model = LiquidNN(config)
    
    # Test forward pass
    key = jax.random.PRNGKey(0)
    params = model.init(key, jnp.ones((1, 4)))
    output = model.apply(params, jnp.ones((1, 4)))
    
    assert output.shape == (1, 2)
    assert jnp.all(jnp.isfinite(output))
```

## Code Organization

```
src/liquid_edge/
├── core/           # Core neural network implementation
├── layers/         # Individual layer implementations  
├── deploy/         # Hardware deployment tools
├── training/       # Training utilities
├── integrations/   # ROS 2, sensor integrations
└── utils/          # Shared utilities

tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
└── hardware/       # Hardware-specific tests

docs/
├── api/            # API documentation
├── tutorials/      # Step-by-step guides
├── examples/       # Code examples
└── deployment/     # Deployment guides
```

## Performance Guidelines

### Memory Usage
- Prefer in-place operations
- Use JAX's memory-efficient primitives
- Profile memory allocation patterns
- Target <256KB for MCU deployment

### Computation
- Leverage JAX JIT compilation
- Use vectorized operations
- Minimize Python loops
- Profile with JAX tools

### Energy Optimization
- Sparse connectivity patterns
- Quantized arithmetic where possible
- Adaptive computation based on input
- Hardware-specific optimizations

## Debugging Tips

### JAX Debugging
```python
# Enable debugging
import jax
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_disable_jit", True)  # For debugging

# Use JAX debugging tools
from jax import debug
debug.print("Debug value: {}", x)
```

### Hardware Debugging
- Use UART for logging on MCU
- LED indicators for basic status
- Logic analyzer for timing analysis
- Energy profiler for power measurement

## Contributing Guidelines

1. **Code Style**: Follow Black and Ruff configurations
2. **Documentation**: Update docstrings and docs/
3. **Tests**: Maintain >90% test coverage
4. **Performance**: No performance regressions
5. **Compatibility**: Support Python 3.10+

## Release Process

1. Update version in `pyproject.toml`
2. Update CHANGELOG.md
3. Create release branch
4. Run full test suite
5. Tag release and push
6. Build and upload to PyPI