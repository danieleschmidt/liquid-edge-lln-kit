# Testing Documentation

This directory contains comprehensive testing documentation for the Liquid Edge LLN Kit.

## Testing Strategy

### Test Pyramid
```
    /\
   /  \     E2E Tests (Few, Slow, High Confidence)
  /____\    Integration Tests (Some, Medium Speed)  
 /______\   Unit Tests (Many, Fast, Low Level)
/________\  Static Analysis (Linting, Type Checking)
```

### Test Categories

1. **Unit Tests** (`tests/unit/`)
   - Individual function and class testing
   - Mock dependencies
   - Fast execution (<1s per test)
   - 90%+ code coverage target

2. **Integration Tests** (`tests/integration/`)
   - Component interaction testing
   - Real dependencies where practical
   - Medium execution time (<30s per test)
   - Focus on interface correctness

3. **End-to-End Tests** (`tests/e2e/`)
   - Complete workflow validation
   - Real or simulated hardware
   - Slow execution (minutes per test)
   - User story validation

4. **Performance Tests** (`tests/benchmarks/`)
   - Energy consumption measurement
   - Inference latency benchmarking
   - Memory usage profiling
   - Regression detection

## Test Execution

### Local Development
```bash
# Fast development cycle
make test-unit

# Full test suite
make test

# With coverage report
make test-coverage

# Performance benchmarks
make test-benchmark
```

### Continuous Integration
```bash
# Parallel execution
tox -p auto

# Specific environment
tox -e py310,lint,type

# Hardware tests (if available)
tox -e hardware
```

## Test Markers

Use pytest markers to categorize tests:

```python
@pytest.mark.unit
def test_liquid_cell_forward():
    """Unit test for liquid cell forward pass."""
    pass

@pytest.mark.integration
def test_model_deployment_pipeline():
    """Integration test for deployment pipeline."""
    pass

@pytest.mark.slow
def test_long_training_session():
    """Slow test that takes >30 seconds."""
    pass

@pytest.mark.hardware
def test_real_mcu_deployment():
    """Test requiring physical hardware."""
    pass

@pytest.mark.benchmark
def test_inference_performance():
    """Performance benchmark test."""
    pass
```

## Hardware Testing

### Simulation vs Real Hardware

**Simulation** (Default):
- Fast execution
- Deterministic results
- No hardware dependencies
- Limited accuracy validation

**Real Hardware** (Optional):
- Accurate performance metrics
- Real-world validation
- Hardware dependencies
- Slower execution

### Hardware Test Configuration

Create `hardware.yaml` for real hardware tests:

```yaml
devices:
  stm32h743:
    port: /dev/ttyUSB0
    programmer: stlink
    enabled: true
  
  esp32s3:
    port: /dev/ttyUSB1
    programmer: esptool
    enabled: false

energy_measurement:
  interface: ina219
  i2c_address: 0x40
  shunt_resistance: 0.1
```

## Test Data Management

### Fixtures
- Small test data in `tests/fixtures/`
- Large datasets via download scripts
- Version controlled expected outputs
- Automated data generation where possible

### Reproducibility
- Fixed random seeds
- Deterministic algorithms
- Version pinned dependencies
- Containerized execution environments

## Quality Gates

### Code Coverage
- Minimum 90% coverage
- Focus on critical paths
- Exclude trivial code (getters/setters)
- Line and branch coverage

### Performance Benchmarks
- Energy consumption regression <5%
- Inference latency regression <10%
- Memory usage regression <5%
- Accuracy degradation <1%

### Static Analysis
- No linting errors
- Type checking with mypy
- Security scanning with bandit
- Dependency vulnerability scanning

## Writing Good Tests

### Best Practices
- **Arrange-Act-Assert** pattern
- **Single responsibility** per test
- **Descriptive test names** that explain the scenario
- **Independent tests** that don't rely on each other
- **Fast feedback** - optimize for quick execution

### Example Test Structure
```python
def test_liquid_cell_with_time_constant_adaptation():
    """Test that liquid cell adapts time constants during training."""
    # Arrange
    config = LiquidConfig(hidden_dim=8, tau_min=1.0, tau_max=100.0)
    cell = LiquidCell(config)
    initial_params = cell.init(PRNGKey(42), jnp.ones((1, 4)))
    
    # Act
    trained_params = train_liquid_cell(cell, initial_params, training_data)
    
    # Assert
    initial_tau = initial_params['tau']
    trained_tau = trained_params['tau'] 
    assert not jnp.allclose(initial_tau, trained_tau), "Time constants should adapt"
    assert jnp.all(trained_tau >= config.tau_min), "Tau should be >= tau_min"
    assert jnp.all(trained_tau <= config.tau_max), "Tau should be <= tau_max"
```

## Debugging Tests

### Common Issues
- **Flaky tests**: Use fixed seeds, avoid timing dependencies
- **Slow tests**: Profile and optimize, consider mocking
- **Hardware failures**: Check connections, verify configurations
- **Environment issues**: Use containerization, pin dependencies

### Debugging Tools
```bash
# Run single test with verbose output
pytest tests/test_core.py::test_liquid_forward -v -s

# Debug test with pdb
pytest tests/test_core.py::test_liquid_forward --pdb

# Profile test performance
pytest tests/test_core.py --profile-svg

# Check test coverage
pytest tests/test_core.py --cov=liquid_edge --cov-report=html
```