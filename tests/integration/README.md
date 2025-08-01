# Integration Tests

This directory contains integration tests that verify the interaction between different components of the Liquid Edge LLN Kit.

## Test Categories

### Model Integration
- JAX to deployment pipeline
- Quantization accuracy preservation
- Cross-platform model compatibility

### Hardware Integration
- CMSIS-NN code generation
- ESP-NN optimization verification
- MCU deployment end-to-end

### ROS 2 Integration
- Liquid controller node functionality
- Multi-sensor fusion pipelines
- Real-time performance validation

### Energy Integration
- Profiling accuracy validation
- Energy budget enforcement
- Hardware measurement correlation

## Running Integration Tests

```bash
# Run all integration tests
pytest tests/integration/ -v

# Run specific integration suite
pytest tests/integration/test_deployment_pipeline.py -v

# Run with hardware (requires physical devices)
pytest tests/integration/ -m hardware -v

# Skip slow integration tests
pytest tests/integration/ -m "not slow" -v
```

## Test Environment

Integration tests may require:
- Docker for containerized testing
- Hardware devices for hardware tests
- ROS 2 environment for robotics tests
- Specific Python environments for cross-version testing