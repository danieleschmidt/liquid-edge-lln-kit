# End-to-End Tests

This directory contains end-to-end tests that validate complete user workflows from training to deployment.

## Test Scenarios

### Training to Deployment Workflow
1. Train liquid neural network on sample data
2. Optimize for target hardware platform
3. Generate deployment code
4. Validate performance on simulated hardware

### ROS 2 Robot Workflow
1. Configure liquid controller for TurtleBot
2. Deploy to simulation environment
3. Validate navigation and obstacle avoidance
4. Measure energy consumption

### Multi-Platform Deployment
1. Train single model
2. Deploy to ARM Cortex-M, ESP32, and simulation
3. Validate consistent behavior across platforms
4. Compare performance metrics

## Running E2E Tests

```bash
# Run all end-to-end tests (slow)
pytest tests/e2e/ -v

# Run specific workflow
pytest tests/e2e/test_training_to_deployment.py -v

# Run with real hardware
pytest tests/e2e/ -m hardware --hardware-config hardware.yaml

# Run in Docker environment
docker-compose -f docker-compose.test.yml run e2e-tests
```

## Test Data

E2E tests use realistic datasets and configurations:
- Sample sensor traces from real robots
- Pre-trained baseline models for comparison
- Hardware configuration files
- Performance benchmarks

## CI/CD Integration

E2E tests are designed to run in continuous integration:
- Containerized execution
- Hardware simulation when physical devices unavailable
- Performance regression detection
- Deployment validation