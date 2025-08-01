# Test Fixtures

This directory contains test data, model fixtures, and other resources used by the test suite.

## Structure

```
fixtures/
├── models/           # Pre-trained liquid neural network models for testing
├── data/            # Sample datasets and sensor data
├── configs/         # Test configuration files
├── hardware/        # Hardware simulation data
└── expected/        # Expected outputs for regression tests
```

## Usage

Test fixtures can be loaded using the utilities in `conftest.py`:

```python
import pytest

def test_liquid_model_inference(liquid_model_fixture):
    """Test using a pre-loaded liquid model fixture."""
    model, params = liquid_model_fixture
    # Test inference...

def test_sensor_data_processing(sensor_data_fixture):
    """Test using sample sensor data."""
    imu_data, camera_data = sensor_data_fixture
    # Test processing...
```

## Adding New Fixtures

1. Place data files in appropriate subdirectory
2. Add loading logic to `conftest.py`
3. Document the fixture in this README
4. Ensure data is small (<1MB per file) to avoid bloating repository