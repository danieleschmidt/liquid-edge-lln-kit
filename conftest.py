# Copyright (c) 2025 Liquid Edge LLN Kit Contributors
# SPDX-License-Identifier: MIT

"""Global pytest configuration and fixtures for Liquid Edge LLN Kit."""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import random

# Ensure reproducible tests
os.environ["JAX_ENABLE_X64"] = "false"
jax.config.update("jax_enable_x64", False)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Get the test data directory."""
    return Path(__file__).parent / "tests" / "fixtures"


@pytest.fixture(scope="session")
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def rng_key():
    """Provide a JAX random key for tests."""
    return random.PRNGKey(42)


@pytest.fixture
def liquid_config():
    """Provide a standard liquid neural network configuration."""
    from liquid_edge.core import LiquidConfig
    
    return LiquidConfig(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        tau_min=1.0,
        tau_max=100.0,
        use_sparse=True,
        sparsity=0.3
    )


@pytest.fixture
def liquid_model_fixture(liquid_config, rng_key):
    """Provide a pre-initialized liquid model with parameters."""
    from liquid_edge.core import LiquidNN
    
    model = LiquidNN(liquid_config)
    params = model.init(rng_key, jnp.ones((1, liquid_config.input_dim)))
    return model, params


@pytest.fixture
def sensor_data_fixture():
    """Provide sample sensor data for testing."""
    # IMU data (accelerometer + gyroscope)
    imu_data = {
        "accel": np.random.normal(0, 1, (100, 3)),  # 100 samples, 3 axes
        "gyro": np.random.normal(0, 0.1, (100, 3)),
        "timestamp": np.linspace(0, 1, 100)
    }
    
    # Camera features (compressed)
    camera_data = {
        "features": np.random.normal(0, 1, (100, 64)),  # 64D feature vector
        "timestamp": np.linspace(0, 1, 100)
    }
    
    return imu_data, camera_data


@pytest.fixture
def training_data_fixture(liquid_config):
    """Provide sample training data."""
    n_samples = 1000
    sequence_length = 50
    
    # Generate synthetic sensor inputs
    inputs = np.random.normal(0, 1, (n_samples, sequence_length, liquid_config.input_dim))
    
    # Generate synthetic motor commands
    targets = np.random.uniform(-1, 1, (n_samples, sequence_length, liquid_config.output_dim))
    
    return jnp.array(inputs), jnp.array(targets)


@pytest.fixture
def energy_measurement_mock():
    """Mock energy measurement interface for testing."""
    class MockEnergyMeter:
        def __init__(self):
            self.measurements = []
            self.is_measuring = False
        
        def start_measurement(self):
            self.is_measuring = True
            self.measurements = []
        
        def stop_measurement(self):
            self.is_measuring = False
            # Return fake measurements
            return {
                "energy_mj": np.random.uniform(50, 150),
                "power_mw": np.random.uniform(80, 120),
                "duration_ms": np.random.uniform(8, 12)
            }
        
        def get_instant_power(self):
            return np.random.uniform(80, 120)
    
    return MockEnergyMeter()


@pytest.fixture
def mcu_simulator():
    """Provide MCU simulator for hardware-less testing."""
    class MCUSimulator:
        def __init__(self, platform="stm32h743"):
            self.platform = platform
            self.flash_memory = {}
            self.ram_usage = 0
        
        def flash_model(self, model_data):
            self.flash_memory["model"] = model_data
            return True
        
        def run_inference(self, input_data):
            # Simulate inference delay
            import time
            time.sleep(0.008)  # 8ms simulation
            
            # Return mock output
            return np.random.uniform(-1, 1, 2)
        
        def get_memory_usage(self):
            return {
                "flash_used": len(str(self.flash_memory)),
                "ram_used": self.ram_usage,
                "flash_total": 2048 * 1024,  # 2MB flash
                "ram_total": 512 * 1024      # 512KB RAM
            }
    
    return MCUSimulator()


@pytest.fixture(params=["stm32h743", "esp32s3", "nrf52840"])
def target_platform(request):
    """Parametrize tests across different target platforms."""
    return request.param


@pytest.fixture
def ros2_environment():
    """Mock ROS 2 environment for testing."""
    try:
        import rclpy
        # Real ROS 2 environment
        rclpy.init()
        yield {"type": "real", "rclpy": rclpy}
        rclpy.shutdown()
    except ImportError:
        # Mock environment
        class MockROS2:
            def init(self):
                pass
            
            def shutdown(self):
                pass
            
            def create_node(self, name):
                return MockNode(name)
        
        class MockNode:
            def __init__(self, name):
                self.name = name
            
            def create_publisher(self, msg_type, topic, qos):
                return MockPublisher()
            
            def create_subscription(self, msg_type, topic, callback, qos):
                return MockSubscription()
        
        class MockPublisher:
            def publish(self, msg):
                pass
        
        class MockSubscription:
            pass
        
        yield {"type": "mock", "rclpy": MockROS2()}


# Test markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "e2e: mark test as an end-to-end test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow (>30 seconds)"
    )
    config.addinivalue_line(
        "markers", "hardware: mark test as requiring physical hardware"
    )
    config.addinivalue_line(
        "markers", "benchmark: mark test as a performance benchmark"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on file location."""
    for item in items:
        # Add markers based on file path
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        elif "benchmark" in str(item.fspath):
            item.add_marker(pytest.mark.benchmark)


# Skip hardware tests if hardware not available
def pytest_runtest_setup(item):
    """Skip hardware tests if hardware not available."""
    if "hardware" in item.keywords:
        hardware_available = os.getenv("HARDWARE_TESTING_ENABLED", "false").lower() == "true"
        if not hardware_available:
            pytest.skip("Hardware testing not enabled (set HARDWARE_TESTING_ENABLED=true)")


# Performance test utilities
@pytest.fixture
def performance_monitor():
    """Monitor performance metrics during test execution."""
    import psutil
    import time
    
    class PerformanceMonitor:
        def __init__(self):
            self.start_time = None
            self.start_memory = None
            self.peak_memory = 0
        
        def start(self):
            self.start_time = time.time()
            self.start_memory = psutil.Process().memory_info().rss
            self.peak_memory = self.start_memory
        
        def update(self):
            current_memory = psutil.Process().memory_info().rss
            self.peak_memory = max(self.peak_memory, current_memory)
        
        def stop(self):
            end_time = time.time()
            return {
                "duration_s": end_time - self.start_time,
                "peak_memory_mb": self.peak_memory / (1024 * 1024),
                "memory_increase_mb": (self.peak_memory - self.start_memory) / (1024 * 1024)
            }
    
    return PerformanceMonitor()


# Load test configuration
@pytest.fixture(scope="session")
def test_config():
    """Load test configuration from environment and files."""
    config = {
        "hardware_enabled": os.getenv("HARDWARE_TESTING_ENABLED", "false").lower() == "true",
        "skip_slow": os.getenv("SKIP_SLOW_TESTS", "false").lower() == "true",
        "energy_interface": os.getenv("POWER_MEASUREMENT_INTERFACE", "simulation"),
        "serial_port": os.getenv("SERIAL_PORT", "/dev/ttyUSB0"),
        "parallel_workers": int(os.getenv("PYTEST_XDIST_WORKERS", "auto") if os.getenv("PYTEST_XDIST_WORKERS") != "auto" else 4)
    }
    return config