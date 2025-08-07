"""Pytest configuration and fixtures for Liquid Edge LLN Kit tests."""

import pytest
import jax
import jax.numpy as jnp
from liquid_edge import LiquidConfig


@pytest.fixture
def rng_key():
    """Provide a JAX random key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def basic_config():
    """Provide a basic liquid network configuration."""
    return LiquidConfig(
        input_dim=4,
        hidden_dim=8,
        output_dim=2,
        tau_min=10.0,
        tau_max=100.0,
        sensory_sigma=0.1,
        sensory_mu=0.3,
        learning_rate=0.001
    )


@pytest.fixture
def sample_input():
    """Provide sample input data for testing."""
    return jnp.ones((1, 4))


@pytest.fixture
def sample_batch():
    """Provide sample batch data for testing."""
    return jnp.ones((10, 4))


@pytest.fixture
def sensor_data():
    """Simulated sensor data sequence."""
    return jnp.array([
        [1.0, 0.5, -0.2, 0.8],
        [0.9, 0.6, -0.1, 0.7],
        [0.8, 0.7, 0.0, 0.6],
        [0.7, 0.8, 0.1, 0.5],
    ])


@pytest.fixture
def motor_targets():
    """Target motor commands."""
    return jnp.array([
        [0.5, -0.3],
        [0.4, -0.2],
        [0.3, -0.1],
        [0.2, 0.0],
    ])


def pytest_configure(config):
    """Configure test markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "hardware: requires physical hardware")
    config.addinivalue_line("markers", "benchmark: performance benchmarks")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers."""
    for item in items:
        # Mark slow tests
        if "slow" in item.nodeid or any(
            marker in item.name.lower() 
            for marker in ["integration", "benchmark", "hardware"]
        ):
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in item.nodeid or "test_integration" in item.name:
            item.add_marker(pytest.mark.integration)
        
        # Mark hardware tests
        if "hardware" in item.nodeid or "test_hardware" in item.name:
            item.add_marker(pytest.mark.hardware)
        
        # Mark benchmark tests
        if "benchmark" in item.nodeid or "test_benchmark" in item.name:
            item.add_marker(pytest.mark.benchmark)


@pytest.fixture(scope="session")
def jax_config():
    """Configure JAX for testing."""
    # Enable debugging for tests
    jax.config.update("jax_debug_nans", True)
    # Use CPU for consistent testing
    jax.config.update("jax_platform_name", "cpu")
    # Disable JIT for faster test execution
    jax.config.update('jax_disable_jit', True)
    yield
    # Reset to defaults after tests
    jax.config.update('jax_disable_jit', False)