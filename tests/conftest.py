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
        use_sparse=True,
        sparsity=0.3
    )


@pytest.fixture
def sample_input():
    """Provide sample input data for testing."""
    return jnp.ones((1, 4))


@pytest.fixture
def sample_batch():
    """Provide sample batch data for testing."""
    return jnp.ones((10, 4))


@pytest.fixture(scope="session")
def jax_config():
    """Configure JAX for testing."""
    # Enable debugging for tests
    jax.config.update("jax_debug_nans", True)
    # Use CPU for consistent testing
    jax.config.update("jax_platform_name", "cpu")
    yield
    # Reset to defaults after tests